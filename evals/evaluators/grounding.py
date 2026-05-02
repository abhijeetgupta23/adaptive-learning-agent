from __future__ import annotations

import logging
import math
from typing import Any, Protocol

from pydantic import BaseModel, Field

from backend.schemas.curriculum import Curriculum

_logger = logging.getLogger(__name__)


class _HttpClient(Protocol):
    async def head(self, url: str, **kwargs: Any) -> Any: ...
    async def get(self, url: str, **kwargs: Any) -> Any: ...


class SourceGrounding(BaseModel):
    url: str
    reachable: bool | None = None  # None when not checked


class UnitGrounding(BaseModel):
    unit_id: str
    source_count: int
    sources: list[SourceGrounding]
    score: float = Field(ge=0.0, le=1.0)


class GroundingResult(BaseModel):
    per_unit: list[UnitGrounding]
    unique_source_count: int
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    rationale: str


async def check_grounding(
    curriculum: Curriculum,
    *,
    verify_reachable: bool = False,
    http_client: _HttpClient | None = None,
    timeout_s: float = 5.0,
) -> GroundingResult:
    """Evaluate structural + uniqueness grounding. Optionally verify URL reachability."""
    all_urls: list[str] = []
    per_unit: list[UnitGrounding] = []

    checked_urls: dict[str, bool] = {}
    if verify_reachable:
        unique = {str(s.url) for u in curriculum.units for s in u.grounding_sources}
        checked_urls = await _check_urls_reachable(list(unique), http_client, timeout_s)

    for unit in curriculum.units:
        source_records: list[SourceGrounding] = []
        for src in unit.grounding_sources:
            url = str(src.url)
            all_urls.append(url)
            reachable = checked_urls.get(url) if verify_reachable else None
            source_records.append(SourceGrounding(url=url, reachable=reachable))

        if verify_reachable:
            reachable_count = sum(1 for s in source_records if s.reachable)
            unit_score = reachable_count / max(len(source_records), 1)
        else:
            unit_score = 1.0 if source_records else 0.0

        per_unit.append(UnitGrounding(
            unit_id=unit.id,
            source_count=len(source_records),
            sources=source_records,
            score=unit_score,
        ))

    unique_source_count = len(set(all_urls))
    min_expected_unique = math.ceil(len(curriculum.units) / 2)
    diversity_ok = unique_source_count >= min_expected_unique

    mean_unit_score = (
        sum(u.score for u in per_unit) / len(per_unit) if per_unit else 0.0
    )
    overall = mean_unit_score * (1.0 if diversity_ok else 0.7)
    passed = overall >= 0.8 and all(u.score >= 0.5 for u in per_unit)

    rationale = (
        f"{len(curriculum.units)} units; {unique_source_count} unique sources "
        f"(expected >= {min_expected_unique}). "
        + ("All sources reachable." if verify_reachable and mean_unit_score == 1.0
           else f"Mean per-unit grounding score: {mean_unit_score:.2f}.")
    )

    _logger.info(
        "grounding_check",
        extra={
            "unit_count": len(curriculum.units),
            "unique_sources": unique_source_count,
            "passed": passed,
            "verify_reachable": verify_reachable,
        },
    )

    return GroundingResult(
        per_unit=per_unit,
        unique_source_count=unique_source_count,
        overall_score=overall,
        passed=passed,
        rationale=rationale,
    )


async def _check_urls_reachable(
    urls: list[str],
    client: _HttpClient | None,
    timeout_s: float,
) -> dict[str, bool]:
    import httpx

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=timeout_s)

    results: dict[str, bool] = {}
    try:
        for url in urls:
            try:
                resp = await client.head(url)
                status = getattr(resp, "status_code", 0)
                if status == 405:  # HEAD not allowed — retry GET
                    resp = await client.get(url)
                    status = getattr(resp, "status_code", 0)
                results[url] = 200 <= status < 400
            except Exception:
                results[url] = False
    finally:
        if owns_client:
            await client.aclose()  # type: ignore[attr-defined]

    return results
