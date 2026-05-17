"""Persistence helpers for the `session_runs` audit table.

The orchestrator inserts a row at session start (with `started_at` only),
then updates that row at session end (totals filled, `finished_at` set) or
on error (`error` filled). The `/ops/dashboard` endpoint reads aggregates
from this table — see `backend/api/ops.py`.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.persistence.models import SessionRunRow


async def start_session_run(
    *,
    session: AsyncSession,
    session_id: str,
    user_id: str,
    domain: str | None,
    sub_goal: str | None,
) -> int:
    """Insert a session_runs row at session start; return its PK."""
    row = SessionRunRow(
        session_id=session_id,
        user_id=user_id,
        domain=domain,
        sub_goal=sub_goal,
        started_at=datetime.now(UTC),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row.id


async def finish_session_run(
    *,
    session: AsyncSession,
    run_id: int,
    unit_count: int | None,
    units_passed: int | None,
    total_cost_usd: float,
    total_latency_s: float,
    cost_by_agent: dict[str, float],
    latency_by_agent: dict[str, list[float]],
    error: str | None = None,
) -> None:
    """Update the row with totals + finished_at. `error` is the failure
    message (or None on clean exit). latency_by_agent is the raw per-call
    list — kept verbatim so the dashboard can compute p50/p99 windows."""
    row = await session.get(SessionRunRow, run_id)
    if row is None:
        return  # row was deleted out from under us; nothing we can do
    row.unit_count = unit_count
    row.units_passed = units_passed
    row.total_cost_usd = total_cost_usd
    row.total_latency_s = total_latency_s
    row.cost_by_agent_json = json.dumps(cost_by_agent)
    row.latency_by_agent_json = json.dumps(latency_by_agent)
    row.finished_at = datetime.now(UTC)
    row.error = error
    await session.commit()


async def list_recent_completed(
    *, session: AsyncSession, limit: int = 10,
) -> list[SessionRunRow]:
    """Last N completed-or-errored sessions (finished_at IS NOT NULL),
    newest first."""
    stmt = (
        select(SessionRunRow)
        .where(SessionRunRow.finished_at.is_not(None))
        .order_by(SessionRunRow.finished_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_recent_errors(
    *, session: AsyncSession, limit: int = 10,
) -> list[SessionRunRow]:
    """Last N rows where error IS NOT NULL, newest first."""
    stmt = (
        select(SessionRunRow)
        .where(SessionRunRow.error.is_not(None))
        .order_by(SessionRunRow.finished_at.desc().nullslast())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_finished_within(
    *, session: AsyncSession, hours: float,
) -> list[SessionRunRow]:
    """All rows that finished within the last `hours`. Used for rolling
    24h aggregates on the dashboard."""
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    stmt = (
        select(SessionRunRow)
        .where(SessionRunRow.finished_at.is_not(None))
        .where(SessionRunRow.finished_at >= cutoff)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _safe_load(blob: str | None) -> Any:
    if not blob:
        return None
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_cost_by_agent(row: SessionRunRow) -> dict[str, float]:
    data = _safe_load(row.cost_by_agent_json)
    return data if isinstance(data, dict) else {}


def parse_latency_by_agent(row: SessionRunRow) -> dict[str, list[float]]:
    data = _safe_load(row.latency_by_agent_json)
    if not isinstance(data, dict):
        return {}
    # Coerce to list[float] — defensive against legacy rows.
    out: dict[str, list[float]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[k] = [float(x) for x in v if isinstance(x, int | float)]
    return out
