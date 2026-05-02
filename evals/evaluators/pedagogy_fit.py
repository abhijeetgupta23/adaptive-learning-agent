from __future__ import annotations

from pydantic import BaseModel

from backend.schemas.teaching import TeachingMethod


class PedagogyFitResult(BaseModel):
    case_id: str
    chosen: TeachingMethod
    acceptable: list[TeachingMethod]
    passed: bool
    rationale: str


def evaluate_pedagogy_fit(
    *,
    case_id: str,
    chosen: TeachingMethod,
    acceptable: list[TeachingMethod],
) -> PedagogyFitResult:
    """Deterministic check: did the router pick a method the gold set considers valid?"""
    passed = chosen in acceptable
    if passed:
        rationale = f"Router chose {chosen.value}, in acceptable set."
    else:
        rationale = (
            f"Router chose {chosen.value}, not in acceptable set "
            f"{[m.value for m in acceptable]}."
        )
    return PedagogyFitResult(
        case_id=case_id,
        chosen=chosen,
        acceptable=acceptable,
        passed=passed,
        rationale=rationale,
    )
