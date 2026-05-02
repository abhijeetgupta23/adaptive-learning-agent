from __future__ import annotations

from pydantic import BaseModel

from backend.schemas.assessment import Verdict


class AssessmentAccuracyResult(BaseModel):
    case_id: str
    predicted: Verdict
    expected: Verdict
    passed: bool
    rationale: str


_PASS_SET = {Verdict.PASS}
_FAIL_SET = {Verdict.NEEDS_REINFORCEMENT, Verdict.NEEDS_DIFFERENT_APPROACH}


def evaluate_assessment_accuracy(
    *,
    case_id: str,
    predicted: Verdict,
    expected: Verdict,
) -> AssessmentAccuracyResult:
    """Deterministic check on assessment verdict vs human gold.

    Strict pass = exact match. Near-miss (both in fail-family but different) still fails
    but is noted in rationale since it's less bad than crossing the pass/fail boundary.
    """
    passed = predicted == expected
    if passed:
        rationale = f"Exact match on {expected.value}."
    elif predicted in _PASS_SET and expected in _FAIL_SET:
        rationale = f"False pass: predicted {predicted.value}, expected {expected.value}."
    elif predicted in _FAIL_SET and expected in _PASS_SET:
        rationale = f"False fail: predicted {predicted.value}, expected {expected.value}."
    else:
        rationale = (
            f"Near-miss within fail family: predicted {predicted.value}, "
            f"expected {expected.value}."
        )
    return AssessmentAccuracyResult(
        case_id=case_id,
        predicted=predicted,
        expected=expected,
        passed=passed,
        rationale=rationale,
    )
