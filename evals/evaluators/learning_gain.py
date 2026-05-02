from __future__ import annotations

from pydantic import BaseModel, Field


class LearningGainResult(BaseModel):
    case_id: str
    pre_score: float = Field(ge=0.0, le=1.0)
    post_score: float = Field(ge=0.0, le=1.0)
    delta: float
    passed: bool
    rationale: str


def evaluate_learning_gain(
    *,
    case_id: str,
    pre_score: float,
    post_score: float,
    min_delta: float = 0.2,
) -> LearningGainResult:
    """Knowledge-delta check. Pre/post scores come from a structured quiz against a
    synthetic learner; we just measure the delta here.

    v1 stub: callers plug in scores from whatever quiz rubric they use. The
    synthetic-learner simulation itself is out of scope for v1 (Task 10+).
    """
    delta = post_score - pre_score
    passed = delta >= min_delta
    rationale = (
        f"pre={pre_score:.2f}, post={post_score:.2f}, delta={delta:+.2f} "
        f"(threshold {min_delta:+.2f}): {'PASS' if passed else 'FAIL'}"
    )
    return LearningGainResult(
        case_id=case_id,
        pre_score=pre_score,
        post_score=post_score,
        delta=delta,
        passed=passed,
        rationale=rationale,
    )
