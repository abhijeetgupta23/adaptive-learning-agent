from __future__ import annotations

import logging

from backend.schemas.assessment import AssessmentResult, Verdict
from backend.schemas.curriculum import LearningUnit
from backend.schemas.state import LearnerState
from backend.schemas.teaching import TeachingMethod

_logger = logging.getLogger(__name__)


def apply_assessment(
    state: LearnerState,
    *,
    unit: LearningUnit,
    method: TeachingMethod,
    result: AssessmentResult,
) -> LearnerState:
    """Return a new LearnerState reflecting the outcome of one teaching attempt."""
    data = state.model_dump()
    concept = unit.objective

    mastered = list(data["mastered_concepts"])
    struggled = list(data["struggled_concepts"])
    pedagogies = dict(data["effective_pedagogies"])

    if result.verdict is Verdict.PASS:
        if concept not in mastered:
            mastered.append(concept)
        # If concept was on the struggled list, it's no longer struggling.
        struggled = [c for c in struggled if c != concept]
        pedagogies[concept] = method
    else:
        if concept not in struggled:
            struggled.append(concept)

    data["mastered_concepts"] = mastered
    data["struggled_concepts"] = struggled
    data["effective_pedagogies"] = pedagogies

    updated = LearnerState.model_validate(data)
    _logger.info(
        "memory_update",
        extra={
            "user_id": state.user_id,
            "concept": concept,
            "verdict": result.verdict.value,
            "mastered_count": len(updated.mastered_concepts),
            "struggled_count": len(updated.struggled_concepts),
        },
    )
    return updated
