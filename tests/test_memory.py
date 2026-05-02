from backend.agents.memory import apply_assessment
from backend.schemas import (
    AssessmentResult,
    GroundingSource,
    LearnerState,
    LearningUnit,
    TeachingMethod,
    Verdict,
)


def _unit(objective: str = "Understand SQL joins") -> LearningUnit:
    return LearningUnit(
        id="u1",
        objective=objective,
        recommended_pedagogy=TeachingMethod.VISUAL,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )


def _result(verdict: Verdict) -> AssessmentResult:
    return AssessmentResult(
        unit_id="u1",
        verdict=verdict,
        diagnostic_notes="",
        confidence=0.9,
    )


def test_pass_records_mastery_and_effective_pedagogy() -> None:
    state = LearnerState(user_id="u")
    updated = apply_assessment(
        state, unit=_unit(), method=TeachingMethod.GAME, result=_result(Verdict.PASS),
    )
    assert updated.mastered_concepts == ["Understand SQL joins"]
    assert updated.effective_pedagogies == {"Understand SQL joins": TeachingMethod.GAME}
    assert updated.struggled_concepts == []


def test_fail_records_struggle_not_mastery() -> None:
    state = LearnerState(user_id="u")
    updated = apply_assessment(
        state, unit=_unit(), method=TeachingMethod.VISUAL,
        result=_result(Verdict.NEEDS_REINFORCEMENT),
    )
    assert updated.struggled_concepts == ["Understand SQL joins"]
    assert updated.mastered_concepts == []
    assert updated.effective_pedagogies == {}


def test_needs_different_approach_records_struggle() -> None:
    state = LearnerState(user_id="u")
    updated = apply_assessment(
        state, unit=_unit(), method=TeachingMethod.VISUAL,
        result=_result(Verdict.NEEDS_DIFFERENT_APPROACH),
    )
    assert "Understand SQL joins" in updated.struggled_concepts


def test_pass_removes_concept_from_struggled_list() -> None:
    state = LearnerState(user_id="u", struggled_concepts=["Understand SQL joins", "other"])
    updated = apply_assessment(
        state, unit=_unit(), method=TeachingMethod.GAME, result=_result(Verdict.PASS),
    )
    assert "Understand SQL joins" not in updated.struggled_concepts
    assert "other" in updated.struggled_concepts
    assert "Understand SQL joins" in updated.mastered_concepts


def test_apply_assessment_does_not_mutate_input() -> None:
    state = LearnerState(
        user_id="u",
        mastered_concepts=["prior"],
        effective_pedagogies={"prior": TeachingMethod.ANALOGY},
    )
    _ = apply_assessment(
        state, unit=_unit(), method=TeachingMethod.GAME, result=_result(Verdict.PASS),
    )
    # Original unchanged
    assert state.mastered_concepts == ["prior"]
    assert state.effective_pedagogies == {"prior": TeachingMethod.ANALOGY}


def test_repeat_pass_is_idempotent_on_mastery() -> None:
    state = LearnerState(user_id="u")
    once = apply_assessment(
        state, unit=_unit(), method=TeachingMethod.GAME, result=_result(Verdict.PASS),
    )
    twice = apply_assessment(
        once, unit=_unit(), method=TeachingMethod.GAME, result=_result(Verdict.PASS),
    )
    assert twice.mastered_concepts == ["Understand SQL joins"]  # no duplicate
