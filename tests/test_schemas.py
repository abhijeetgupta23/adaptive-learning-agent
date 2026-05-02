import pytest
from pydantic import ValidationError

from backend.schemas import (
    AssessmentResult,
    AssessmentResultEvent,
    Curriculum,
    CurriculumReadyEvent,
    DesiredDepth,
    GroundingSource,
    KnowledgeLevel,
    LearnerState,
    LearningGoal,
    LearningUnit,
    MethodCall,
    Modality,
    SSEEventAdapter,
    TeachingChunkEvent,
    TeachingMethod,
    TeachingPlan,
    Verdict,
)


def _goal() -> LearningGoal:
    return LearningGoal(
        domain="databases",
        sub_goal="understand SQL joins",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=2.0,
        preferred_modalities=[Modality.VISUAL, Modality.INTERACTIVE],
    )


def _source() -> GroundingSource:
    return GroundingSource(
        url="https://example.com/joins",
        title="SQL Joins Explained",
        snippet="Inner joins match rows where a predicate holds.",
    )


def _unit(uid: str = "u1") -> LearningUnit:
    return LearningUnit(
        id=uid,
        objective="Distinguish inner vs outer joins",
        prerequisites=[],
        recommended_pedagogy=TeachingMethod.VISUAL,
        grounding_sources=[_source()],
    )


class TestLearningGoal:
    def test_valid_goal_roundtrips(self) -> None:
        goal = _goal()
        assert goal.domain == "databases"
        assert goal.desired_depth is DesiredDepth.WORKING
        assert LearningGoal.model_validate(goal.model_dump()) == goal

    def test_empty_domain_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LearningGoal(
                domain="",
                sub_goal="x",
                current_level=KnowledgeLevel.NOVICE,
                desired_depth=DesiredDepth.COCKTAIL,
                time_budget_hours=1.0,
            )

    def test_non_positive_time_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LearningGoal(
                domain="x",
                sub_goal="y",
                current_level=KnowledgeLevel.NOVICE,
                desired_depth=DesiredDepth.COCKTAIL,
                time_budget_hours=0,
            )


class TestCurriculum:
    def test_valid_curriculum(self) -> None:
        curriculum = Curriculum(goal=_goal(), units=[_unit("u1"), _unit("u2")])
        assert len(curriculum.units) == 2

    def test_curriculum_requires_at_least_one_unit(self) -> None:
        with pytest.raises(ValidationError):
            Curriculum(goal=_goal(), units=[])

    def test_unit_requires_at_least_one_grounding_source(self) -> None:
        with pytest.raises(ValidationError):
            LearningUnit(
                id="u1",
                objective="obj",
                recommended_pedagogy=TeachingMethod.GAME,
                grounding_sources=[],
            )


class TestTeachingPlan:
    def test_valid_plan(self) -> None:
        plan = TeachingPlan(
            unit_id="u1",
            methods=[
                MethodCall(method=TeachingMethod.ANALOGY),
                MethodCall(method=TeachingMethod.GAME, params={"difficulty": "easy"}),
            ],
            rationale="warm up then engage",
        )
        assert plan.methods[1].params["difficulty"] == "easy"

    def test_empty_methods_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TeachingPlan(unit_id="u1", methods=[])


class TestAssessmentResult:
    def test_valid_result(self) -> None:
        result = AssessmentResult(
            unit_id="u1",
            verdict=Verdict.PASS,
            diagnostic_notes="learner explained the concept unprompted",
            confidence=0.85,
        )
        assert result.verdict is Verdict.PASS

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AssessmentResult(
                unit_id="u1",
                verdict=Verdict.PASS,
                diagnostic_notes="",
                confidence=1.5,
            )


class TestLearnerState:
    def test_defaults_are_empty(self) -> None:
        state = LearnerState(user_id="u1")
        assert state.mastered_concepts == []
        assert state.effective_pedagogies == {}

    def test_roundtrip_preserves_pedagogy_enum(self) -> None:
        state = LearnerState(
            user_id="u1",
            effective_pedagogies={"joins": TeachingMethod.GAME},
        )
        restored = LearnerState.model_validate(state.model_dump())
        assert restored.effective_pedagogies["joins"] is TeachingMethod.GAME


class TestSSEEvents:
    def test_discriminated_union_dispatches_by_event(self) -> None:
        payload = TeachingChunkEvent(
            unit_id="u1", method=TeachingMethod.ANALOGY, content="like a key..."
        ).model_dump()
        parsed = SSEEventAdapter.validate_python(payload)
        assert isinstance(parsed, TeachingChunkEvent)
        assert parsed.content == "like a key..."

    def test_curriculum_event_roundtrips(self) -> None:
        curriculum = Curriculum(goal=_goal(), units=[_unit()])
        event = CurriculumReadyEvent(curriculum=curriculum)
        parsed = SSEEventAdapter.validate_python(event.model_dump())
        assert isinstance(parsed, CurriculumReadyEvent)
        assert parsed.curriculum.units[0].id == "u1"

    def test_assessment_event_roundtrips(self) -> None:
        result = AssessmentResult(
            unit_id="u1",
            verdict=Verdict.NEEDS_DIFFERENT_APPROACH,
            diagnostic_notes="stuck on terminology",
            confidence=0.4,
        )
        event = AssessmentResultEvent(result=result)
        parsed = SSEEventAdapter.validate_python(event.model_dump())
        assert isinstance(parsed, AssessmentResultEvent)
        assert parsed.result.verdict is Verdict.NEEDS_DIFFERENT_APPROACH

    def test_unknown_event_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SSEEventAdapter.validate_python({"event": "nope"})
