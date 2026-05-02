import json

import pytest

from backend.agents.pedagogy_router import RouterError, route_pedagogy
from backend.schemas import (
    GroundingSource,
    LearnerState,
    LearningUnit,
    TeachingMethod,
)
from tests._fakes import FakeAnthropic, text_response


def _unit(
    method: TeachingMethod = TeachingMethod.VISUAL,
    objective: str = "Distinguish INNER vs LEFT joins",
) -> LearningUnit:
    return LearningUnit(
        id="u1",
        objective=objective,
        recommended_pedagogy=method,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )


async def test_uses_learner_effective_pedagogy_when_present() -> None:
    state = LearnerState(
        user_id="me",
        effective_pedagogies={"Distinguish INNER vs LEFT joins": TeachingMethod.GAME},
    )
    plan = await route_pedagogy(_unit(TeachingMethod.VISUAL), state)
    assert plan.methods[0].method is TeachingMethod.GAME
    assert "previously succeeded" in plan.rationale.lower()


async def test_falls_back_to_curriculum_recommendation() -> None:
    state = LearnerState(user_id="me")
    plan = await route_pedagogy(_unit(TeachingMethod.ANALOGY), state)
    assert plan.methods[0].method is TeachingMethod.ANALOGY


async def test_excludes_previously_failed_method() -> None:
    state = LearnerState(user_id="me")
    plan = await route_pedagogy(
        _unit(TeachingMethod.VISUAL),
        state,
        excluded_methods=[TeachingMethod.VISUAL],
    )
    # No LLM, no preference — deterministic fallback picks first non-excluded.
    assert plan.methods[0].method is not TeachingMethod.VISUAL


async def test_effective_pedagogy_respects_exclusion() -> None:
    state = LearnerState(
        user_id="me",
        effective_pedagogies={"Distinguish INNER vs LEFT joins": TeachingMethod.GAME},
    )
    plan = await route_pedagogy(
        _unit(TeachingMethod.VISUAL),
        state,
        excluded_methods=[TeachingMethod.GAME],
    )
    # GAME is excluded → falls back to curriculum recommendation (VISUAL).
    assert plan.methods[0].method is TeachingMethod.VISUAL


async def test_llm_fallback_when_curriculum_recommendation_excluded() -> None:
    client = FakeAnthropic([text_response(json.dumps(
        {"method": "analogy", "rationale": "bridges to known concepts"}
    ))])
    plan = await route_pedagogy(
        _unit(TeachingMethod.VISUAL),
        LearnerState(user_id="me"),
        excluded_methods=[TeachingMethod.VISUAL],
        llm_client=client,
    )
    assert plan.methods[0].method is TeachingMethod.ANALOGY
    assert "bridges to known concepts" in plan.rationale


async def test_llm_rejects_invalid_method_pick() -> None:
    client = FakeAnthropic([text_response(json.dumps(
        {"method": "telepathy", "rationale": "meh"}
    ))])
    with pytest.raises(RouterError):
        await route_pedagogy(
            _unit(TeachingMethod.VISUAL),
            LearnerState(user_id="me"),
            excluded_methods=[TeachingMethod.VISUAL],
            llm_client=client,
        )


async def test_llm_rejects_excluded_method_pick() -> None:
    client = FakeAnthropic([text_response(json.dumps(
        {"method": "visual", "rationale": "ignoring exclusion"}
    ))])
    with pytest.raises(RouterError):
        await route_pedagogy(
            _unit(TeachingMethod.VISUAL),
            LearnerState(user_id="me"),
            excluded_methods=[TeachingMethod.VISUAL],
            llm_client=client,
        )


async def test_all_methods_excluded_raises() -> None:
    state = LearnerState(user_id="me")
    with pytest.raises(RouterError):
        await route_pedagogy(
            _unit(TeachingMethod.VISUAL),
            state,
            excluded_methods=list(TeachingMethod),
        )
