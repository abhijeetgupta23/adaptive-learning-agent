"""Verify every agent attaches cache_control to its system prompt."""
import json

from backend.agents.assessment import assess_understanding
from backend.agents.curriculum import build_curriculum
from backend.agents.intent import intent_turn
from backend.agents.pedagogy_router import route_pedagogy
from backend.agents.teaching.game import generate_game
from backend.orchestration.cache import cached_system, normalize_system
from backend.orchestration.tools import Tool
from backend.schemas import (
    DesiredDepth,
    GroundingSource,
    KnowledgeLevel,
    LearnerState,
    LearningGoal,
    LearningUnit,
    TeachingMethod,
)
from evals.evaluators.curriculum_coherence import evaluate_curriculum_coherence
from evals.evaluators.game_quality import evaluate_game_quality
from evals.evaluators.intent_fidelity import evaluate_intent_fidelity
from tests._fakes import FakeAnthropic, text_response, tool_use_response


def _assert_cached(call: dict) -> None:
    system = call["system"]
    assert isinstance(system, list), f"expected list, got {type(system).__name__}"
    assert len(system) >= 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["type"] == "text"
    assert isinstance(system[0]["text"], str)


# -------- helpers tests --------

def test_cached_system_wraps_with_ephemeral_marker() -> None:
    out = cached_system("hello")
    assert out == [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]


def test_normalize_system_passes_list_through() -> None:
    pre = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
    assert normalize_system(pre) is pre


def test_normalize_system_wraps_string() -> None:
    out = normalize_system("hi")
    assert out[0]["cache_control"] == {"type": "ephemeral"}


def test_normalize_system_returns_none_unchanged() -> None:
    assert normalize_system(None) is None


# -------- per-agent caching --------

async def test_intent_turn_caches_system() -> None:
    client = FakeAnthropic([text_response(json.dumps({
        "status": "asking", "question": "current level?",
    }))])
    await intent_turn(client, conversation=[{"role": "user", "content": "x"}])
    _assert_cached(client.messages.calls[0])


async def test_assessment_caches_system() -> None:
    unit = LearningUnit(
        id="u1", objective="o",
        recommended_pedagogy=TeachingMethod.VISUAL,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )
    client = FakeAnthropic([text_response(json.dumps({
        "verdict": "pass", "diagnostic_notes": "x", "confidence": 0.9,
    }))])
    await assess_understanding(unit, TeachingMethod.VISUAL, [], client=client)
    _assert_cached(client.messages.calls[0])


async def test_game_caches_system() -> None:
    from tests._fakes import game_happy_path_responses
    unit = LearningUnit(
        id="u1", objective="o",
        recommended_pedagogy=TeachingMethod.GAME,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )
    client = FakeAnthropic(game_happy_path_responses())
    await generate_game(unit, client=client)
    # The PTC loop calls the LLM for every iteration with the same system
    # prompt — every call should be cache-marked, not just the first.
    for call in client.messages.calls:
        _assert_cached(call)


async def test_pedagogy_router_llm_caches_system() -> None:
    unit = LearningUnit(
        id="u1", objective="o",
        recommended_pedagogy=TeachingMethod.VISUAL,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )
    client = FakeAnthropic([text_response(json.dumps({
        "method": "analogy", "rationale": "r",
    }))])
    await route_pedagogy(
        unit, LearnerState(user_id="u"),
        excluded_methods=[TeachingMethod.VISUAL],
        llm_client=client,
    )
    _assert_cached(client.messages.calls[0])


async def test_curriculum_ptc_caches_system_on_every_iteration() -> None:
    """The PTC loop's biggest win: same system prompt across iterations is cached."""
    async def fake_search(inp: dict) -> list[dict]:
        return [{"url": "https://x.com", "title": "t", "snippet": "s"}]

    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )

    payload = {"units": [{
        "id": "u1", "objective": "o", "prerequisites": [],
        "recommended_pedagogy": "visual",
        "grounding_sources": [{"url": "https://x.com", "title": "t", "snippet": "s"}],
    }]}

    client = FakeAnthropic([
        tool_use_response("t1", "web_search", {"query": "x"}),
        tool_use_response("t2", "web_search", {"query": "y"}),
        text_response(json.dumps(payload)),
    ])

    goal = LearningGoal(
        domain="d", sub_goal="s",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=1.0,
    )
    from backend.agents.teaching.registry import build_teaching_registry
    await build_curriculum(
        goal, client=client, search_tool=search_tool,
        teaching_registry=build_teaching_registry(client=client),
    )

    # Three iterations; system should be cache-marked on every single one.
    assert len(client.messages.calls) == 3
    for call in client.messages.calls:
        _assert_cached(call)
        # And the cached prefix is identical across calls (precondition for cache hit).
        assert call["system"] == client.messages.calls[0]["system"]


# -------- evaluators --------

async def test_intent_fidelity_judge_caches_system() -> None:
    client = FakeAnthropic([text_response(json.dumps({
        "field_scores": [
            {"field": f, "score": 1.0, "rationale": "x"}
            for f in [
                "domain", "sub_goal", "current_level",
                "desired_depth", "time_budget_hours", "preferred_modalities",
            ]
        ],
        "overall_score": 1.0, "passed": True, "rationale": "ok",
    }))])
    goal = LearningGoal(
        domain="d", sub_goal="s",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=1.0,
    )
    await evaluate_intent_fidelity(expected=goal, predicted=goal, judge=client)
    _assert_cached(client.messages.calls[0])


async def test_curriculum_coherence_judge_caches_system() -> None:
    from backend.schemas.curriculum import Curriculum
    goal = LearningGoal(
        domain="d", sub_goal="s",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=1.0,
    )
    unit = LearningUnit(
        id="u1", objective="o",
        recommended_pedagogy=TeachingMethod.VISUAL,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )
    curriculum = Curriculum(goal=goal, units=[unit])
    client = FakeAnthropic([text_response(json.dumps({
        "axis_scores": [
            {"axis": a, "score": 1.0, "rationale": "x"}
            for a in ["ordering", "completeness", "scope", "pedagogy_fit"]
        ],
        "overall_score": 1.0, "passed": True, "rationale": "ok",
    }))])
    await evaluate_curriculum_coherence(curriculum, judge=client)
    _assert_cached(client.messages.calls[0])


async def test_game_quality_judge_caches_system() -> None:
    from backend.agents.teaching.game import GameSpec
    spec = GameSpec(
        unit_id="u1", title="t", description="d", concept="c",
        instructions="i", html="<div><button onclick='x()'>x</button></div>",
    )
    unit = LearningUnit(
        id="u1", objective="o",
        recommended_pedagogy=TeachingMethod.GAME,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )
    client = FakeAnthropic([text_response(json.dumps({
        "axis_scores": [
            {"axis": a, "score": 1.0, "rationale": "x"}
            for a in ["solvable", "on_concept", "engagement"]
        ],
        "overall_score": 1.0, "passed": True, "rationale": "ok",
    }))])
    await evaluate_game_quality(spec, unit, judge=client)
    _assert_cached(client.messages.calls[0])
