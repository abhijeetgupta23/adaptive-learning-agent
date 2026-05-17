"""Tests for the per-session Game-agent circuit breaker."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio

from backend.agents.teaching.registry import build_teaching_registry
from backend.api.session import run_session_stream
from backend.orchestration.circuit_breaker import SessionCircuitBreaker
from backend.orchestration.cost import session_accumulator
from backend.orchestration.tools import Tool
from backend.persistence import (
    IN_MEMORY_DB_URL,
    create_sqlite_engine,
    init_db,
    make_session_factory,
)
from backend.schemas import (
    CircuitBreakerTrippedEvent,
    DesiredDepth,
    KnowledgeLevel,
    LearningGoal,
    TeachingChunkEvent,
    TeachingMethod,
    UnitStartedEvent,
)
from tests._fakes import FakeAnthropic, text_response, tool_use_response

# ---------- unit tests for SessionCircuitBreaker ----------


def test_does_not_trip_on_first_pass() -> None:
    cb = SessionCircuitBreaker()
    cb.record_game_attempt(passed=True, cost_usd=0.01)
    tripped, reason = cb.should_trip()
    assert tripped is False
    assert reason is None


def test_does_not_trip_below_failure_threshold() -> None:
    cb = SessionCircuitBreaker(max_consecutive_failures=3, max_spend_usd=1.0)
    cb.record_game_attempt(passed=False, cost_usd=0.01)
    cb.record_game_attempt(passed=False, cost_usd=0.01)
    assert cb.should_trip() == (False, None)


def test_trips_after_n_consecutive_failures() -> None:
    cb = SessionCircuitBreaker(max_consecutive_failures=3, max_spend_usd=10.0)
    for _ in range(3):
        cb.record_game_attempt(passed=False, cost_usd=0.001)
    tripped, reason = cb.should_trip()
    assert tripped is True
    assert reason is not None
    assert "3 consecutive game failures" in reason


def test_trips_on_spend_threshold() -> None:
    cb = SessionCircuitBreaker(max_consecutive_failures=99, max_spend_usd=0.50)
    cb.record_game_attempt(passed=True, cost_usd=0.30)
    assert cb.should_trip()[0] is False
    cb.record_game_attempt(passed=True, cost_usd=0.25)
    tripped, reason = cb.should_trip()
    assert tripped is True
    assert reason is not None
    assert "exceeded budget" in reason


def test_pass_resets_consecutive_failure_counter() -> None:
    cb = SessionCircuitBreaker(max_consecutive_failures=3, max_spend_usd=10.0)
    cb.record_game_attempt(passed=False, cost_usd=0.0)
    cb.record_game_attempt(passed=False, cost_usd=0.0)
    cb.record_game_attempt(passed=True, cost_usd=0.0)
    assert cb.consecutive_failures == 0
    cb.record_game_attempt(passed=False, cost_usd=0.0)
    cb.record_game_attempt(passed=False, cost_usd=0.0)
    assert cb.should_trip()[0] is False  # only 2 in a row since the pass


def test_tripped_state_is_sticky() -> None:
    cb = SessionCircuitBreaker(max_consecutive_failures=2, max_spend_usd=10.0)
    cb.record_game_attempt(passed=False, cost_usd=0.0)
    cb.record_game_attempt(passed=False, cost_usd=0.0)
    assert cb.should_trip()[0] is True
    # Even a pass afterward doesn't un-trip the breaker.
    cb.record_game_attempt(passed=True, cost_usd=0.0)
    assert cb.should_trip()[0] is True


def test_reset_clears_state() -> None:
    cb = SessionCircuitBreaker(max_consecutive_failures=2, max_spend_usd=10.0)
    cb.record_game_attempt(passed=False, cost_usd=0.20)
    cb.record_game_attempt(passed=False, cost_usd=0.20)
    assert cb.should_trip()[0] is True
    cb.reset()
    assert cb.should_trip() == (False, None)
    assert cb.cumulative_spend_usd == 0.0


def test_negative_cost_rejected() -> None:
    cb = SessionCircuitBreaker()
    with pytest.raises(ValueError):
        cb.record_game_attempt(passed=True, cost_usd=-0.01)


def test_defaults_are_three_and_fifty_cents() -> None:
    cb = SessionCircuitBreaker()
    assert cb.max_consecutive_failures == 3
    assert cb.max_spend_usd == 0.50


# ---------- integration smoke test through run_session_stream ----------


@pytest_asyncio.fixture
async def session_factory():
    engine = create_sqlite_engine(IN_MEMORY_DB_URL)
    await init_db(engine)
    factory = make_session_factory(engine)
    yield factory
    await engine.dispose()


def _goal() -> LearningGoal:
    return LearningGoal(
        domain="databases",
        sub_goal="understand SQL joins",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=4.0,
    )


def _curriculum_payload(methods: list[str]) -> dict:
    return {
        "units": [
            {
                "id": f"u{i+1}",
                "objective": f"Learn topic {i+1}",
                "prerequisites": [],
                "recommended_pedagogy": methods[i],
                "grounding_sources": [{
                    "url": f"https://example.com/{i+1}",
                    "title": f"source {i+1}",
                    "snippet": "s",
                }],
            }
            for i in range(len(methods))
        ]
    }


async def test_pre_tripped_breaker_routes_game_units_to_worked_example(
    session_factory,
) -> None:
    """Integration: when the breaker is pre-tripped, game-recommended units
    flip to worked_example AND a CircuitBreakerTrippedEvent is emitted exactly
    once."""
    async def fake_search(inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]

    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )

    # Curriculum recommends `game` for both units, but the pre-tripped breaker
    # should redirect both to worked_example. LLM calls expected:
    #   1. curriculum web_search
    #   2. curriculum JSON
    #   3. worked_example for u1
    #   4. worked_example for u2
    # No game-agent PTC calls happen — the breaker short-circuits them.
    client = FakeAnthropic([
        tool_use_response("c1", "web_search", {"query": "sql joins"}),
        text_response(json.dumps(_curriculum_payload(["game", "game"]))),
        text_response("# Worked example for u1\n\nstep by step..."),
        text_response("# Worked example for u2\n\nstep by step..."),
    ])

    breaker = SessionCircuitBreaker(max_consecutive_failures=1, max_spend_usd=10.0)
    breaker.record_game_attempt(passed=False, cost_usd=0.0)  # pre-trip it
    assert breaker.should_trip()[0] is True

    events = []
    with session_accumulator():
        async for ev in run_session_stream(
            goal=_goal(), user_id="alice",
            client=client, search_tool=search_tool,
            session_factory=session_factory,
            teaching_registry=build_teaching_registry(client=client),
            session_id="cb-test",
            circuit_breaker=breaker,
        ):
            events.append(ev)

    trip_events = [e for e in events if isinstance(e, CircuitBreakerTrippedEvent)]
    assert len(trip_events) == 1, "Trip event should be emitted exactly once"
    assert trip_events[0].fallback_method is TeachingMethod.WORKED_EXAMPLE

    unit_starts = [e for e in events if isinstance(e, UnitStartedEvent)]
    teaching_chunks = [e for e in events if isinstance(e, TeachingChunkEvent)]
    # Both units routed to worked_example by the breaker override.
    assert len(unit_starts) == 2
    assert all(
        u.plan.methods[0].method is TeachingMethod.WORKED_EXAMPLE
        for u in unit_starts
    )
    assert len(teaching_chunks) == 2
    assert all(t.method is TeachingMethod.WORKED_EXAMPLE for t in teaching_chunks)
