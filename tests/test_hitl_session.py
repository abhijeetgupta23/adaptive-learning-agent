"""HITL flow: SSE stream pauses on intent question, resumes after queue.put."""
from __future__ import annotations

import asyncio
import json

import pytest_asyncio

from backend.agents.teaching.registry import build_teaching_registry
from backend.api.session import run_session_stream
from backend.orchestration.tools import Tool
from backend.persistence import (
    IN_MEMORY_DB_URL,
    create_sqlite_engine,
    init_db,
    make_session_factory,
)
from backend.schemas import (
    CompleteEvent,
    CurriculumReadyEvent,
    IntentAskEvent,
    IntentUpdateEvent,
    SessionStartedEvent,
)
from tests._fakes import FakeAnthropic, text_response, tool_use_response


@pytest_asyncio.fixture
async def session_factory():
    engine = create_sqlite_engine(IN_MEMORY_DB_URL)
    await init_db(engine)
    factory = make_session_factory(engine)
    yield factory
    await engine.dispose()


def _intent_ask(question: str) -> str:
    return json.dumps({"status": "asking", "question": question})


def _intent_complete() -> str:
    return json.dumps({
        "status": "complete",
        "goal": {
            "domain": "python",
            "sub_goal": "asyncio basics",
            "current_level": "intermediate",
            "desired_depth": "working",
            "time_budget_hours": 2.0,
            "preferred_modalities": [],
        },
    })


def _curriculum_payload() -> dict:
    return {
        "units": [
            {
                "id": "u1",
                "objective": "Learn async/await semantics",
                "prerequisites": [],
                "recommended_pedagogy": "analogy",
                "grounding_sources": [{
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "title": "asyncio",
                    "snippet": "s",
                }],
            },
        ]
    }


async def test_hitl_intent_pauses_then_resumes_with_user_answer(session_factory):
    """When the intent agent asks, the stream yields IntentAskEvent and waits
    for the queue. After the queue receives an answer, the loop runs another
    intent_turn and (eventually) resolves with a complete goal."""

    async def fake_search(inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]

    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )

    client = FakeAnthropic([
        # Turn 1: intent agent asks.
        text_response(_intent_ask("How deep do you want to go?")),
        # Turn 2: after user answers, intent agent completes.
        text_response(_intent_complete()),
        # Curriculum agent: search, then JSON.
        tool_use_response("t1", "web_search", {"query": "asyncio"}),
        text_response(json.dumps(_curriculum_payload())),
    ])

    queue: asyncio.Queue[str] = asyncio.Queue()
    queue.put_nowait("Working knowledge — I write async code.")

    events = []
    async for ev in run_session_stream(
        user_id="alice",
        client=client,
        search_tool=search_tool,
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="sid-1",
        response_queue=queue,
        initial_message="how does Python's asyncio scheduler work",
    ):
        events.append(ev)

    # First event: SessionStartedEvent with the session_id.
    assert isinstance(events[0], SessionStartedEvent)
    assert events[0].session_id == "sid-1"

    # Then: an IntentAskEvent with the question.
    assert isinstance(events[1], IntentAskEvent)
    assert "deep" in events[1].question.lower()
    assert events[1].session_id == "sid-1"

    # The user's answer was already on the queue; intent loop resumed and
    # produced an IntentUpdateEvent with the resolved goal.
    intent_update_idx = next(
        i for i, ev in enumerate(events) if isinstance(ev, IntentUpdateEvent)
    )
    assert events[intent_update_idx].goal is not None
    assert events[intent_update_idx].goal.sub_goal == "asyncio basics"

    # Curriculum then proceeds normally and the session completes.
    assert any(isinstance(ev, CurriculumReadyEvent) for ev in events)
    assert isinstance(events[-1], CompleteEvent)


async def test_hitl_skipped_when_goal_provided(session_factory):
    """If `goal` is supplied, the intent loop is bypassed entirely."""
    from backend.schemas import (
        DesiredDepth,
        KnowledgeLevel,
        LearningGoal,
    )

    async def fake_search(inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]

    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )

    client = FakeAnthropic([
        # Only curriculum + (no game/worked unit) — no intent calls expected.
        tool_use_response("t1", "web_search", {"query": "asyncio"}),
        text_response(json.dumps(_curriculum_payload())),
    ])

    goal = LearningGoal(
        domain="python", sub_goal="asyncio basics",
        current_level=KnowledgeLevel.INTERMEDIATE,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=2.0,
    )

    events = []
    async for ev in run_session_stream(
        user_id="bob",
        client=client,
        search_tool=search_tool,
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="sid-2",
        goal=goal,
    ):
        events.append(ev)

    # No IntentAskEvent — the loop was bypassed.
    assert not any(isinstance(ev, IntentAskEvent) for ev in events)
    # First IntentUpdateEvent carries the resolved goal directly.
    iu = next(ev for ev in events if isinstance(ev, IntentUpdateEvent))
    assert iu.goal == goal
