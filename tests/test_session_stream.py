import json

import pytest_asyncio

from backend.agents.teaching.registry import build_teaching_registry
from backend.api.session import run_session_stream
from backend.orchestration.tools import Tool
from backend.persistence import (
    IN_MEMORY_DB_URL,
    create_sqlite_engine,
    init_db,
    load_learner_state,
    make_session_factory,
)
from backend.schemas import (
    AssessmentResultEvent,
    CompleteEvent,
    CurriculumReadyEvent,
    DesiredDepth,
    IntentUpdateEvent,
    KnowledgeLevel,
    LearningGoal,
    SessionStartedEvent,
    TeachingChunkEvent,
    TeachingMethod,
    UnitStartedEvent,
    Verdict,
)
from tests._fakes import (
    FakeAnthropic,
    game_happy_path_responses,
    text_response,
    tool_use_response,
)


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


async def test_session_stream_end_to_end(session_factory) -> None:
    async def fake_search(inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]

    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )

    client = FakeAnthropic([
        # Curriculum agent: search, then final JSON
        tool_use_response("c1", "web_search", {"query": "sql joins"}),
        text_response(json.dumps(_curriculum_payload(["game", "visual"]))),
        # Game agent: full PTC tool-use sequence for u1
        *game_happy_path_responses(title="Join Matchmaker"),
    ])

    events = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="alice",
        client=client, search_tool=search_tool,
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="test-alice",
    ):
        events.append(ev)

    # Expected sequence for 2 units (u1=game, u2=visual):
    # session_started, intent_update, curriculum_ready,
    # unit_started(u1), teaching_chunk(u1=game, html), assessment_result(u1),
    # unit_started(u2), teaching_chunk(u2=visual, stub), assessment_result(u2),
    # complete
    assert isinstance(events[0], SessionStartedEvent)
    assert events[0].session_id == "test-alice"
    assert isinstance(events[1], IntentUpdateEvent)
    assert isinstance(events[2], CurriculumReadyEvent)
    assert len(events[2].curriculum.units) == 2
    assert isinstance(events[3], UnitStartedEvent)
    assert events[3].unit_id == "u1"
    assert events[3].plan.methods[0].method is TeachingMethod.GAME
    assert isinstance(events[4], TeachingChunkEvent)
    assert events[4].method is TeachingMethod.GAME
    assert "INNER" in events[4].content
    assert isinstance(events[5], AssessmentResultEvent)
    assert events[5].result.verdict is Verdict.PASS
    assert isinstance(events[6], UnitStartedEvent)
    assert events[6].plan.methods[0].method is TeachingMethod.VISUAL
    assert isinstance(events[7], TeachingChunkEvent)
    assert events[7].method is TeachingMethod.VISUAL
    assert "Demo visual teaching" in events[7].content
    assert isinstance(events[8], AssessmentResultEvent)
    assert isinstance(events[-1], CompleteEvent)


async def test_session_stream_persists_learner_state(session_factory) -> None:
    async def fake_search(inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]

    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )
    client = FakeAnthropic([
        tool_use_response("t1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy"]))),
    ])

    async for _ in run_session_stream(
        goal=_goal(), user_id="bob",
        client=client, search_tool=search_tool,
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="test-bob",
    ):
        pass

    async with session_factory() as session:
        saved = await load_learner_state("bob", session=session)
    assert saved is not None
    assert "Learn topic 1" in saved.mastered_concepts
    assert saved.effective_pedagogies["Learn topic 1"] is TeachingMethod.ANALOGY
