"""Interactive assessment Q→A→verdict loop tests."""
from __future__ import annotations

import asyncio
import json

import pytest_asyncio

from backend.agents.assessment import generate_assessment_question
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
    AssessmentQuestionEvent,
    AssessmentResultEvent,
    DesiredDepth,
    GroundingSource,
    KnowledgeLevel,
    LearningGoal,
    LearningUnit,
    TeachingChunkEvent,
    TeachingMethod,
    Verdict,
)
from tests._fakes import FakeAnthropic, text_response, tool_use_response


@pytest_asyncio.fixture
async def session_factory():
    engine = create_sqlite_engine(IN_MEMORY_DB_URL)
    await init_db(engine)
    factory = make_session_factory(engine)
    yield factory
    await engine.dispose()


def _goal() -> LearningGoal:
    return LearningGoal(
        domain="x", sub_goal="y",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=1.0,
    )


def _curriculum_payload(methods: list[str]) -> dict:
    return {
        "units": [
            {
                "id": f"u{i+1}",
                "objective": f"Learn topic {i+1}",
                "prerequisites": [],
                "recommended_pedagogy": m,
                "grounding_sources": [{
                    "url": f"https://example.com/{i+1}",
                    "title": "t", "snippet": "s",
                }],
            }
            for i, m in enumerate(methods)
        ]
    }


def _verdict_payload(v: str) -> dict:
    return {
        "verdict": v,
        "diagnostic_notes": f"learner showed {v}",
        "confidence": 0.9,
    }


def _search_tool() -> Tool:
    async def fake_search(_inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]
    return Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )


# ---------- generate_assessment_question (unit) ----------

async def test_generate_assessment_question_returns_clean_text():
    unit = LearningUnit(
        id="u1", objective="Trace what await does",
        prerequisites=[],
        recommended_pedagogy=TeachingMethod.WORKED_EXAMPLE,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )
    client = FakeAnthropic([
        text_response("What does the loop do when a coroutine yields a Future?"),
    ])
    q = await generate_assessment_question(
        unit, TeachingMethod.WORKED_EXAMPLE,
        teaching_content="Some markdown about await...",
        client=client,
    )
    assert "Future" in q
    assert not q.startswith('"')


async def test_generate_assessment_question_strips_quote_wrap():
    unit = LearningUnit(
        id="u1", objective="x", prerequisites=[],
        recommended_pedagogy=TeachingMethod.ANALOGY,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )
    client = FakeAnthropic([text_response('"trace one iteration"')])
    q = await generate_assessment_question(
        unit, TeachingMethod.ANALOGY, "x", client=client,
    )
    assert q == "trace one iteration"


# ---------- end-to-end interactive flow ----------

async def test_interactive_assessment_pass_path(session_factory):
    """Q→A→pass: agent asks, user answers, judge marks pass, advance."""
    client = FakeAnthropic([
        # Curriculum (PTC)
        tool_use_response("c1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy"]))),
        # u1 assessment: question
        text_response("Predict what happens at the first await."),
        # u1 assessment: verdict
        text_response(json.dumps(_verdict_payload("pass"))),
    ])
    assess_q: asyncio.Queue[str] = asyncio.Queue()
    assess_q.put_nowait("The coroutine yields a Future to the loop and pauses.")

    qs = []
    rs = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="alice",
        client=client, search_tool=_search_tool(),
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="t-int-pass",
        assessment_queue=assess_q,
    ):
        if isinstance(ev, AssessmentQuestionEvent):
            qs.append(ev)
        elif isinstance(ev, AssessmentResultEvent):
            rs.append(ev)

    # Exactly one question yielded; verdict from the assessment LLM is pass.
    assert len(qs) == 1
    assert qs[0].unit_id == "u1"
    assert qs[0].attempt == 1
    assert "await" in qs[0].question.lower()
    assert len(rs) == 1
    assert rs[0].result.verdict is Verdict.PASS


async def test_interactive_assessment_reinforce_then_pass(session_factory):
    """Q→A→reinforce → re-teach → Q→A→pass."""
    client = FakeAnthropic([
        tool_use_response("c1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy"]))),
        # Attempt 1: question + verdict (reinforce)
        text_response("Trace one iteration."),
        text_response(json.dumps(_verdict_payload("needs_reinforcement"))),
        # Attempt 2: question + verdict (pass)
        text_response("Now predict the next iteration."),
        text_response(json.dumps(_verdict_payload("pass"))),
    ])
    assess_q: asyncio.Queue[str] = asyncio.Queue()
    assess_q.put_nowait("Hmm I'm shaky on the yield part.")
    assess_q.put_nowait("Now I see — the Task.__step calls send().")

    chunks = []
    qs = []
    rs = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="bob",
        client=client, search_tool=_search_tool(),
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="t-int-reinforce",
        assessment_queue=assess_q,
    ):
        if isinstance(ev, TeachingChunkEvent):
            chunks.append(ev)
        elif isinstance(ev, AssessmentQuestionEvent):
            qs.append(ev)
        elif isinstance(ev, AssessmentResultEvent):
            rs.append(ev)

    # Same unit, taught twice, asked twice, two verdicts.
    assert len(chunks) == 2
    assert len(qs) == 2
    assert qs[0].attempt == 1 and qs[1].attempt == 2
    assert len(rs) == 2
    assert rs[0].result.verdict is Verdict.NEEDS_REINFORCEMENT
    assert rs[1].result.verdict is Verdict.PASS


async def test_assessment_queue_optional_falls_back_to_verdict_queue(session_factory):
    """If only verdict_queue is provided, the legacy click-button path fires."""
    client = FakeAnthropic([
        tool_use_response("c1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy"]))),
    ])
    verdict_q: asyncio.Queue[str] = asyncio.Queue()
    verdict_q.put_nowait("pass")

    qs = []
    rs = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="carol",
        client=client, search_tool=_search_tool(),
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="t-fallback",
        verdict_queue=verdict_q,  # legacy path
        # assessment_queue intentionally omitted
    ):
        if isinstance(ev, AssessmentQuestionEvent):
            qs.append(ev)
        elif isinstance(ev, AssessmentResultEvent):
            rs.append(ev)

    # No interactive Q&A happened; fell back to verdict_queue.
    assert qs == []
    assert len(rs) == 1
    assert rs[0].result.verdict is Verdict.PASS
