"""Per-unit verdict loop: pass / needs_reinforcement / needs_different_approach."""
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
    AssessmentResultEvent,
    DesiredDepth,
    KnowledgeLevel,
    LearningGoal,
    TeachingChunkEvent,
    TeachingMethod,
    UnitStartedEvent,
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


def _search_tool() -> Tool:
    async def fake_search(_inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]
    return Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )


# ---------- pass path ----------

async def test_verdict_pass_advances_to_next_unit(session_factory):
    """User clicks 'Got it' on each unit → assessment is pass, no re-teach."""
    client = FakeAnthropic([
        # Curriculum
        tool_use_response("c1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy", "visual"]))),
    ])
    verdict_q: asyncio.Queue[str] = asyncio.Queue()
    verdict_q.put_nowait("pass")
    verdict_q.put_nowait("pass")

    teaching_chunks = []
    assessments = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="alice",
        client=client, search_tool=_search_tool(),
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="t-pass",
        verdict_queue=verdict_q,
    ):
        if isinstance(ev, TeachingChunkEvent):
            teaching_chunks.append(ev)
        elif isinstance(ev, AssessmentResultEvent):
            assessments.append(ev)

    # Two units, each taught once, both pass.
    assert len(teaching_chunks) == 2
    assert len(assessments) == 2
    assert all(a.result.verdict is Verdict.PASS for a in assessments)


# ---------- needs_reinforcement path ----------

async def test_verdict_reinforce_then_pass_re_teaches_same_method(session_factory):
    """First verdict 'needs_reinforcement' → re-teach with same method; then pass."""
    client = FakeAnthropic([
        tool_use_response("c1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy"]))),
    ])
    verdict_q: asyncio.Queue[str] = asyncio.Queue()
    verdict_q.put_nowait("needs_reinforcement")
    verdict_q.put_nowait("pass")

    starts = []
    chunks = []
    assessments = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="bob",
        client=client, search_tool=_search_tool(),
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="t-reinforce",
        verdict_queue=verdict_q,
    ):
        if isinstance(ev, UnitStartedEvent):
            starts.append(ev)
        elif isinstance(ev, TeachingChunkEvent):
            chunks.append(ev)
        elif isinstance(ev, AssessmentResultEvent):
            assessments.append(ev)

    # Same unit taught twice with the SAME method.
    assert len(starts) == 2
    assert all(s.unit_id == "u1" for s in starts)
    assert {s.plan.methods[0].method for s in starts} == {TeachingMethod.ANALOGY}
    # First assessment is needs_reinforcement, second is pass.
    assert assessments[0].result.verdict is Verdict.NEEDS_REINFORCEMENT
    assert assessments[1].result.verdict is Verdict.PASS


# ---------- needs_different_approach path ----------

async def test_verdict_different_approach_excludes_method(session_factory):
    """First verdict 'needs_different_approach' → re-teach with a DIFFERENT method."""
    client = FakeAnthropic([
        tool_use_response("c1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy"]))),
        # The router falls back to LLM when curriculum's pick is excluded; pick
        # `visual` (a stub skill) so we don't need to script another LLM call
        # for the second teaching pass.
        text_response(json.dumps({
            "method": "visual",
            "rationale": "Switching from analogy to visual.",
        })),
    ])
    verdict_q: asyncio.Queue[str] = asyncio.Queue()
    verdict_q.put_nowait("needs_different_approach")
    verdict_q.put_nowait("pass")

    starts = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="carol",
        client=client, search_tool=_search_tool(),
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="t-different",
        verdict_queue=verdict_q,
    ):
        if isinstance(ev, UnitStartedEvent):
            starts.append(ev)

    # Two attempts on u1 with different methods.
    assert len(starts) == 2
    methods = [s.plan.methods[0].method for s in starts]
    assert methods[0] is TeachingMethod.ANALOGY
    assert methods[1] is TeachingMethod.VISUAL
    assert methods[0] != methods[1]  # core invariant: rejected method is excluded


# ---------- attempt cap ----------

async def test_verdict_attempt_cap_stops_after_max(session_factory):
    """If the user keeps clicking 'reinforce', the loop exits after MAX_TEACH_ATTEMPTS."""
    from backend.api.session import MAX_TEACH_ATTEMPTS
    client = FakeAnthropic([
        tool_use_response("c1", "web_search", {"query": "x"}),
        text_response(json.dumps(_curriculum_payload(["analogy"]))),
    ])
    verdict_q: asyncio.Queue[str] = asyncio.Queue()
    for _ in range(MAX_TEACH_ATTEMPTS):
        verdict_q.put_nowait("needs_reinforcement")

    starts = []
    async for ev in run_session_stream(
        goal=_goal(), user_id="dave",
        client=client, search_tool=_search_tool(),
        session_factory=session_factory,
        teaching_registry=build_teaching_registry(client=client),
        session_id="t-cap",
        verdict_queue=verdict_q,
    ):
        if isinstance(ev, UnitStartedEvent):
            starts.append(ev)

    # Loop honored the cap exactly.
    assert len(starts) == MAX_TEACH_ATTEMPTS


# ---------- HTTP endpoint ----------

async def test_verdict_endpoint_404_when_session_unknown():
    """POST /verdict to an unknown session returns 404, not a hang."""
    from fastapi import HTTPException

    from backend.api.session import VerdictRequest
    from backend.api.session import verdict as verdict_route
    try:
        await verdict_route("not-a-real-session", VerdictRequest(verdict="pass"))
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("expected HTTPException(404)")


async def test_verdict_endpoint_pushes_to_queue():
    """POST /verdict on an active session puts the value on the verdict queue."""
    from backend.api.session import (
        VerdictRequest,
        _pending_sessions,
        _PendingSession,
    )
    from backend.api.session import (
        verdict as verdict_route,
    )
    sid = "test-active"
    pending = _PendingSession()
    _pending_sessions[sid] = pending
    try:
        result = await verdict_route(sid, VerdictRequest(verdict="needs_reinforcement"))
        assert result == {"status": "queued"}
        # Queue now has the verdict ready to consume.
        got = pending.verdict_queue.get_nowait()
        assert got == "needs_reinforcement"
    finally:
        _pending_sessions.pop(sid, None)
