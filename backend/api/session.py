from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from backend.agents.assessment import (
    assess_understanding,
    generate_assessment_question,
)
from backend.agents.curriculum import build_curriculum
from backend.agents.intent import IntentCompleteResponse, intent_turn
from backend.agents.memory import apply_assessment
from backend.agents.pedagogy_router import route_pedagogy
from backend.agents.teaching.game import GameAgentError
from backend.api.ops import register_live_session, unregister_live_session
from backend.orchestration.circuit_breaker import SessionCircuitBreaker
from backend.orchestration.cost import (
    CostAccumulator,
    current_accumulator,
    session_accumulator,
)
from backend.orchestration.logging import new_request_id
from backend.persistence.learner_state import load_learner_state, save_learner_state
from backend.persistence.models import SessionRunRow
from backend.persistence.session_runs import (
    finish_session_run,
    start_session_run,
)
from backend.schemas.assessment import AssessmentResult, Verdict
from backend.schemas.events import (
    AssessmentQuestionEvent,
    AssessmentResultEvent,
    CircuitBreakerTrippedEvent,
    CompleteEvent,
    CostUpdateEvent,
    CurriculumReadyEvent,
    ErrorEvent,
    IntentAskEvent,
    IntentUpdateEvent,
    SessionStartedEvent,
    SSEEvent,
    TeachingChunkEvent,
    UnitStartedEvent,
)
from backend.schemas.goal import LearningGoal
from backend.schemas.state import LearnerState
from backend.schemas.teaching import MethodCall, TeachingMethod, TeachingPlan
from backend.skills import SkillRegistry, StubSkillError

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["session"])

# Hard caps so a misbehaving model (or stuck user) can't loop forever.
MAX_INTENT_TURNS = 6
MAX_TEACH_ATTEMPTS = 3  # per unit; one initial + up to two re-teaches


@dataclass
class _PendingSession:
    """Holds async queues the SSE coroutine awaits while paused for input."""
    # User answers to intent_ask events.
    response_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    # User verdicts after each teaching_chunk: pass / needs_reinforcement /
    # needs_different_approach. Used for the click-button verdict path.
    verdict_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    # User free-text answers to assessment_question events. Used for the
    # interactive Q→A→verdict assessment path.
    assessment_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)


# Module-level: lifetime is bounded by the SSE connection that owns it.
# Cleared in the `finally` of the route handler.
_pending_sessions: dict[str, _PendingSession] = {}


class RunSessionRequest(BaseModel):
    """Either supply a parsed `goal` (skip HITL) OR a raw `initial_message`
    (run the intent-clarification loop with the user)."""
    initial_message: str | None = None
    goal: LearningGoal | None = None
    user_id: str = "default"

    @model_validator(mode="after")
    def _exactly_one(self) -> RunSessionRequest:
        if self.initial_message is None and self.goal is None:
            raise ValueError("Provide either `initial_message` or `goal`.")
        return self


class RespondRequest(BaseModel):
    answer: str = Field(min_length=1)


class VerdictRequest(BaseModel):
    verdict: Literal[
        "pass",
        "needs_reinforcement",
        "needs_different_approach",
    ]
    diagnostic_notes: str = "User verdict."
    confidence: float = 1.0


class AssessmentAnswerRequest(BaseModel):
    answer: str = Field(min_length=1)


async def _resolve_goal_via_intent(
    initial_message: str,
    *,
    client,
    session_id: str,
    response_queue: asyncio.Queue[str],
) -> AsyncIterator[SSEEvent]:
    """HITL intent loop. Yields IntentAskEvents and pauses on the queue.

    Final yield is an `IntentUpdateEvent(goal=...)` carrying the resolved goal.
    """
    conversation: list[dict[str, str]] = [
        {"role": "user", "content": initial_message},
    ]
    for _ in range(MAX_INTENT_TURNS):
        result = await intent_turn(client, conversation=conversation)
        if isinstance(result, IntentCompleteResponse):
            yield IntentUpdateEvent(
                message="Got it. Researching and planning curriculum…",
                goal=result.goal,
            )
            return
        # Asking. Yield the question, pause until user posts a response.
        yield IntentAskEvent(session_id=session_id, question=result.question)
        answer = await response_queue.get()
        # Mirror the model's question + the user's answer back into the
        # conversation so the next intent_turn has full context.
        conversation.extend([
            {
                "role": "assistant",
                "content": json.dumps(
                    {"status": "asking", "question": result.question}
                ),
            },
            {"role": "user", "content": answer},
        ])
    raise RuntimeError(
        f"Intent loop exceeded {MAX_INTENT_TURNS} turns without resolving."
    )


async def _teach_one_attempt(
    unit,
    method,
    *,
    teaching_registry: SkillRegistry,
) -> tuple[str, bool]:
    """Run one teaching attempt.

    Returns `(content, game_quality_failure)`. The second element is True only
    when the Game agent raised `GameAgentError` (PTC loop couldn't produce a
    valid artifact); False otherwise. Callers feed this into the per-session
    circuit breaker.
    """
    skill = teaching_registry.get(method.value)
    try:
        if skill is None or skill.is_stub():
            return f"[Demo {method.value} teaching for: {unit.objective}]", False
        return await skill.run(unit), False
    except StubSkillError:
        return f"[Demo {method.value} teaching for: {unit.objective}]", False
    except GameAgentError as exc:
        _logger.warning(
            "game_agent_quality_failure",
            extra={"unit_id": unit.id, "error": str(exc)},
        )
        return (
            f"[Game agent could not produce a valid artifact for: {unit.objective}]",
            True,
        )


async def _await_verdict(
    verdict_queue: asyncio.Queue[str] | None,
) -> Verdict:
    """If a queue is provided, block until the user posts a verdict.

    No queue → auto-pass (used by tests + the back-compat goal-only path)."""
    if verdict_queue is None:
        return Verdict.PASS
    raw = await verdict_queue.get()
    try:
        return Verdict(raw)
    except ValueError as exc:
        raise RuntimeError(f"Unknown verdict from user: {raw!r}") from exc


async def _interactive_assessment(
    *,
    unit,
    method,
    teaching_content: str,
    attempt: int,
    session_id: str,
    client,
    assessment_queue: asyncio.Queue[str],
) -> AsyncIterator[SSEEvent | AssessmentResult]:
    """Interactive Q→A→verdict cycle for one unit attempt.

    Yields:
      - one AssessmentQuestionEvent (the agent's probe)
      - then awaits the learner's free-text answer on assessment_queue
      - finally yields an AssessmentResult (the verdict, derived by
        assess_understanding from the learner's answer).
    """
    question = await generate_assessment_question(
        unit, method, teaching_content, client=client,
    )
    yield AssessmentQuestionEvent(
        session_id=session_id,
        unit_id=unit.id,
        question=question,
        attempt=attempt,
    )
    answer = await assessment_queue.get()
    transcript = [
        {"role": "assistant", "content": question},
        {"role": "user", "content": answer},
    ]
    result = await assess_understanding(
        unit, method, transcript, client=client,
    )
    yield result


def _game_spend_usd(acc: CostAccumulator | None) -> float:
    """Return cumulative USD spent under agent='game' in the active session."""
    if acc is None:
        return 0.0
    return acc.by_agent().get("game", 0.0)


async def run_session_stream(
    *,
    user_id: str,
    client,
    search_tool,
    session_factory,
    teaching_registry: SkillRegistry,
    session_id: str,
    response_queue: asyncio.Queue[str] | None = None,
    verdict_queue: asyncio.Queue[str] | None = None,
    assessment_queue: asyncio.Queue[str] | None = None,
    goal: LearningGoal | None = None,
    initial_message: str | None = None,
    additional_tools: list | None = None,
    circuit_breaker: SessionCircuitBreaker | None = None,
) -> AsyncIterator[SSEEvent]:
    """Orchestrate a full session: optional HITL intent → curriculum → teach → assess.

    Exactly one of `goal` or `initial_message` must be provided. When
    `verdict_queue` is supplied, the per-unit teaching loop awaits a user
    verdict after each teaching_chunk and branches on it (pass / re-teach /
    different-approach). Without a queue, the loop auto-passes (used by tests
    and any non-interactive caller)."""
    yield SessionStartedEvent(session_id=session_id)

    if goal is None:
        if initial_message is None:
            raise ValueError("run_session_stream needs goal or initial_message")
        if response_queue is None:
            raise ValueError("HITL flow requires a response_queue")
        async for ev in _resolve_goal_via_intent(
            initial_message,
            client=client,
            session_id=session_id,
            response_queue=response_queue,
        ):
            yield ev
            if isinstance(ev, IntentUpdateEvent) and ev.goal is not None:
                goal = ev.goal
        if goal is None:
            raise RuntimeError("Intent flow finished without producing a goal")
    else:
        yield IntentUpdateEvent(
            message="Goal received. Researching and planning curriculum…",
            goal=goal,
        )

    curriculum = await build_curriculum(
        goal,
        client=client,
        search_tool=search_tool,
        teaching_registry=teaching_registry,
        additional_tools=additional_tools,
    )
    yield CurriculumReadyEvent(curriculum=curriculum)

    async with session_factory() as session:
        state = (
            await load_learner_state(user_id, session=session)
            or LearnerState(user_id=user_id)
        )

    breaker = circuit_breaker or SessionCircuitBreaker()
    breaker_announced = False

    for unit in curriculum.units:
        excluded_methods = []  # methods the learner has explicitly rejected
        last_verdict = None
        for attempt in range(1, MAX_TEACH_ATTEMPTS + 1):
            plan = await route_pedagogy(
                unit, state,
                excluded_methods=excluded_methods or None,
                teaching_registry=teaching_registry,
                llm_client=client if excluded_methods else None,
            )
            method = plan.methods[0].method

            # Circuit-breaker override: if tripped, force worked_example for
            # any unit the router still wants to teach with `game`.
            tripped, reason = breaker.should_trip()
            if tripped and method is TeachingMethod.GAME:
                if not breaker_announced:
                    yield CircuitBreakerTrippedEvent(reason=reason or "tripped")
                    breaker_announced = True
                method = TeachingMethod.WORKED_EXAMPLE
                plan = TeachingPlan(
                    unit_id=unit.id,
                    methods=[MethodCall(method=method)],
                    rationale=(
                        f"Circuit-broken on Game agent ({reason}); "
                        "falling back to worked_example."
                    ),
                )

            yield UnitStartedEvent(unit_id=unit.id, plan=plan)

            # Snapshot game spend BEFORE the attempt so we can attribute the
            # delta back to this single attempt (cost accumulator is per-session).
            spend_before = _game_spend_usd(current_accumulator())
            content, game_failed = await _teach_one_attempt(
                unit, method, teaching_registry=teaching_registry,
            )
            if method is TeachingMethod.GAME:
                spend_delta = max(
                    0.0, _game_spend_usd(current_accumulator()) - spend_before,
                )
                breaker.record_game_attempt(
                    passed=not game_failed, cost_usd=spend_delta,
                )
            yield TeachingChunkEvent(
                unit_id=unit.id, method=method, content=content,
            )

            # Three paths to a verdict, in priority order:
            # 1. Interactive Q→A→assess (preferred, closes the adaptive loop)
            # 2. Click-button verdict (legacy fallback)
            # 3. Auto-pass (tests / non-interactive)
            if assessment_queue is not None:
                result: AssessmentResult | None = None
                async for ev in _interactive_assessment(
                    unit=unit, method=method, teaching_content=content,
                    attempt=attempt, session_id=session_id, client=client,
                    assessment_queue=assessment_queue,
                ):
                    if isinstance(ev, AssessmentResult):
                        result = ev
                    else:
                        yield ev
                if result is None:
                    raise RuntimeError(
                        "Interactive assessment finished without a verdict."
                    )
                verdict = result.verdict
            else:
                verdict = await _await_verdict(verdict_queue)
                notes = (
                    f"User got it on attempt {attempt}."
                    if verdict == Verdict.PASS
                    else "Learner asked for more reinforcement on the same method."
                    if verdict == Verdict.NEEDS_REINFORCEMENT
                    else f"Learner rejected {method.value}; routing to a different method."
                )
                result = AssessmentResult(
                    unit_id=unit.id,
                    verdict=verdict,
                    diagnostic_notes=notes,
                    confidence=1.0,
                )

            last_verdict = verdict
            yield AssessmentResultEvent(result=result)
            state = apply_assessment(
                state, unit=unit, method=method, result=result,
            )

            if verdict == Verdict.PASS:
                break
            if verdict == Verdict.NEEDS_DIFFERENT_APPROACH:
                excluded_methods.append(method)
            # NEEDS_REINFORCEMENT → loop with same method.
        else:
            # MAX_TEACH_ATTEMPTS exhausted without a pass; record what we got.
            _logger.info(
                "unit_attempts_exhausted",
                extra={
                    "unit_id": unit.id,
                    "max_attempts": MAX_TEACH_ATTEMPTS,
                    "last_verdict": last_verdict.value if last_verdict else None,
                },
            )

    async with session_factory() as session:
        await save_learner_state(state, session=session)

    yield CompleteEvent()


@router.post("/run")
async def run_session(request: Request, body: RunSessionRequest) -> EventSourceResponse:
    rid = new_request_id()
    session_id = secrets.token_urlsafe(8)

    client = request.app.state.anthropic
    search_tool = request.app.state.search_tool
    session_factory = request.app.state.session_factory
    teaching_registry = request.app.state.teaching_registry
    wikipedia_tool = request.app.state.wikipedia_tool

    pending = _PendingSession()
    _pending_sessions[session_id] = pending

    async def event_generator():
        # Pre-seed domain/sub_goal if the caller skipped HITL; otherwise we
        # back-fill these from the IntentUpdate event mid-stream.
        run_domain = body.goal.domain if body.goal else None
        run_sub_goal = body.goal.sub_goal if body.goal else None
        run_id: int | None = None
        unit_count = 0
        units_passed = 0
        try:
            await register_live_session(session_id, user_id=body.user_id)
            async with session_factory() as db:
                run_id = await start_session_run(
                    session=db,
                    session_id=session_id,
                    user_id=body.user_id,
                    domain=run_domain,
                    sub_goal=run_sub_goal,
                )

            if client is None:
                err = ErrorEvent(
                    message="Server missing env var: ANTHROPIC_API_KEY",
                )
                yield {"data": err.model_dump_json()}
                # Record this as an errored run so the dashboard sees it.
                async with session_factory() as db:
                    await finish_session_run(
                        session=db, run_id=run_id,
                        unit_count=0, units_passed=0,
                        total_cost_usd=0.0, total_latency_s=0.0,
                        cost_by_agent={}, latency_by_agent={},
                        error="ConfigError: ANTHROPIC_API_KEY missing",
                    )
                return
            with session_accumulator() as acc:
                error_message: str | None = None
                try:
                    async for event in run_session_stream(
                        user_id=body.user_id,
                        client=client,
                        search_tool=search_tool,
                        session_factory=session_factory,
                        teaching_registry=teaching_registry,
                        session_id=session_id,
                        response_queue=pending.response_queue,
                        verdict_queue=pending.verdict_queue,
                        assessment_queue=pending.assessment_queue,
                        goal=body.goal,
                        initial_message=body.initial_message,
                        additional_tools=[wikipedia_tool] if wikipedia_tool else None,
                        circuit_breaker=SessionCircuitBreaker(),
                    ):
                        # Back-fill goal fields once intent loop resolves them.
                        if (
                            isinstance(event, IntentUpdateEvent)
                            and event.goal is not None
                        ):
                            run_domain = event.goal.domain
                            run_sub_goal = event.goal.sub_goal
                        elif isinstance(event, CurriculumReadyEvent):
                            unit_count = len(event.curriculum.units)
                        elif isinstance(event, AssessmentResultEvent):
                            if event.result.verdict is Verdict.PASS:
                                units_passed += 1
                        yield {"data": event.model_dump_json()}
                        yield {"data": _cost_event(acc).model_dump_json()}
                except Exception as exc:  # noqa: BLE001
                    _logger.exception("session_error", extra={"request_id": rid})
                    error_message = f"{type(exc).__name__}: {exc}"
                    err = ErrorEvent(message=error_message)
                    yield {"data": err.model_dump_json()}
                    yield {"data": _cost_event(acc).model_dump_json()}
                _logger.info(
                    "session_cost",
                    extra={
                        "request_id": rid,
                        "total_usd": round(acc.total_usd, 4),
                        "by_agent": {k: round(v, 4) for k, v in acc.by_agent().items()},
                        "cache_hit_rate": round(acc.cache_hit_rate(), 3),
                        "call_count": len(acc.entries),
                    },
                )
                # Always close out the session_runs row — success or failure.
                async with session_factory() as db:
                    await finish_session_run(
                        session=db,
                        run_id=run_id,
                        unit_count=unit_count or None,
                        units_passed=units_passed,
                        total_cost_usd=round(acc.total_usd, 6),
                        total_latency_s=round(acc.total_latency_s, 3),
                        cost_by_agent={
                            k: round(v, 6) for k, v in acc.by_agent().items()
                        },
                        latency_by_agent=acc.latency_by_agent_ms(),
                        error=error_message,
                    )
                    # Domain/sub_goal may have been back-filled — refresh row.
                    if run_domain or run_sub_goal:
                        row = await db.get(SessionRunRow, run_id)
                        if row is not None:
                            row.domain = run_domain
                            row.sub_goal = run_sub_goal
                            await db.commit()
        finally:
            _pending_sessions.pop(session_id, None)
            await unregister_live_session(session_id)

    return EventSourceResponse(event_generator(), ping=2)


@router.post("/{session_id}/respond")
async def respond(session_id: str, body: RespondRequest) -> dict[str, str]:
    """Push a user answer onto the paused session's queue."""
    pending = _pending_sessions.get(session_id)
    if pending is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not active or already finished.",
        )
    await pending.response_queue.put(body.answer)
    return {"status": "queued"}


@router.post("/{session_id}/verdict")
async def verdict(session_id: str, body: VerdictRequest) -> dict[str, str]:
    """Push a user verdict onto the paused session's verdict queue."""
    pending = _pending_sessions.get(session_id)
    if pending is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not active or already finished.",
        )
    await pending.verdict_queue.put(body.verdict)
    return {"status": "queued"}


@router.post("/{session_id}/assessment-answer")
async def assessment_answer(
    session_id: str, body: AssessmentAnswerRequest,
) -> dict[str, str]:
    """Push the learner's free-text answer onto the assessment queue.

    The session's per-unit assessment loop is paused on this queue between
    `assessment_question` and `assessment_result` events.
    """
    pending = _pending_sessions.get(session_id)
    if pending is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not active or already finished.",
        )
    await pending.assessment_queue.put(body.answer)
    return {"status": "queued"}


def _cost_event(acc: CostAccumulator) -> CostUpdateEvent:
    return CostUpdateEvent(
        total_usd=round(acc.total_usd, 4),
        by_agent={k: round(v, 4) for k, v in acc.by_agent().items()},
        cache_hit_rate=round(acc.cache_hit_rate(), 3),
        call_count=len(acc.entries),
    )
