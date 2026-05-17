from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from backend.agents._parsing import (
    StructuredOutputError,
    extract_text,
    parse_json_payload,
)
from backend.orchestration.cache import cached_system
from backend.orchestration.cost import CallTimer, record_usage
from backend.orchestration.ptc import AnthropicLike
from backend.schemas.assessment import AssessmentResult
from backend.schemas.curriculum import LearningUnit
from backend.schemas.teaching import TeachingMethod

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

ASSESSMENT_SYSTEM_PROMPT = """\
You are the Assessment Agent. Judge whether the learner demonstrated understanding \
of the unit's objective — not whether they got a "right answer," but whether their \
reasoning shows they actually understand.

Rubric:
- pass: learner transferred the concept (applied it to a new case, explained \
accurately in their own words, or solved a problem without hints).
- needs_reinforcement: learner got the basic idea but with gaps, hesitation, or \
minor errors that suggest fragile understanding. Same method, more practice.
- needs_different_approach: learner is still confused about the core idea. The \
current method is not landing — switch to a different pedagogy.

confidence: 0.0 to 1.0 — how confident are you in this verdict given the evidence?

Return JSON exactly, no prose outside:

{
  "verdict": "pass|needs_reinforcement|needs_different_approach",
  "diagnostic_notes": "<1-3 sentences on what the learner showed>",
  "confidence": <0.0-1.0>
}
"""

ASSESSMENT_QUESTION_SYSTEM_PROMPT = """\
You are the Assessment Agent's question-writer. Given a unit's objective and the \
teaching content the learner just saw, write ONE probing question that surfaces \
whether they actually understood the *core* of the unit, not the surface.

Constraints:
- ONE question, 1-3 sentences. No multi-part with sub-bullets.
- Open-ended: avoid yes/no. Ask them to *apply*, *predict*, *trace*, or *explain*.
- Specific to the unit: reference a concrete element from the teaching content \
when possible (a variable name, a step in a trace, a piece of code).
- Calibrated to the unit's depth: novice questions are recall + simple apply; \
intermediate are trace + predict; expert are extend + critique.
- DO NOT prepend "Question:" or any framing. Just the question text.

Return PLAIN TEXT only — no JSON, no markdown, no quotes around the question.
"""


class AssessmentAgentError(Exception):
    """Raised when the Assessment Agent's output cannot be parsed."""


def _format_transcript(transcript: list[dict[str, str]]) -> str:
    lines = []
    for msg in transcript:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _build_user_message(
    unit: LearningUnit,
    method: TeachingMethod,
    transcript: list[dict[str, str]],
) -> str:
    return (
        f"Unit objective: {unit.objective}\n"
        f"Method used: {method.value}\n\n"
        "Teaching interaction transcript:\n"
        f"{_format_transcript(transcript)}"
    )


async def generate_assessment_question(
    unit: LearningUnit,
    method: TeachingMethod,
    teaching_content: str,
    *,
    client: AnthropicLike,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
) -> str:
    """Produce one probing question for the learner about this unit.

    The question is asked AFTER the teaching has been presented; the learner's
    answer is then evaluated by `assess_understanding`. Together they form the
    interactive-assessment Q→A→verdict loop.
    """
    user_msg = (
        f"Unit objective: {unit.objective}\n"
        f"Method used: {method.value}\n\n"
        "Teaching content the learner just saw (truncated):\n"
        f"{teaching_content[:2000]}"
    )
    with CallTimer() as _t:
        response = await client.messages.create(
            model=model,
            system=cached_system(ASSESSMENT_QUESTION_SYSTEM_PROMPT),
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
        )
    record_usage(
        model=model, agent="assessment", response=response, latency_ms=_t.elapsed_ms,
    )
    try:
        text = extract_text(response).strip()
    except StructuredOutputError as exc:
        raise AssessmentAgentError(str(exc)) from exc
    if not text:
        raise AssessmentAgentError("Question agent returned empty text.")
    # Strip surrounding quotes if the model wrapped it.
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    _logger.info(
        "assessment_question_generated",
        extra={"unit_id": unit.id, "chars": len(text)},
    )
    return text


async def assess_understanding(
    unit: LearningUnit,
    method: TeachingMethod,
    transcript: list[dict[str, str]],
    *,
    client: AnthropicLike,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> AssessmentResult:
    with CallTimer() as _t:
        response = await client.messages.create(
            model=model,
            system=cached_system(ASSESSMENT_SYSTEM_PROMPT),
            messages=[{
                "role": "user",
                "content": _build_user_message(unit, method, transcript),
            }],
            max_tokens=max_tokens,
        )
    record_usage(
        model=model, agent="assessment", response=response, latency_ms=_t.elapsed_ms,
    )
    try:
        text = extract_text(response)
        payload: dict[str, Any] = parse_json_payload(text)
    except StructuredOutputError as exc:
        raise AssessmentAgentError(str(exc)) from exc

    payload["unit_id"] = unit.id  # stamp server-side
    try:
        result = AssessmentResult.model_validate(payload)
    except ValidationError as exc:
        raise AssessmentAgentError(f"Assessment failed schema validation: {exc}") from exc

    _logger.info(
        "assessment",
        extra={
            "unit_id": unit.id,
            "verdict": result.verdict.value,
            "confidence": result.confidence,
        },
    )
    return result
