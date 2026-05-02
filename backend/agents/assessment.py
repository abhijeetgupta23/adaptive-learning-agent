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
from backend.orchestration.cost import record_usage
from backend.orchestration.ptc import AnthropicLike
from backend.schemas.assessment import AssessmentResult
from backend.schemas.curriculum import LearningUnit
from backend.schemas.teaching import TeachingMethod

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

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


async def assess_understanding(
    unit: LearningUnit,
    method: TeachingMethod,
    transcript: list[dict[str, str]],
    *,
    client: AnthropicLike,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> AssessmentResult:
    response = await client.messages.create(
        model=model,
        system=cached_system(ASSESSMENT_SYSTEM_PROMPT),
        messages=[{
            "role": "user",
            "content": _build_user_message(unit, method, transcript),
        }],
        max_tokens=max_tokens,
    )
    record_usage(model=model, agent="assessment", response=response)
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
