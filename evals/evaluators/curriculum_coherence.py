from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.agents._parsing import extract_text, parse_json_payload
from backend.orchestration.cache import cached_system
from backend.orchestration.ptc import AnthropicLike
from backend.schemas.curriculum import Curriculum

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

CURRICULUM_COHERENCE_JUDGE_PROMPT = """\
You are evaluating a Curriculum produced by an Adaptive Learning Agent against the \
learner's stated goal.

Score each of these axes 0.0 to 1.0:

- ordering: do units progress from prerequisites to consequences? A unit that \
requires knowledge from a later unit fails this axis.
- completeness: does the curriculum cover the sub_goal adequately? Missing core \
subtopics counts against this.
- scope: is the curriculum right-sized for the user's current_level, desired_depth, \
and time_budget_hours? Too advanced for a novice, too shallow for an expert, or \
too long/short for the time budget all count against this.
- pedagogy_fit: for each unit, is the recommended_pedagogy appropriate for the \
kind of concept? (e.g., visual for relational/topological ideas, worked_example for \
procedures, game for interactive concepts, socratic/feynman for conceptual understanding, \
retrieval for vocabulary/facts).

Overall pass threshold: mean score >= 0.75 AND no axis below 0.5.

Return JSON exactly in this shape, no prose outside the JSON:

{
  "axis_scores": [
    {"axis": "ordering", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"axis": "completeness", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"axis": "scope", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"axis": "pedagogy_fit", "score": <0.0-1.0>, "rationale": "<one sentence>"}
  ],
  "overall_score": <mean>,
  "passed": <bool>,
  "rationale": "<one-sentence summary>"
}
"""


class AxisScore(BaseModel):
    axis: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class CurriculumCoherenceResult(BaseModel):
    axis_scores: list[AxisScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    rationale: str


class CurriculumCoherenceEvalError(Exception):
    """Raised when the judge's output cannot be parsed."""


def _build_judge_user_message(curriculum: Curriculum) -> str:
    return (
        "Learner's goal:\n"
        f"{curriculum.goal.model_dump_json(indent=2)}\n\n"
        "Proposed curriculum (units only — goal shown above):\n"
        f"{curriculum.model_dump_json(include={'units'}, indent=2)}"
    )


async def evaluate_curriculum_coherence(
    curriculum: Curriculum,
    *,
    judge: AnthropicLike,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 1024,
) -> CurriculumCoherenceResult:
    response = await judge.messages.create(
        model=model,
        system=cached_system(CURRICULUM_COHERENCE_JUDGE_PROMPT),
        messages=[{"role": "user", "content": _build_judge_user_message(curriculum)}],
        max_tokens=max_tokens,
    )
    try:
        text = extract_text(response)
        payload: dict[str, Any] = parse_json_payload(text)
    except Exception as exc:
        raise CurriculumCoherenceEvalError(
            f"Judge output could not be parsed: {exc}"
        ) from exc
    try:
        return CurriculumCoherenceResult.model_validate(payload)
    except ValidationError as exc:
        raise CurriculumCoherenceEvalError(
            f"Judge output failed schema validation: {payload}"
        ) from exc
