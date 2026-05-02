from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.agents._parsing import extract_text, parse_json_payload
from backend.orchestration.cache import cached_system
from backend.orchestration.ptc import AnthropicLike
from backend.schemas.goal import LearningGoal

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

INTENT_FIDELITY_JUDGE_PROMPT = """\
You are evaluating an Intent Agent's extraction of a structured learning goal.

You will see two LearningGoal objects: the expected goal (ground truth) and the \
predicted goal produced by the agent. Score each field 0.0 to 1.0 by this rubric:

- domain: semantic equivalence. Exact wording not required; same practical scope = 1.0, \
related but too broad or too narrow = 0.5, wrong area = 0.0.
- sub_goal: semantic equivalence. Same specific learning target = 1.0, related but \
materially different scope = 0.5, wrong target = 0.0.
- current_level: exact enum match = 1.0, off by one level = 0.5, else 0.0.
- desired_depth: exact enum match = 1.0, off by one = 0.5, else 0.0.
- time_budget_hours: within 30% = 1.0, within 50% = 0.5, else 0.0. If expected is 0, \
match only if predicted is 0.
- preferred_modalities: Jaccard similarity of the two sets. If both empty, 1.0.

Overall pass threshold: mean(field_scores) >= 0.8 AND every field >= 0.5.

Return JSON exactly in this shape, no prose outside the JSON:

{
  "field_scores": [
    {"field": "domain", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"field": "sub_goal", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"field": "current_level", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"field": "desired_depth", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"field": "time_budget_hours", "score": <0.0-1.0>, "rationale": "<one sentence>"},
    {"field": "preferred_modalities", "score": <0.0-1.0>, "rationale": "<one sentence>"}
  ],
  "overall_score": <mean of field_scores>,
  "passed": <bool>,
  "rationale": "<one-sentence summary>"
}
"""


class FieldScore(BaseModel):
    field: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class IntentFidelityResult(BaseModel):
    field_scores: list[FieldScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    rationale: str


class IntentFidelityEvalError(Exception):
    """Raised when the judge's output cannot be parsed."""


def _build_judge_user_message(expected: LearningGoal, predicted: LearningGoal) -> str:
    return (
        "Expected goal:\n"
        f"{expected.model_dump_json(indent=2)}\n\n"
        "Predicted goal:\n"
        f"{predicted.model_dump_json(indent=2)}"
    )


async def evaluate_intent_fidelity(
    *,
    expected: LearningGoal,
    predicted: LearningGoal,
    judge: AnthropicLike,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 1024,
) -> IntentFidelityResult:
    response = await judge.messages.create(
        model=model,
        system=cached_system(INTENT_FIDELITY_JUDGE_PROMPT),
        messages=[{"role": "user", "content": _build_judge_user_message(expected, predicted)}],
        max_tokens=max_tokens,
    )
    try:
        text = extract_text(response)
        payload: dict[str, Any] = parse_json_payload(text)
    except Exception as exc:
        raise IntentFidelityEvalError(f"Judge output could not be parsed: {exc}") from exc
    try:
        return IntentFidelityResult.model_validate(payload)
    except ValidationError as exc:
        raise IntentFidelityEvalError(
            f"Judge output failed schema validation: {payload}"
        ) from exc


def load_golden_intent_cases(directory: str) -> list[dict[str, Any]]:
    """Load all intent golden cases from a directory of JSON files."""
    import os

    cases: list[dict[str, Any]] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            cases.append(json.load(f))
    return cases
