from __future__ import annotations

import logging
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from backend.agents._parsing import (
    StructuredOutputError,
    extract_text,
    parse_json_payload,
)
from backend.orchestration.cache import cached_system
from backend.orchestration.cost import record_usage
from backend.orchestration.ptc import AnthropicLike
from backend.schemas.goal import LearningGoal

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

INTENT_SYSTEM_PROMPT = """\
You are the Intent Agent for an adaptive learning platform. Your job is to interview \
the user, then finalize a structured LearningGoal.

Required fields you must produce:
- domain: broader subject area (e.g., "databases", "italian language", "machine learning")
- sub_goal: the specific thing they want to learn
- current_level: one of "novice" | "beginner" | "intermediate" | "advanced"
- desired_depth: one of "cocktail" | "working" | "expert"
    - cocktail: can hold a conversation about it
    - working: can apply it day-to-day
    - expert: can teach or build with it professionally
- time_budget_hours: positive float. Total hours the user is willing to invest. \
Infer reasonable defaults from phrases like "a weekend" (~10h), "this month" (~30h), \
"over the summer" (~80h).
- preferred_modalities: subset of ["visual", "verbal", "interactive"]. Default to [] \
if the user has not indicated a preference.

Conversation principles:
- Be conversational and brief. Probe only what you genuinely need.
- Combine questions when natural.
- Do not require explicit answers for obvious inferences.
- Finalize as soon as you can make sensible defaults for the rest; do not keep \
asking for nice-to-haves.

Output ONE of these JSON shapes, and nothing else. No prose, no code fences, \
no explanation around the JSON:

{"status": "asking", "question": "<your question>"}

OR

{"status": "complete", "goal": {
  "domain": "...",
  "sub_goal": "...",
  "current_level": "novice|beginner|intermediate|advanced",
  "desired_depth": "cocktail|working|expert",
  "time_budget_hours": <float>,
  "preferred_modalities": ["visual"|"verbal"|"interactive"]
}}
"""


class IntentAskResponse(BaseModel):
    status: Literal["asking"]
    question: str = Field(min_length=1)


class IntentCompleteResponse(BaseModel):
    status: Literal["complete"]
    goal: LearningGoal


IntentTurnResult = Annotated[
    IntentAskResponse | IntentCompleteResponse,
    Field(discriminator="status"),
]

IntentTurnAdapter: TypeAdapter[IntentTurnResult] = TypeAdapter(IntentTurnResult)


class IntentAgentError(Exception):
    """Raised when the Intent Agent's output cannot be parsed."""


async def intent_turn(
    client: AnthropicLike,
    *,
    conversation: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> IntentTurnResult:
    """Run one Intent Agent turn. Either asks a question or finalizes the goal."""
    response = await client.messages.create(
        model=model,
        system=cached_system(INTENT_SYSTEM_PROMPT),
        messages=conversation,
        max_tokens=max_tokens,
    )
    record_usage(model=model, agent="intent", response=response)
    try:
        text = extract_text(response)
        payload = parse_json_payload(text)
    except StructuredOutputError as exc:
        raise IntentAgentError(str(exc)) from exc
    try:
        result = IntentTurnAdapter.validate_python(payload)
    except ValidationError as exc:
        raise IntentAgentError(
            f"Intent response failed schema validation: {payload}"
        ) from exc
    _logger.info(
        "intent_turn",
        extra={
            "status": result.status,
            "turn_count": len(conversation),
        },
    )
    return result
