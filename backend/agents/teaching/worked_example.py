"""Worked-example teaching agent: produces a runnable, well-explained walkthrough."""
from __future__ import annotations

import logging

from backend.agents._parsing import StructuredOutputError, extract_text
from backend.orchestration.cache import cached_system
from backend.orchestration.cost import CallTimer, record_usage
from backend.orchestration.ptc import AnthropicLike
from backend.schemas.curriculum import LearningUnit

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

WORKED_EXAMPLE_SYSTEM_PROMPT = """\
You are the Worked Example Teaching Agent. Produce a concrete, runnable walkthrough \
that teaches the learning unit by *example*. The learner should be able to copy \
the code and run it.

Format your response as MARKDOWN. Structure:

1. **One-paragraph framing** — name the concept and the example you'll use.
2. **The code**, in a single ```language``` fenced block. Annotate with line \
comments where the subtlety lives.
3. **Step-by-step trace** — walk through what happens when the code runs, \
in numbered steps, calling out the surprising bits.
4. **Why this matters** — one paragraph on the underlying principle the example \
illustrates, and the common gotcha you'd hit if you forgot it.

Constraints:
- Self-contained code that runs without setup beyond the standard library.
- ~25-60 lines of code (long enough to be meaningful, short enough to read).
- Output MARKDOWN ONLY. No surrounding JSON, no preamble like "Here is...".
- Use ```python or ```javascript or ```sql etc. — pick the right language.
"""


class WorkedExampleAgentError(Exception):
    """Raised when the worked-example agent's output cannot be extracted."""


def _build_user_message(unit: LearningUnit) -> str:
    return (
        "Build a worked example for this learning unit:\n\n"
        f"Unit id: {unit.id}\n"
        f"Objective: {unit.objective}\n"
    )


async def generate_worked_example(
    unit: LearningUnit,
    *,
    client: AnthropicLike,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    with CallTimer() as _t:
        response = await client.messages.create(
            model=model,
            system=cached_system(WORKED_EXAMPLE_SYSTEM_PROMPT),
            messages=[{"role": "user", "content": _build_user_message(unit)}],
            max_tokens=max_tokens,
        )
    record_usage(
        model=model,
        agent="worked_example",
        response=response,
        latency_ms=_t.elapsed_ms,
    )
    try:
        text = extract_text(response)
    except StructuredOutputError as exc:
        raise WorkedExampleAgentError(str(exc)) from exc
    if not text.strip():
        raise WorkedExampleAgentError("Worked-example agent returned empty text")
    _logger.info(
        "worked_example_generated",
        extra={"unit_id": unit.id, "chars": len(text)},
    )
    return text
