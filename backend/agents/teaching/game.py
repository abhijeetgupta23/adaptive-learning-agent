"""Game Teaching Agent — drives a PTC tool-use loop to design then implement.

The flow forces design-before-code via three commit_* tools (learning_link,
core_loop, aesthetic), then validate_html / validate_js / submit_game. See
backend/agents/teaching/game_design.py for the per-tool rationale grounded
in MDA (Hunicke 2004) and serious-games methodology (Plass / Kapp).
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from backend.agents.teaching.game_design import (
    GameDesignState,
    build_game_design_registry,
)
from backend.agents.teaching.game_validators import (
    validate_html_string,
    validate_js_string,
)
from backend.orchestration.middleware import AnthropicLike
from backend.orchestration.ptc import PTCError, run_ptc
from backend.schemas.curriculum import LearningUnit

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

GAME_SYSTEM_PROMPT = """\
You are the Game Teaching Agent. You generate a playable micro-game that \
teaches the given learning unit. You design before you code.

You have six tools, **call them in this order**:

1. `commit_learning_link` — articulate the learning_objective, learning_mechanic, \
and the game_mechanic that delivers it. The game must teach via interaction, not \
narration; this tool forces the bridge.
2. `commit_core_loop` — the four-element gameplay loop: player_action, \
system_response, feedback, win_condition. One short sentence each.
3. `commit_aesthetic` — intended_feeling (e.g. mastery, aha, curiosity) + \
ui_archetype, one of: grid, drag_drop, button_array, canvas, list.
4. `validate_html` — call after writing your HTML draft. Statically checks tag \
balance and archetype-expected elements. Iterate until errors are gone.
5. `validate_js` — call on your inline <script> body (no <script> tags). Runs \
`node --check`. Iterate until clean.
6. `submit_game` — final submission. Will fail unless 1-5 have all succeeded.

After submit_game returns success, stop calling tools and reply with a one-line \
acknowledgement.

Implementation constraints (apply when you write HTML):
- Self-contained: one outer <div>, inline <style> and <script>, no external \
resources, no fetch.
- Vanilla HTML/CSS/JS only — no React, jQuery, no libraries.
- DO NOT use inline `onclick="..."` HTML attributes; bind handlers via \
`addEventListener` from the script body. (Inline handlers reference identifiers \
in global scope; if the script has any error your buttons silently die.)
- Mobile-friendly tap targets, responsive layout.
- 3-10 minutes of play; clear win condition; visible feedback per action.
"""


class GameSpec(BaseModel):
    unit_id: str
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    html: str = Field(min_length=10)
    # Design metadata committed via the PTC tools — surfaced for transparency.
    learning_link: dict[str, str] | None = None
    core_loop: dict[str, str] | None = None
    aesthetic: dict[str, str] | None = None
    tool_call_log: list[str] = Field(default_factory=list)


class GameAgentError(Exception):
    """Raised when the Game Agent's PTC loop cannot produce a valid game."""


def _build_user_message(unit: LearningUnit) -> str:
    return (
        "Design and build a micro-game that teaches this learning unit:\n\n"
        f"Unit id: {unit.id}\n"
        f"Objective: {unit.objective}\n\n"
        "Start by calling commit_learning_link."
    )


async def generate_game(
    unit: LearningUnit,
    *,
    client: AnthropicLike,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16384,
    max_iterations: int = 12,
) -> GameSpec:
    """Drive a PTC loop with design + validation tools to produce a game."""
    state = GameDesignState()
    registry = build_game_design_registry(state)

    try:
        result = await run_ptc(
            client,
            model=model,
            system=GAME_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(unit)}],
            tools=registry,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            agent="game",
        )
    except PTCError as exc:
        raise GameAgentError(
            f"PTC loop failed: {exc}. call_log={state.call_log}"
        ) from exc

    # record_usage already happened inside run_ptc per iteration; nothing to do here.
    _ = result

    if state.submitted is None:
        raise GameAgentError(
            "Game agent finished without calling submit_game. "
            f"call_log={state.call_log}"
        )

    spec = GameSpec(
        unit_id=unit.id,
        title=state.submitted["title"],
        description=state.submitted["description"],
        concept=state.learning_link["learning_objective"]  # type: ignore[index]
            if state.learning_link else state.submitted["description"],
        instructions=state.submitted["instructions"],
        html=state.submitted["html"],
        learning_link=state.learning_link,
        core_loop=state.core_loop,
        aesthetic=state.aesthetic,
        tool_call_log=list(state.call_log),
    )

    _logger.info(
        "game_generated",
        extra={
            "unit_id": unit.id,
            "html_bytes": len(spec.html),
            "tool_calls": len(state.call_log),
            "ui_archetype": (state.aesthetic or {}).get("ui_archetype"),
        },
    )
    return spec


# ---- back-compat surface for tests / external callers ----

# Keep the old function names resolvable for any tests still importing them.
_SCRIPT_RE = re.compile(
    r"<script\b[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL,
)


async def validate_inline_js(html: str, *, timeout: float = 8.0) -> str | None:
    """Back-compat: extract inline scripts and validate each."""
    for idx, js in enumerate(_SCRIPT_RE.findall(html)):
        if not js.strip():
            continue
        err = await validate_js_string(js, timeout=timeout)
        if err is not None:
            return f"<script #{idx + 1}> failed `node --check`:\n{err}"
    return None


__all__ = [
    "GAME_SYSTEM_PROMPT",
    "GameAgentError",
    "GameSpec",
    "generate_game",
    "validate_html_string",
    "validate_inline_js",
    "validate_js_string",
]
