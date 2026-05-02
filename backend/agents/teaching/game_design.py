"""MDA + serious-games-grounded design state and tools for the Game Agent.

Decomposition references:
  - Hunicke et al. "MDA: A Formal Approach to Game Design and Game Research" (2004).
    Mechanics → Dynamics → Aesthetics; designer writes only mechanics.
  - Plass / Kapp et al. on serious-games methodology: the **learning mechanic ↔
    game mechanic** bridge is the #1 failure mode of educational games.

The PTC tools below force the agent to commit to (1) the learning link,
(2) the core gameplay loop, and (3) the aesthetic before writing a single tag
of HTML — and then require validators to pass before the game is submitted.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from backend.agents.teaching.game_validators import (
    validate_html_string,
    validate_js_string,
)
from backend.orchestration.tools import Tool, ToolRegistry

# UI archetypes — small enum to constrain the layout space the LLM searches.
UI_ARCHETYPES = ["grid", "drag_drop", "button_array", "canvas", "list"]


@dataclass
class GameDesignState:
    """Mutated by tool handlers across PTC iterations."""
    learning_link: dict[str, str] | None = None
    core_loop: dict[str, str] | None = None
    aesthetic: dict[str, str] | None = None
    html_validated: bool = False
    js_validated: bool = False
    submitted: dict[str, Any] | None = None
    # log of tool-call order for debugging / portfolio surfacing
    call_log: list[str] = field(default_factory=list)

    def design_complete(self) -> bool:
        return (
            self.learning_link is not None
            and self.core_loop is not None
            and self.aesthetic is not None
        )


# ---------- tool handlers ----------

def _ok(msg: str) -> str:
    return msg


def _err(msg: str) -> str:
    # Returned to the LLM as the tool result; it should self-correct.
    return f"ERROR: {msg}"


def _commit_learning_link_handler(state: GameDesignState):
    async def handler(inp: dict[str, Any]) -> str:
        state.call_log.append("commit_learning_link")
        for k in ("learning_objective", "learning_mechanic", "game_mechanic"):
            if not isinstance(inp.get(k), str) or not inp[k].strip():
                return _err(
                    f"Field {k!r} is required and must be a non-empty string."
                )
        state.learning_link = {
            "learning_objective": inp["learning_objective"].strip(),
            "learning_mechanic": inp["learning_mechanic"].strip(),
            "game_mechanic": inp["game_mechanic"].strip(),
        }
        return _ok(
            "Learning link committed. Next: commit_core_loop with the four "
            "fields {player_action, system_response, feedback, win_condition}."
        )
    return handler


def _commit_core_loop_handler(state: GameDesignState):
    async def handler(inp: dict[str, Any]) -> str:
        state.call_log.append("commit_core_loop")
        if state.learning_link is None:
            return _err(
                "Call commit_learning_link first — the core loop must serve "
                "the learning mechanic you commit there."
            )
        for k in ("player_action", "system_response", "feedback", "win_condition"):
            if not isinstance(inp.get(k), str) or not inp[k].strip():
                return _err(f"Field {k!r} is required (non-empty string).")
        state.core_loop = {k: inp[k].strip() for k in (
            "player_action", "system_response", "feedback", "win_condition",
        )}
        return _ok(
            "Core loop committed. Next: commit_aesthetic with "
            "{intended_feeling, ui_archetype}. ui_archetype must be one of "
            f"{UI_ARCHETYPES}."
        )
    return handler


def _commit_aesthetic_handler(state: GameDesignState):
    async def handler(inp: dict[str, Any]) -> str:
        state.call_log.append("commit_aesthetic")
        if state.core_loop is None:
            return _err(
                "Call commit_core_loop first; aesthetic depends on the loop."
            )
        feeling = inp.get("intended_feeling")
        archetype = inp.get("ui_archetype")
        if not isinstance(feeling, str) or not feeling.strip():
            return _err("Field 'intended_feeling' required (non-empty string).")
        if archetype not in UI_ARCHETYPES:
            return _err(
                f"ui_archetype must be one of {UI_ARCHETYPES}; got {archetype!r}."
            )
        state.aesthetic = {
            "intended_feeling": feeling.strip(),
            "ui_archetype": archetype,
        }
        return _ok(
            f"Aesthetic committed (archetype={archetype}). Now write the "
            "self-contained HTML and call validate_html({html}). After that, "
            "validate_js({js}) on the inline <script> body."
        )
    return handler


def _validate_html_handler(state: GameDesignState):
    async def handler(inp: dict[str, Any]) -> str:
        state.call_log.append("validate_html")
        if state.aesthetic is None:
            return _err("Call commit_aesthetic first so we know the archetype.")
        html = inp.get("html")
        if not isinstance(html, str) or len(html) < 20:
            return _err("Field 'html' is required and must be a non-trivial string.")
        archetype = state.aesthetic["ui_archetype"]
        errors = validate_html_string(html, archetype=archetype)
        if errors:
            state.html_validated = False
            return _err(
                "HTML validation failed:\n- " + "\n- ".join(errors[:6])
            )
        state.html_validated = True
        return _ok("HTML validated. Now call validate_js with the inline script body.")
    return handler


def _validate_js_handler(state: GameDesignState):
    async def handler(inp: dict[str, Any]) -> str:
        state.call_log.append("validate_js")
        js = inp.get("js")
        if not isinstance(js, str):
            return _err("Field 'js' is required (string).")
        err = await validate_js_string(js)
        if err:
            state.js_validated = False
            return _err(f"JS syntax error from `node --check`:\n{err}")
        state.js_validated = True
        return _ok("JS validated. Call submit_game when ready.")
    return handler


def _submit_game_handler(state: GameDesignState):
    async def handler(inp: dict[str, Any]) -> str:
        state.call_log.append("submit_game")
        missing = []
        if not state.design_complete():
            missing.append(
                "design commits (learning_link, core_loop, aesthetic)"
            )
        if not state.html_validated:
            missing.append("validate_html must have passed")
        if not state.js_validated:
            missing.append("validate_js must have passed")
        if missing:
            return _err("Cannot submit: " + "; ".join(missing) + ".")

        for k in ("title", "description", "instructions", "html"):
            if not isinstance(inp.get(k), str) or not inp[k].strip():
                return _err(f"Field {k!r} is required (non-empty string).")
        # Re-run validators on the submitted html as a final guard.
        archetype = state.aesthetic["ui_archetype"]  # type: ignore[index]
        html_errors = validate_html_string(inp["html"], archetype=archetype)
        if html_errors:
            return _err(
                "Final HTML validation failed:\n- " + "\n- ".join(html_errors[:6])
            )
        scripts = re.findall(
            r"<script\b[^>]*>(.*?)</script>",
            inp["html"], re.IGNORECASE | re.DOTALL,
        )
        for idx, js_body in enumerate(scripts):
            if not js_body.strip():
                continue
            err = await validate_js_string(js_body)
            if err:
                return _err(
                    f"Final JS validation failed on <script #{idx + 1}>: {err}"
                )

        state.submitted = {
            "title": inp["title"].strip(),
            "description": inp["description"].strip(),
            "instructions": inp["instructions"].strip(),
            "html": inp["html"],
        }
        return _ok(
            "Game submitted successfully. You can now stop calling tools and "
            "return a brief confirmation message."
        )
    return handler


# ---------- registry builder ----------

def build_game_design_registry(state: GameDesignState) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="commit_learning_link",
        description=(
            "FIRST CALL. Commit how the game's mechanic delivers the unit's "
            "learning objective. This is the bridge that makes the game "
            "*educational* rather than just fun. Per Plass/Kapp serious-games "
            "literature, missing this bridge is the #1 failure mode."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "learning_objective": {
                    "type": "string",
                    "description": "Restate the unit's objective in one sentence.",
                },
                "learning_mechanic": {
                    "type": "string",
                    "description": (
                        "The cognitive operation the player must perform "
                        "(e.g. 'predict', 'classify', 'order', 'simulate')."
                    ),
                },
                "game_mechanic": {
                    "type": "string",
                    "description": (
                        "The concrete game action that triggers the learning "
                        "mechanic (e.g. 'tap to predict next coroutine')."
                    ),
                },
            },
            "required": ["learning_objective", "learning_mechanic", "game_mechanic"],
        },
        handler=_commit_learning_link_handler(state),
    ))
    reg.register(Tool(
        name="commit_core_loop",
        description=(
            "SECOND CALL. Commit the four-element core gameplay loop. Each "
            "field is one short sentence. The loop should answer one precise "
            "question (per game-jam GDD practice)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "player_action": {"type": "string"},
                "system_response": {"type": "string"},
                "feedback": {"type": "string"},
                "win_condition": {"type": "string"},
            },
            "required": [
                "player_action", "system_response", "feedback", "win_condition",
            ],
        },
        handler=_commit_core_loop_handler(state),
    ))
    reg.register(Tool(
        name="commit_aesthetic",
        description=(
            "THIRD CALL. Commit the MDA aesthetic — what should the player "
            "feel — and pick a ui_archetype that constrains the layout space."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "intended_feeling": {
                    "type": "string",
                    "description": (
                        "1-2 words: e.g. 'aha', 'mastery', 'curiosity', "
                        "'flow', 'surprise'."
                    ),
                },
                "ui_archetype": {
                    "type": "string",
                    "enum": UI_ARCHETYPES,
                    "description": (
                        "Layout pattern. grid=2D cells; drag_drop=tiles "
                        "between containers; button_array=labeled buttons "
                        "with state; canvas=<canvas> with mouse/touch; "
                        "list=ordered/reorderable items."
                    ),
                },
            },
            "required": ["intended_feeling", "ui_archetype"],
        },
        handler=_commit_aesthetic_handler(state),
    ))
    reg.register(Tool(
        name="validate_html",
        description=(
            "Statically validate an HTML draft. Checks tag balance and that "
            "the chosen ui_archetype's expected elements are present. Call "
            "this AFTER the three commit_* tools and BEFORE writing JS."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "html": {"type": "string", "description": "Full HTML to validate."},
            },
            "required": ["html"],
        },
        handler=_validate_html_handler(state),
    ))
    reg.register(Tool(
        name="validate_js",
        description=(
            "Validate a JavaScript body via `node --check`. Use this on the "
            "inline <script> contents. Returns syntax errors if any."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "js": {"type": "string", "description": "JS body, no <script> tags."},
            },
            "required": ["js"],
        },
        handler=_validate_js_handler(state),
    ))
    reg.register(Tool(
        name="submit_game",
        description=(
            "FINAL CALL. Submit the finished game. Will fail unless all "
            "three commit_* tools and both validate_* tools have passed. "
            "After this returns success, stop calling tools."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "instructions": {"type": "string"},
                "html": {"type": "string"},
            },
            "required": ["title", "description", "instructions", "html"],
        },
        handler=_submit_game_handler(state),
    ))
    return reg


# Re-export for convenience
__all__ = [
    "UI_ARCHETYPES",
    "GameDesignState",
    "build_game_design_registry",
]


# Module-level guard so we can import asyncio lazily
_ = asyncio  # silence unused import; asyncio is used inside handlers via subprocess
