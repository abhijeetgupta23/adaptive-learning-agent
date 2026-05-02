"""Shared test fakes for Anthropic client."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return next(self._responses)


class FakeAnthropic:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _FakeMessages(responses)


def text_response(
    text: str, stop_reason: str = "end_turn", in_tok: int = 10, out_tok: int = 5
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def tool_use_response(
    tool_id: str,
    name: str,
    tool_input: dict[str, Any],
    stop_reason: str = "tool_use",
    in_tok: int = 10,
    out_tok: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


# A tag-balanced minimal HTML that satisfies archetype="drag_drop" + node --check JS.
GAME_HAPPY_HTML = (
    "<div>"
    "<style>.row{padding:8px}</style>"
    '<div draggable="true" id="r1">Row 1</div>'
    '<div id="inner" data-drop="inner">INNER</div>'
    '<div id="left" data-drop="left">LEFT</div>'
    "<script>let mode='inner';</script>"
    "</div>"
)


def game_happy_path_responses(
    *,
    html: str = GAME_HAPPY_HTML,
    archetype: str = "drag_drop",
    title: str = "Test Game",
    description: str = "A test game",
    instructions: str = "Tap things to learn",
    learning_objective: str = "Distinguish INNER vs LEFT joins",
) -> list[SimpleNamespace]:
    """Scripts the FakeAnthropic message sequence for a full happy-path PTC run.

    Six tool calls + one end_turn text response. Use with FakeAnthropic.
    """
    return [
        tool_use_response("t1", "commit_learning_link", {
            "learning_objective": learning_objective,
            "learning_mechanic": "classify",
            "game_mechanic": "tap or drag to choose join type",
        }),
        tool_use_response("t2", "commit_core_loop", {
            "player_action": "Drag a row to a join bucket",
            "system_response": "Show whether the row would survive the join",
            "feedback": "Green check or red cross with explanation",
            "win_condition": "Correctly classify all 5 row pairs",
        }),
        tool_use_response("t3", "commit_aesthetic", {
            "intended_feeling": "mastery",
            "ui_archetype": archetype,
        }),
        tool_use_response("t4", "validate_html", {"html": html}),
        tool_use_response("t5", "validate_js", {"js": "let mode='inner';"}),
        tool_use_response("t6", "submit_game", {
            "title": title,
            "description": description,
            "instructions": instructions,
            "html": html,
        }),
        text_response("Game submitted.", stop_reason="end_turn"),
    ]
