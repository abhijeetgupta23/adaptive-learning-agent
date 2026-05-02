import json
import shutil

import pytest

from backend.agents.teaching.game import (
    GameAgentError,
    GameSpec,
    generate_game,
    validate_inline_js,
)
from backend.agents.teaching.game_design import (
    UI_ARCHETYPES,
    GameDesignState,
    build_game_design_registry,
)
from backend.agents.teaching.game_validators import (
    validate_html_string,
    validate_js_string,
)
from backend.schemas import GroundingSource, LearningUnit, TeachingMethod
from evals.evaluators.game_quality import (
    GameQualityEvalError,
    evaluate_game_quality,
)
from tests._fakes import (
    GAME_HAPPY_HTML,
    FakeAnthropic,
    game_happy_path_responses,
    text_response,
    tool_use_response,
)

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not installed",
)


def _unit() -> LearningUnit:
    return LearningUnit(
        id="u2",
        objective="Distinguish INNER vs LEFT joins",
        recommended_pedagogy=TeachingMethod.GAME,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )


# ============================================================================
# game_validators
# ============================================================================

def test_validate_html_passes_balanced_markup():
    html = "<div><span>x</span><br></div>"
    assert validate_html_string(html) == []


def test_validate_html_flags_unclosed_tag():
    errors = validate_html_string("<div><span>x</div>")
    assert errors and any("span" in e.lower() for e in errors)


def test_validate_html_skips_inside_script_blocks():
    """Tag-like patterns inside <script> shouldn't break balance."""
    html = "<div><script>const x = '<not-a-tag>'</script></div>"
    assert validate_html_string(html) == []


def test_validate_html_archetype_drag_drop_requires_draggable():
    html = "<div><button>x</button></div>"
    errors = validate_html_string(html, archetype="drag_drop")
    assert errors and any("draggable" in e for e in errors)


def test_validate_html_archetype_drag_drop_passes_with_draggable():
    html = '<div><div draggable="true">x</div></div>'
    assert validate_html_string(html, archetype="drag_drop") == []


def test_validate_html_archetype_button_array_needs_buttons():
    html = "<div><span>only one button:</span><button>x</button></div>"
    errors = validate_html_string(html, archetype="button_array")
    assert errors and any("button" in e.lower() for e in errors)


def test_validate_html_archetype_canvas_needs_canvas():
    html = "<div>no canvas here</div>"
    errors = validate_html_string(html, archetype="canvas")
    assert errors and any("canvas" in e.lower() for e in errors)


@requires_node
async def test_validate_js_passes_clean_js():
    assert await validate_js_string("let x = 1; const f = () => x + 1;") is None


@requires_node
async def test_validate_js_flags_syntax_error():
    err = await validate_js_string("let x = 1 +;")
    assert err is not None


# ============================================================================
# game_design tools — exercising the registry directly (no LLM)
# ============================================================================

async def test_commit_learning_link_stores_state():
    state = GameDesignState()
    reg = build_game_design_registry(state)
    out = await reg.execute("commit_learning_link", {
        "learning_objective": "obj",
        "learning_mechanic": "classify",
        "game_mechanic": "tap",
    })
    assert "committed" in out.lower()
    assert state.learning_link == {
        "learning_objective": "obj",
        "learning_mechanic": "classify",
        "game_mechanic": "tap",
    }


async def test_commit_core_loop_requires_learning_link_first():
    state = GameDesignState()
    reg = build_game_design_registry(state)
    out = await reg.execute("commit_core_loop", {
        "player_action": "a", "system_response": "b",
        "feedback": "c", "win_condition": "d",
    })
    assert out.startswith("ERROR")
    assert "commit_learning_link first" in out


async def test_commit_aesthetic_validates_archetype_enum():
    state = GameDesignState()
    state.learning_link = {"x": "y"}  # bypass first-tool check
    state.core_loop = {"x": "y"}
    reg = build_game_design_registry(state)
    out = await reg.execute("commit_aesthetic", {
        "intended_feeling": "mastery",
        "ui_archetype": "not_an_archetype",
    })
    assert out.startswith("ERROR")
    assert "ui_archetype" in out


async def test_submit_game_blocks_until_design_complete():
    state = GameDesignState()
    reg = build_game_design_registry(state)
    out = await reg.execute("submit_game", {
        "title": "t", "description": "d",
        "instructions": "i", "html": "<div>x</div>",
    })
    assert out.startswith("ERROR")
    assert "design commits" in out


async def test_validate_html_tool_requires_aesthetic_first():
    state = GameDesignState()
    reg = build_game_design_registry(state)
    out = await reg.execute("validate_html", {"html": "<div>x</div>"})
    assert out.startswith("ERROR")
    assert "aesthetic" in out.lower()


# ============================================================================
# generate_game — full PTC happy path with mocked tool-use sequence
# ============================================================================

async def test_generate_game_happy_path():
    """Full 6-tool PTC flow with a clean HTML draft drives a valid GameSpec."""
    client = FakeAnthropic(game_happy_path_responses())
    spec = await generate_game(_unit(), client=client)

    assert isinstance(spec, GameSpec)
    assert spec.unit_id == "u2"
    assert spec.title == "Test Game"
    assert "draggable" in spec.html

    # Design metadata surfaced on the spec.
    assert spec.learning_link is not None
    assert spec.learning_link["learning_mechanic"] == "classify"
    assert spec.core_loop is not None
    assert spec.aesthetic == {
        "intended_feeling": "mastery",
        "ui_archetype": "drag_drop",
    }

    # Tool-call log records the agent's traversal — useful for portfolio surface.
    assert spec.tool_call_log == [
        "commit_learning_link",
        "commit_core_loop",
        "commit_aesthetic",
        "validate_html",
        "validate_js",
        "submit_game",
    ]


async def test_generate_game_recovers_from_html_validation_failure():
    """Agent submits bad HTML, tool returns ERROR, agent retries with good HTML."""
    bad_html = "<div><button>x</button></div>"  # no draggable → fails archetype
    good_html = GAME_HAPPY_HTML

    responses = [
        tool_use_response("t1", "commit_learning_link", {
            "learning_objective": "o",
            "learning_mechanic": "classify",
            "game_mechanic": "tap",
        }),
        tool_use_response("t2", "commit_core_loop", {
            "player_action": "a", "system_response": "b",
            "feedback": "c", "win_condition": "d",
        }),
        tool_use_response("t3", "commit_aesthetic", {
            "intended_feeling": "mastery",
            "ui_archetype": "drag_drop",
        }),
        tool_use_response("t4a", "validate_html", {"html": bad_html}),  # FAILS
        tool_use_response("t4b", "validate_html", {"html": good_html}),  # OK
        tool_use_response("t5", "validate_js", {"js": "let x = 1;"}),
        tool_use_response("t6", "submit_game", {
            "title": "g", "description": "d",
            "instructions": "i", "html": good_html,
        }),
        text_response("ok", stop_reason="end_turn"),
    ]
    client = FakeAnthropic(responses)
    spec = await generate_game(_unit(), client=client)
    assert spec.html == good_html
    # Two validate_html calls in the log — one failed, one succeeded.
    assert spec.tool_call_log.count("validate_html") == 2


async def test_generate_game_raises_when_loop_ends_without_submit():
    """If the LLM gives up before calling submit_game, raise GameAgentError."""
    client = FakeAnthropic([
        tool_use_response("t1", "commit_learning_link", {
            "learning_objective": "o",
            "learning_mechanic": "m",
            "game_mechanic": "g",
        }),
        text_response("I give up", stop_reason="end_turn"),
    ])
    with pytest.raises(GameAgentError, match="without calling submit_game"):
        await generate_game(_unit(), client=client)


# ============================================================================
# Inline-JS back-compat (used by older callers)
# ============================================================================

@requires_node
async def test_validate_inline_js_returns_none_for_no_scripts():
    assert await validate_inline_js("<div><p>just html</p></div>") is None


@requires_node
async def test_validate_inline_js_returns_none_for_valid_js():
    html = (
        "<div><button id='b'>x</button>"
        "<script>document.getElementById('b').addEventListener('click',()=>{});</script>"
        "</div>"
    )
    assert await validate_inline_js(html) is None


@requires_node
async def test_validate_inline_js_flags_syntax_error():
    html = "<div><script>let x = 1 +;</script></div>"
    err = await validate_inline_js(html)
    assert err is not None
    assert "script #1" in err.lower()


# ============================================================================
# Game Quality evaluator (separate component; unaffected by PTC refactor)
# ============================================================================

def _good_spec() -> GameSpec:
    return GameSpec(
        unit_id="u2",
        title="Join Matchmaker",
        description="Drag rows between two tables to see which join keeps them.",
        concept="INNER vs LEFT join row-matching semantics",
        instructions="Tap an INNER or LEFT button, then tap pairs of rows to match.",
        html=(
            "<div><style>.cell{padding:8px}</style>"
            "<button onclick=\"pick('inner')\">INNER</button>"
            "<button onclick=\"pick('left')\">LEFT</button>"
            "<div id='board'></div>"
            "<script>let mode='inner'; function pick(m){mode=m;}</script></div>"
        ),
    )


def _judge_payload(passed: bool = True, overall: float = 0.85) -> dict:
    return {
        "axis_scores": [
            {"axis": "solvable", "score": 0.9, "rationale": "reachable win state"},
            {"axis": "on_concept", "score": 0.85, "rationale": "mechanic teaches matching"},
            {"axis": "engagement", "score": 0.8, "rationale": "interactive feedback"},
        ],
        "overall_score": overall,
        "passed": passed,
        "rationale": "Solid concept-as-interaction.",
    }


async def test_game_quality_passes_for_interactive_safe_html():
    client = FakeAnthropic([text_response(json.dumps(_judge_payload(True)))])
    result = await evaluate_game_quality(_good_spec(), _unit(), judge=client)
    assert result.passed is True
    assert result.playable is True
    assert result.has_forbidden_resource is False


async def test_game_quality_fails_when_no_interactive_elements():
    spec = GameSpec(
        unit_id="u2",
        title="Static Demo",
        description="A static page",
        concept="x",
        instructions="read",
        html="<div>hello world, just text here</div>",
    )
    client = FakeAnthropic([text_response(json.dumps(_judge_payload(True)))])
    result = await evaluate_game_quality(spec, _unit(), judge=client)
    assert result.playable is False
    assert result.passed is False
    assert "no interactive elements" in result.rationale.lower()


async def test_game_quality_flags_forbidden_external_resource():
    spec = GameSpec(
        unit_id="u2",
        title="Bad Game",
        description="sneaks a CDN",
        concept="x",
        instructions="play",
        html=(
            "<div><script src='https://cdn.example.com/evil.js'></script>"
            "<button onclick='x()'>go</button></div>"
        ),
    )
    client = FakeAnthropic([text_response(json.dumps(_judge_payload(True)))])
    result = await evaluate_game_quality(spec, _unit(), judge=client)
    assert result.has_forbidden_resource is True
    assert result.passed is False


async def test_game_quality_flags_fetch_call():
    html = (
        "<div><button onclick='go()'>go</button>"
        "<script>function go(){fetch('/x')}</script></div>"
    )
    spec = GameSpec(
        unit_id="u2", title="Bad", description="d",
        concept="c", instructions="i", html=html,
    )
    client = FakeAnthropic([text_response(json.dumps(_judge_payload(True)))])
    result = await evaluate_game_quality(spec, _unit(), judge=client)
    assert result.has_forbidden_resource is True
    assert result.passed is False


async def test_game_quality_rejects_malformed_judge_output():
    client = FakeAnthropic([text_response(json.dumps({"axis_scores": []}))])
    with pytest.raises(GameQualityEvalError):
        await evaluate_game_quality(_good_spec(), _unit(), judge=client)


async def test_game_quality_evaluator_sends_objective_and_html():
    client = FakeAnthropic([text_response(json.dumps(_judge_payload()))])
    await evaluate_game_quality(_good_spec(), _unit(), judge=client)
    call = client.messages.calls[0]
    user_content = call["messages"][0]["content"]
    assert "Distinguish INNER vs LEFT joins" in user_content
    assert "Join Matchmaker" in user_content
    assert "Game HTML:" in user_content


# ============================================================================
# UI archetype enum is the constraint — verify it stays in sync
# ============================================================================

def test_ui_archetypes_enum_has_expected_options():
    assert set(UI_ARCHETYPES) == {
        "grid", "drag_drop", "button_array", "canvas", "list",
    }
