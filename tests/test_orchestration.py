import json
import logging
from types import SimpleNamespace

import pytest

from backend.orchestration import (
    JSONFormatter,
    PTCError,
    Tool,
    ToolRegistry,
    encode_sse_event,
    encode_sse_stream,
    get_request_id,
    new_request_id,
    run_ptc,
    set_request_id,
)
from backend.schemas import (
    AssessmentResult,
    AssessmentResultEvent,
    Curriculum,
    CurriculumReadyEvent,
    DesiredDepth,
    GroundingSource,
    KnowledgeLevel,
    LearningGoal,
    LearningUnit,
    SSEEventAdapter,
    TeachingChunkEvent,
    TeachingMethod,
    Verdict,
)

# -------- Logging --------

def _make_record(msg: str = "hi") -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname="f.py", lineno=1,
        msg=msg, args=None, exc_info=None,
    )


class TestJSONFormatter:
    def setup_method(self) -> None:
        set_request_id(None)

    def test_formats_as_valid_json(self) -> None:
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert payload["msg"] == "hi"
        assert payload["level"] == "INFO"
        assert "ts" in payload

    def test_includes_request_id_from_context(self) -> None:
        set_request_id("req-abc")
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert payload["request_id"] == "req-abc"

    def test_null_request_id_when_unset(self) -> None:
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert payload["request_id"] is None

    def test_extra_fields_included(self) -> None:
        rec = _make_record()
        rec.model = "sonnet"  # type: ignore[attr-defined]
        rec.latency_ms = 123.4  # type: ignore[attr-defined]
        payload = json.loads(JSONFormatter().format(rec))
        assert payload["model"] == "sonnet"
        assert payload["latency_ms"] == 123.4


def test_new_request_id_is_uuid_and_contextual() -> None:
    rid = new_request_id()
    assert get_request_id() == rid
    assert len(rid) == 36


# -------- SSE --------

def _goal() -> LearningGoal:
    return LearningGoal(
        domain="d", sub_goal="s",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=1.0,
    )


def _curriculum() -> Curriculum:
    return Curriculum(
        goal=_goal(),
        units=[LearningUnit(
            id="u1", objective="o",
            recommended_pedagogy=TeachingMethod.GAME,
            grounding_sources=[GroundingSource(url="https://x.com", title="t")],
        )],
    )


class TestSSEEncoding:
    def test_frame_format(self) -> None:
        frame = encode_sse_event(
            TeachingChunkEvent(unit_id="u1", method=TeachingMethod.ANALOGY, content="hi")
        )
        assert frame.startswith("data: ")
        assert frame.endswith("\n\n")
        payload = json.loads(frame[len("data: "):].strip())
        assert payload["event"] == "teaching_chunk"
        assert payload["content"] == "hi"

    def test_nested_curriculum_roundtrips_through_adapter(self) -> None:
        event = CurriculumReadyEvent(curriculum=_curriculum())
        body = json.loads(encode_sse_event(event)[len("data: "):].strip())
        parsed = SSEEventAdapter.validate_python(body)
        assert isinstance(parsed, CurriculumReadyEvent)
        assert parsed.curriculum.units[0].id == "u1"


async def test_encode_sse_stream_yields_frames() -> None:
    async def events():
        yield TeachingChunkEvent(unit_id="u1", method=TeachingMethod.GAME, content="a")
        yield AssessmentResultEvent(result=AssessmentResult(
            unit_id="u1", verdict=Verdict.PASS, diagnostic_notes="", confidence=1.0,
        ))

    frames = [f async for f in encode_sse_stream(events())]
    assert len(frames) == 2
    assert '"teaching_chunk"' in frames[0]
    assert '"assessment_result"' in frames[1]


# -------- Tools --------

class TestToolRegistry:
    def test_register_and_get(self) -> None:
        async def handler(inp: dict) -> str: return f"got {inp}"
        reg = ToolRegistry()
        reg.register(Tool(
            name="echo", description="",
            input_schema={"type": "object"}, handler=handler,
        ))
        assert reg.get("echo").name == "echo"

    def test_duplicate_registration_raises(self) -> None:
        async def handler(inp: dict) -> str: return "x"
        reg = ToolRegistry()
        tool = Tool(name="t", description="", input_schema={}, handler=handler)
        reg.register(tool)
        with pytest.raises(ValueError):
            reg.register(tool)

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(KeyError):
            ToolRegistry().get("nope")

    def test_as_anthropic_specs(self) -> None:
        async def handler(inp: dict) -> str: return "x"
        reg = ToolRegistry()
        reg.register(Tool(
            name="web_search", description="search the web",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            handler=handler,
        ))
        assert reg.as_anthropic_specs() == [{
            "name": "web_search",
            "description": "search the web",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]

    async def test_execute_returns_string_unchanged(self) -> None:
        async def handler(inp: dict) -> str: return "raw"
        reg = ToolRegistry()
        reg.register(Tool(name="t", description="", input_schema={}, handler=handler))
        assert await reg.execute("t", {}) == "raw"

    async def test_execute_json_encodes_non_strings(self) -> None:
        async def handler(inp: dict) -> dict: return {"count": 3}
        reg = ToolRegistry()
        reg.register(Tool(name="t", description="", input_schema={}, handler=handler))
        assert json.loads(await reg.execute("t", {})) == {"count": 3}


# -------- PTC loop --------

class _FakeMessages:
    def __init__(self, responses: list) -> None:
        self._responses = iter(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return next(self._responses)


class FakeAnthropic:
    def __init__(self, responses: list) -> None:
        self.messages = _FakeMessages(responses)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(id_: str, name: str, inp: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def _response(
    content: list, stop_reason: str, in_tok: int = 10, out_tok: int = 5
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


class TestPTCLoop:
    async def test_immediate_end_turn_returns(self) -> None:
        client = FakeAnthropic([_response([_text("hello")], "end_turn")])
        result = await run_ptc(
            client, model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result.iterations == 1
        assert result.input_tokens_total == 10
        assert result.output_tokens_total == 5
        assert result.history[-1] == {
            "role": "assistant", "content": [{"type": "text", "text": "hello"}],
        }

    async def test_tool_use_then_end_turn(self) -> None:
        async def search(inp: dict) -> list[dict]:
            return [{"title": f"result for {inp['q']}"}]

        reg = ToolRegistry()
        reg.register(Tool(
            name="search", description="",
            input_schema={"type": "object"}, handler=search,
        ))

        client = FakeAnthropic([
            _response([_tool_use("id1", "search", {"q": "joins"})], "tool_use"),
            _response([_text("final answer")], "end_turn"),
        ])

        result = await run_ptc(
            client, model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "explain joins"}],
            tools=reg,
        )

        assert result.iterations == 2
        assert len(result.history) == 4
        tool_result_msg = result.history[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "id1"

    async def test_tool_use_without_registry_raises(self) -> None:
        client = FakeAnthropic([_response([_tool_use("id1", "search", {})], "tool_use")])
        with pytest.raises(PTCError):
            await run_ptc(client, model="m", messages=[{"role": "user", "content": "x"}])

    async def test_max_iterations_raises(self) -> None:
        async def h(inp: dict) -> str: return "x"
        reg = ToolRegistry()
        reg.register(Tool(name="t", description="", input_schema={}, handler=h))
        client = FakeAnthropic([
            _response([_tool_use("1", "t", {})], "tool_use"),
            _response([_tool_use("2", "t", {})], "tool_use"),
        ])
        with pytest.raises(PTCError, match="max_iterations"):
            await run_ptc(
                client, model="m",
                messages=[{"role": "user", "content": "x"}],
                tools=reg, max_iterations=2,
            )

    async def test_system_and_tools_passed_through(self) -> None:
        async def h(inp: dict) -> str: return "x"
        reg = ToolRegistry()
        reg.register(Tool(name="t", description="d", input_schema={"type": "object"}, handler=h))
        client = FakeAnthropic([_response([_text("ok")], "end_turn")])
        await run_ptc(
            client, model="m",
            messages=[{"role": "user", "content": "x"}],
            system="you are helpful", tools=reg,
        )
        call = client.messages.calls[0]
        # PTC auto-wraps str system with cache_control for prompt caching.
        assert call["system"][0]["text"] == "you are helpful"
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert call["tools"] == [
            {"name": "t", "description": "d", "input_schema": {"type": "object"}}
        ]

    async def test_usage_accumulates_across_iterations(self) -> None:
        async def h(inp: dict) -> str: return "x"
        reg = ToolRegistry()
        reg.register(Tool(name="t", description="", input_schema={}, handler=h))
        client = FakeAnthropic([
            _response([_tool_use("1", "t", {})], "tool_use", in_tok=10, out_tok=5),
            _response([_text("done")], "end_turn", in_tok=20, out_tok=7),
        ])
        result = await run_ptc(
            client, model="m",
            messages=[{"role": "user", "content": "x"}],
            tools=reg,
        )
        assert result.input_tokens_total == 30
        assert result.output_tokens_total == 12
