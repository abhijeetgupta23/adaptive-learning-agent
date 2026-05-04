from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from backend.orchestration.cache import normalize_system
from backend.orchestration.cost import record_usage
from backend.orchestration.tools import ToolRegistry

_logger = logging.getLogger(__name__)


class _MessagesAPI(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class AnthropicLike(Protocol):
    messages: _MessagesAPI


class PTCError(Exception):
    """Raised when the PTC loop cannot complete."""


@dataclass
class PTCResult:
    final_message: Any
    history: list[dict[str, Any]]
    iterations: int
    input_tokens_total: int = 0
    output_tokens_total: int = 0


def _serialize_block(block: Any) -> dict[str, Any]:
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id"),
            "name": getattr(block, "name"),
            "input": getattr(block, "input", {}),
        }
    raise PTCError(f"Unknown content block type: {btype!r}")


def _with_rolling_cache_breakpoint(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Place a `cache_control: ephemeral` marker on the last block of the
    most recent user message.

    On each PTC iteration this writes a new cache entry at the growing tail
    of the conversation. The next iteration's request reads it via Anthropic's
    20-block lookback window. Combined with the system-prompt breakpoint
    (auto-wrapped by `normalize_system`), this gives us up to 2 of the 4
    available breakpoints per request — enough for SP + rolling tail.

    We skip the marker when the message's content is empty or the last block
    is already a cached one (idempotent).
    """
    if not history:
        return history
    out = [dict(m) for m in history]
    for i in range(len(out) - 1, -1, -1):
        msg = out[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            new_blocks = [dict(b) for b in content]
            last = new_blocks[-1]
            if "cache_control" not in last:
                last["cache_control"] = {"type": "ephemeral"}
            new_blocks[-1] = last
            msg["content"] = new_blocks
        break
    return out


async def run_ptc(
    client: AnthropicLike,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
    tools: ToolRegistry | None = None,
    max_iterations: int = 10,
    max_tokens: int = 4096,
    agent: str = "ptc",
) -> PTCResult:
    history = [dict(m) for m in messages]
    in_total = 0
    out_total = 0
    cached_system_value = normalize_system(system)  # auto-wrap str → cache-marked list

    for iteration in range(1, max_iterations + 1):
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": _with_rolling_cache_breakpoint(history),
            "max_tokens": max_tokens,
        }
        if cached_system_value is not None:
            call_kwargs["system"] = cached_system_value
        if tools and tools.tools:
            call_kwargs["tools"] = tools.as_anthropic_specs()

        start = time.monotonic()
        response = await client.messages.create(**call_kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        usage = getattr(response, "usage", None)
        if usage is not None:
            in_total += getattr(usage, "input_tokens", 0) or 0
            out_total += getattr(usage, "output_tokens", 0) or 0
        record_usage(model=model, agent=agent, response=response)

        _logger.info(
            "ptc_iteration",
            extra={
                "iteration": iteration,
                "latency_ms": round(latency_ms, 1),
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )

        content_blocks = list(getattr(response, "content", []))
        serialized = [_serialize_block(b) for b in content_blocks]
        history.append({"role": "assistant", "content": serialized})

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason != "tool_use":
            return PTCResult(
                final_message=response,
                history=history,
                iterations=iteration,
                input_tokens_total=in_total,
                output_tokens_total=out_total,
            )

        if tools is None:
            raise PTCError("Model requested a tool but no registry was provided")

        tool_results: list[dict[str, Any]] = []
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            result = await tools.execute(
                getattr(block, "name"), getattr(block, "input", {}) or {}
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "id"),
                    "content": result,
                }
            )
        history.append({"role": "user", "content": tool_results})

    raise PTCError(f"Exceeded max_iterations={max_iterations}")
