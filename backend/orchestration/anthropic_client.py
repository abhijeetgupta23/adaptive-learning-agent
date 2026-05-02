from __future__ import annotations

import logging
import os
import time
from typing import Any

from anthropic import AsyncAnthropic

_logger = logging.getLogger(__name__)


def get_anthropic_client(
    api_key: str | None = None,
    max_retries: int = 2,
) -> AsyncAnthropic:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return AsyncAnthropic(api_key=key, max_retries=max_retries)


async def log_messages_call(
    client: AsyncAnthropic,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> Any:
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system is not None:
        call_kwargs["system"] = system
    if tools:
        call_kwargs["tools"] = tools
    call_kwargs.update(kwargs)

    start = time.monotonic()
    response = await client.messages.create(**call_kwargs)
    latency_ms = (time.monotonic() - start) * 1000

    usage = getattr(response, "usage", None)
    _logger.info(
        "anthropic_messages_call",
        extra={
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "tools_count": len(tools or []),
            "messages_count": len(messages),
            "stop_reason": getattr(response, "stop_reason", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        },
    )
    return response
