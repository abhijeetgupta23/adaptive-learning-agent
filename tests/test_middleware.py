"""Tests for backend.orchestration.middleware: retry + circuit breaker."""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from backend.orchestration.middleware import (
    CircuitBreaker,
    CircuitOpenError,
    MiddlewareClient,
    RetryPolicy,
    is_retryable,
)
from tests._fakes import text_response


class _Status5xx(Exception):
    """Mimics anthropic SDK errors that carry a status_code."""
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _ScriptedMessages:
    """Replays a list of side effects: response objects to return, or exceptions to raise."""

    def __init__(self, sequence: list[Any]) -> None:
        self._seq = list(sequence)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._seq.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _ScriptedClient:
    def __init__(self, sequence: list[Any]) -> None:
        self.messages = _ScriptedMessages(sequence)


# ---------- is_retryable ----------

def test_is_retryable_for_httpx_request_error():
    assert is_retryable(httpx.RequestError("timeout"))


def test_is_retryable_for_5xx_status():
    assert is_retryable(_Status5xx(503))


def test_is_retryable_for_429():
    assert is_retryable(_Status5xx(429))


def test_not_retryable_for_4xx():
    assert not is_retryable(_Status5xx(400))


def test_not_retryable_for_value_error():
    assert not is_retryable(ValueError("bad"))


# ---------- retry behaviour ----------

@pytest.mark.asyncio
async def test_passes_through_when_inner_succeeds():
    inner = _ScriptedClient([text_response("ok")])
    client = MiddlewareClient(inner, retry=RetryPolicy(base_delay_s=0.0, max_delay_s=0.0))
    response = await client.messages.create(model="x", messages=[], max_tokens=10)
    assert response.content[0].text == "ok"
    assert len(inner.messages.calls) == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds_on_transient_error():
    inner = _ScriptedClient([
        httpx.RequestError("net hiccup"),
        text_response("recovered"),
    ])
    client = MiddlewareClient(
        inner,
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
    )
    response = await client.messages.create(model="x", messages=[], max_tokens=10)
    assert response.content[0].text == "recovered"
    assert len(inner.messages.calls) == 2


@pytest.mark.asyncio
async def test_retries_exhausted_raises_last_error():
    inner = _ScriptedClient([
        httpx.RequestError("hiccup 1"),
        httpx.RequestError("hiccup 2"),
        httpx.RequestError("hiccup 3"),
    ])
    client = MiddlewareClient(
        inner,
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
    )
    with pytest.raises(httpx.RequestError):
        await client.messages.create(model="x", messages=[], max_tokens=10)
    assert len(inner.messages.calls) == 3


@pytest.mark.asyncio
async def test_non_retryable_error_does_not_retry():
    inner = _ScriptedClient([ValueError("schema bug")])
    client = MiddlewareClient(
        inner,
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
    )
    with pytest.raises(ValueError):
        await client.messages.create(model="x", messages=[], max_tokens=10)
    assert len(inner.messages.calls) == 1


# ---------- circuit breaker ----------

@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_consecutive_failures():
    inner = _ScriptedClient([
        ValueError("a"),  # non-retryable, single attempt → 1 failure
        ValueError("b"),  # 2 failures
        ValueError("c"),  # 3 failures → trips breaker
    ])
    breaker = CircuitBreaker(threshold=3, cooldown_s=10.0)
    client = MiddlewareClient(
        inner,
        retry=RetryPolicy(max_attempts=1),
        breaker=breaker,
    )
    for _ in range(3):
        with pytest.raises(ValueError):
            await client.messages.create(model="x", messages=[], max_tokens=10)
    # Now the breaker should reject without calling inner.
    with pytest.raises(CircuitOpenError):
        await client.messages.create(model="x", messages=[], max_tokens=10)
    # inner was called exactly 3 times — the 4th call short-circuited.
    assert len(inner.messages.calls) == 3


@pytest.mark.asyncio
async def test_circuit_resets_on_success():
    inner = _ScriptedClient([
        ValueError("a"),
        ValueError("b"),
        text_response("ok"),
        text_response("ok2"),
    ])
    breaker = CircuitBreaker(threshold=3, cooldown_s=10.0)
    client = MiddlewareClient(
        inner,
        retry=RetryPolicy(max_attempts=1),
        breaker=breaker,
    )
    for _ in range(2):
        with pytest.raises(ValueError):
            await client.messages.create(model="x", messages=[], max_tokens=10)
    assert breaker.consecutive_failures == 2
    # Successful call resets counter.
    response = await client.messages.create(model="x", messages=[], max_tokens=10)
    assert response.content[0].text == "ok"
    assert breaker.consecutive_failures == 0
    # Another success works fine.
    response = await client.messages.create(model="x", messages=[], max_tokens=10)
    assert response.content[0].text == "ok2"


@pytest.mark.asyncio
async def test_open_circuit_half_opens_after_cooldown(monkeypatch: pytest.MonkeyPatch):
    inner = _ScriptedClient([
        ValueError("a"),
        ValueError("b"),
        text_response("recovered"),
    ])
    breaker = CircuitBreaker(threshold=2, cooldown_s=0.5)
    client = MiddlewareClient(
        inner,
        retry=RetryPolicy(max_attempts=1),
        breaker=breaker,
    )
    for _ in range(2):
        with pytest.raises(ValueError):
            await client.messages.create(model="x", messages=[], max_tokens=10)
    # Breaker should be open now.
    assert not breaker.can_attempt()
    # Fast-forward time past cooldown.
    import backend.orchestration.middleware as mw
    real_monotonic = mw.time.monotonic
    monkeypatch.setattr(mw.time, "monotonic", lambda: real_monotonic() + 1.0)
    # Should accept the probe and recover.
    response = await client.messages.create(model="x", messages=[], max_tokens=10)
    assert response.content[0].text == "recovered"
    assert breaker.opened_at is None
