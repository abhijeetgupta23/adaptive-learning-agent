"""LLM-call middleware: retry-with-backoff + circuit breaker.

Wraps an `AnthropicLike` client. Consumers see the same `client.messages.create(**)`
API but get automatic retry on transient errors and fail-fast when the upstream is
consistently unhealthy.

Wire once at app startup (see backend/main.py lifespan); call sites stay unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

_logger = logging.getLogger(__name__)


class _MessagesAPI(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class AnthropicLike(Protocol):
    messages: _MessagesAPI


_RETRYABLE_HTTPX: tuple[type[BaseException], ...] = (
    httpx.RequestError,
    httpx.ReadTimeout,
    httpx.ConnectError,
)


def is_retryable(exc: BaseException) -> bool:
    """Should this exception be retried? Network-level + 429/5xx HTTP."""
    if isinstance(exc, _RETRYABLE_HTTPX):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        # anthropic SDK exceptions sometimes nest the response
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    if status is not None and (status == 429 or 500 <= status < 600):
        return True
    return False


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open and rejecting requests."""


@dataclass
class CircuitBreaker:
    threshold: int = 3
    cooldown_s: float = 30.0
    consecutive_failures: int = 0
    opened_at: float | None = None

    def can_attempt(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            return True  # half-open: probe
        return False

    def on_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None

    def on_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self.opened_at = time.monotonic()


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.25

    def delay_for(self, attempt: int) -> float:
        d = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return d + random.uniform(-self.jitter * d, self.jitter * d)


class _MiddlewareMessages:
    def __init__(
        self,
        inner: _MessagesAPI,
        *,
        retry: RetryPolicy,
        breaker: CircuitBreaker,
    ) -> None:
        self._inner = inner
        self._retry = retry
        self._breaker = breaker

    async def create(self, **kwargs: Any) -> Any:
        if not self._breaker.can_attempt():
            _logger.warning(
                "circuit_open_reject",
                extra={"model": kwargs.get("model")},
            )
            raise CircuitOpenError("Circuit breaker is open; refusing call.")

        last_exc: BaseException | None = None
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                response = await self._inner.create(**kwargs)
                self._breaker.on_success()
                if attempt > 1:
                    _logger.info(
                        "llm_call_recovered",
                        extra={
                            "attempt": attempt,
                            "model": kwargs.get("model"),
                        },
                    )
                return response
            except BaseException as exc:
                last_exc = exc
                retryable = is_retryable(exc)
                _logger.info(
                    "llm_call_attempt_failed",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self._retry.max_attempts,
                        "model": kwargs.get("model"),
                        "exc_type": type(exc).__name__,
                        "retryable": retryable,
                    },
                )
                if not retryable or attempt >= self._retry.max_attempts:
                    self._breaker.on_failure()
                    raise
                await asyncio.sleep(self._retry.delay_for(attempt))

        assert last_exc is not None  # unreachable
        raise last_exc


class MiddlewareClient:
    """Drop-in `AnthropicLike` wrapper that adds retry + circuit breaker."""

    def __init__(
        self,
        inner: AnthropicLike,
        *,
        retry: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._inner = inner
        self.retry = retry or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        self.messages = _MiddlewareMessages(
            inner.messages, retry=self.retry, breaker=self.breaker,
        )
