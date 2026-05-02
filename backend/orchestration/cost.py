"""Per-session LLM cost tracking.

Usage:
    with session_accumulator() as acc:
        ...                         # any LLM calls inside record into `acc`
        record_usage(model=..., agent=..., response=resp)

Each `record_usage` call pulls token counts (incl. cache_read /
cache_creation) from the response and converts to USD using PRICING below.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# USD per 1M tokens. Cache pricing: read = 10% of base input, write (5-min
# TTL) = 125% of base input. Source: anthropic.com/pricing.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
}


@dataclass
class CallEntry:
    model: str
    agent: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    usd: float


@dataclass
class CostAccumulator:
    entries: list[CallEntry] = field(default_factory=list)

    def record(self, *, model: str, agent: str, usage: Any) -> CallEntry:
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

        prices = PRICING.get(model)
        if prices is None:
            usd = 0.0
        else:
            usd = (
                in_tok * prices["input"]
                + out_tok * prices["output"]
                + cache_read * prices["cache_read"]
                + cache_write * prices["cache_write"]
            ) / 1_000_000

        entry = CallEntry(
            model=model,
            agent=agent,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            usd=usd,
        )
        self.entries.append(entry)
        return entry

    @property
    def total_usd(self) -> float:
        return sum(e.usd for e in self.entries)

    def by_agent(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries:
            out[e.agent] = out.get(e.agent, 0.0) + e.usd
        return out

    def cache_hit_rate(self) -> float:
        # Fraction of input-side tokens that came from a cache read.
        billed_in = sum(
            e.input_tokens + e.cache_read_tokens + e.cache_write_tokens
            for e in self.entries
        )
        if billed_in == 0:
            return 0.0
        return sum(e.cache_read_tokens for e in self.entries) / billed_in


_acc_var: ContextVar[CostAccumulator | None] = ContextVar(
    "cost_accumulator", default=None
)


def current_accumulator() -> CostAccumulator | None:
    return _acc_var.get()


@contextmanager
def session_accumulator():
    acc = CostAccumulator()
    token = _acc_var.set(acc)
    try:
        yield acc
    finally:
        _acc_var.reset(token)


def record_usage(*, model: str, agent: str, response: Any) -> CallEntry | None:
    """Record one LLM call into the active accumulator. No-op outside a session."""
    acc = current_accumulator()
    if acc is None:
        return None
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return acc.record(model=model, agent=agent, usage=usage)
