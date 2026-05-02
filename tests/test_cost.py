"""Unit tests for backend.orchestration.cost."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.orchestration.cost import (
    PRICING,
    CostAccumulator,
    current_accumulator,
    record_usage,
    session_accumulator,
)


def usage(
    *, input_tokens: int = 0, output_tokens: int = 0,
    cache_read: int = 0, cache_write: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def test_record_with_known_model_computes_usd():
    acc = CostAccumulator()
    # Opus: input $15/M, output $75/M
    entry = acc.record(
        model="claude-opus-4-7",
        agent="curriculum",
        usage=usage(input_tokens=10_000, output_tokens=2_000),
    )
    expected = (10_000 * 15.0 + 2_000 * 75.0) / 1_000_000
    assert entry.usd == pytest.approx(expected)
    assert acc.total_usd == pytest.approx(expected)


def test_cache_tokens_priced_at_read_and_write_rates():
    acc = CostAccumulator()
    prices = PRICING["claude-sonnet-4-6"]
    acc.record(
        model="claude-sonnet-4-6",
        agent="game",
        usage=usage(
            input_tokens=1_000, output_tokens=500,
            cache_read=5_000, cache_write=2_000,
        ),
    )
    expected = (
        1_000 * prices["input"]
        + 500 * prices["output"]
        + 5_000 * prices["cache_read"]
        + 2_000 * prices["cache_write"]
    ) / 1_000_000
    assert acc.total_usd == pytest.approx(expected)


def test_unknown_model_records_zero_dollars_but_keeps_tokens():
    acc = CostAccumulator()
    entry = acc.record(
        model="some-future-model",
        agent="x",
        usage=usage(input_tokens=999, output_tokens=999),
    )
    assert entry.usd == 0.0
    assert entry.input_tokens == 999


def test_by_agent_aggregates_across_calls():
    acc = CostAccumulator()
    acc.record(model="claude-sonnet-4-6", agent="intent",
               usage=usage(input_tokens=100, output_tokens=20))
    acc.record(model="claude-sonnet-4-6", agent="intent",
               usage=usage(input_tokens=200, output_tokens=40))
    acc.record(model="claude-opus-4-7", agent="curriculum",
               usage=usage(input_tokens=10, output_tokens=5))
    by_agent = acc.by_agent()
    assert set(by_agent) == {"intent", "curriculum"}
    assert by_agent["intent"] > 0
    assert by_agent["curriculum"] > 0
    assert sum(by_agent.values()) == pytest.approx(acc.total_usd)


def test_cache_hit_rate_is_zero_when_no_input():
    acc = CostAccumulator()
    assert acc.cache_hit_rate() == 0.0


def test_cache_hit_rate_reflects_read_share():
    acc = CostAccumulator()
    acc.record(model="claude-sonnet-4-6", agent="x",
               usage=usage(input_tokens=10, cache_read=90))
    # 90 / (10 + 90 + 0) = 0.9
    assert acc.cache_hit_rate() == pytest.approx(0.9)


def test_record_usage_no_op_outside_session():
    # No accumulator on the contextvar → record_usage is a no-op.
    response = SimpleNamespace(usage=usage(input_tokens=100))
    assert current_accumulator() is None
    assert record_usage(model="claude-opus-4-7", agent="x", response=response) is None


def test_session_accumulator_scopes_recording():
    response = SimpleNamespace(usage=usage(input_tokens=1_000, output_tokens=200))
    with session_accumulator() as acc:
        record_usage(model="claude-opus-4-7", agent="game", response=response)
        assert len(acc.entries) == 1
        assert acc.total_usd > 0
    # Outside the context, no accumulator anymore.
    assert current_accumulator() is None


def test_record_usage_skips_response_without_usage():
    response = SimpleNamespace()  # no .usage
    with session_accumulator() as acc:
        assert record_usage(model="claude-opus-4-7", agent="x", response=response) is None
        assert acc.entries == []
