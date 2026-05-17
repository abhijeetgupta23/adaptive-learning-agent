"""Per-session Game-agent circuit breaker.

Distinct from the LLM-call-level `CircuitBreaker` in `middleware.py`: that one
guards Anthropic API health; this one guards *pedagogical* + *budget* health
for the flagship Game agent within a single learning session.

When tripped, the session dispatcher falls back to `worked_example` for the
remainder of the session (Vision B: worked_example is the explicit fallback).

Trip conditions (whichever comes first):
  (a) N consecutive game attempts that did not produce a passing artifact
  (b) cumulative Game-agent spend exceeds $X for the session

Both thresholds are tunable; defaults (3 / $0.50) are calibrated to Haiku 4.5
where a typical game costs ~$0.02-$0.08 — so $0.50 leaves room for ~6-25
attempts before the budget guard fires.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionCircuitBreaker:
    """Tracks Game-agent failures + cumulative spend for one session."""

    max_consecutive_failures: int = 3
    max_spend_usd: float = 0.50
    consecutive_failures: int = 0
    cumulative_spend_usd: float = 0.0
    tripped_reason: str | None = None

    def should_trip(self) -> tuple[bool, str | None]:
        """Return `(tripped, reason)`. Sticky once tripped."""
        if self.tripped_reason is not None:
            return True, self.tripped_reason
        if self.consecutive_failures >= self.max_consecutive_failures:
            reason = (
                f"{self.consecutive_failures} consecutive game failures "
                f"(threshold {self.max_consecutive_failures})"
            )
            self.tripped_reason = reason
            return True, reason
        if self.cumulative_spend_usd >= self.max_spend_usd:
            reason = (
                f"Game-agent spend ${self.cumulative_spend_usd:.4f} "
                f"exceeded budget ${self.max_spend_usd:.2f}"
            )
            self.tripped_reason = reason
            return True, reason
        return False, None

    def record_game_attempt(self, *, passed: bool, cost_usd: float) -> None:
        """Record one Game-agent attempt's outcome and incremental cost."""
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {cost_usd}")
        self.cumulative_spend_usd += cost_usd
        if passed:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    def reset(self) -> None:
        """Clear all state (test-friendly; not used in the session lifecycle)."""
        self.consecutive_failures = 0
        self.cumulative_spend_usd = 0.0
        self.tripped_reason = None
