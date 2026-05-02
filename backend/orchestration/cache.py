"""Prompt caching helpers.

Anthropic prompt caching uses `cache_control` markers on system prompt blocks
(and on the last tool spec, optionally). Anything up to and including the
marker is cached for 5 minutes (ephemeral) — subsequent calls within that
window with the same cached prefix get input-token rates ~10x cheaper.

For our agents, the biggest win is caching the system prompt:
- Curriculum Agent's PTC loop replays the same ~600-token system prompt every
  iteration. Caching that alone cuts ~70% of input cost.
- Single-call agents (Intent, Game, Assessment) benefit on repeated calls
  within the cache window (e.g. multiple intent turns in a conversation).
"""
from __future__ import annotations

from typing import Any


def cached_system(prompt: str) -> list[dict[str, Any]]:
    """Wrap a system prompt string with an ephemeral cache_control marker."""
    return [{
        "type": "text",
        "text": prompt,
        "cache_control": {"type": "ephemeral"},
    }]


def normalize_system(system: str | list[dict[str, Any]] | None) -> Any:
    """Auto-wrap a string system prompt; pass list-form through unchanged.

    Useful in code paths that accept both raw strings (convenience) and
    pre-built cache-control lists (advanced callers wanting to mark
    non-default cache points).
    """
    if system is None:
        return None
    if isinstance(system, str):
        return cached_system(system)
    return system
