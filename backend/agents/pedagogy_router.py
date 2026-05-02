from __future__ import annotations

import logging
from typing import Any

from backend.agents._parsing import (
    StructuredOutputError,
    extract_text,
    parse_json_payload,
)
from backend.orchestration.cache import cached_system
from backend.orchestration.cost import record_usage
from backend.orchestration.ptc import AnthropicLike
from backend.schemas.curriculum import LearningUnit
from backend.schemas.state import LearnerState
from backend.schemas.teaching import MethodCall, TeachingMethod, TeachingPlan
from backend.skills import SkillRegistry

_logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "claude-sonnet-4-6"

_ROUTER_LLM_SYSTEM_PROMPT_TEMPLATE = """\
You are the Pedagogy Router. Pick the best teaching method for this unit, given \
the learner's state.

Available methods:

{PEDAGOGY_CATALOG}

Return JSON exactly:

{"method": "<skill name from the catalog above>", "rationale": "<one sentence>"}
"""

# Legacy fallback used when no skill registry is supplied (e.g. older tests).
_LEGACY_ROUTER_PROMPT = """\
You are the Pedagogy Router. Pick the best teaching method for this unit, given \
the learner's state.

Guidance:
- worked_example: step-by-step procedures, algorithms, SQL queries
- analogy: novel abstractions mapped to familiar ones
- visual: relational, topological, or spatial concepts
- game: concepts that are games-at-heart (state, rules, transitions, strategy)
- socratic: fuzzy conceptual understanding the learner must construct
- feynman: the learner should be able to teach it back
- retrieval: vocabulary, facts, rote memorization

Return JSON exactly:

{"method": "<one of the methods above>", "rationale": "<one sentence>"}
"""


def build_router_system_prompt(registry: SkillRegistry | None) -> str:
    if registry is None:
        return _LEGACY_ROUTER_PROMPT
    return _ROUTER_LLM_SYSTEM_PROMPT_TEMPLATE.replace(
        "{PEDAGOGY_CATALOG}", registry.catalog_text(),
    )


class RouterError(Exception):
    """Raised when the router cannot pick a method."""


async def route_pedagogy(
    unit: LearningUnit,
    state: LearnerState,
    *,
    excluded_methods: list[TeachingMethod] | None = None,
    llm_client: AnthropicLike | None = None,
    model: str = DEFAULT_LLM_MODEL,
    teaching_registry: SkillRegistry | None = None,
) -> TeachingPlan:
    """Rules-first router with optional LLM fallback for ambiguous cases."""
    excluded = set(excluded_methods or [])

    # Rule 1: learner has a proven-effective pedagogy for this concept.
    pref = state.effective_pedagogies.get(unit.objective)
    if pref is not None and pref not in excluded:
        return _plan(
            unit, pref,
            f"Learner previously succeeded with {pref.value} on this concept.",
        )

    # Rule 2: curriculum's per-unit recommendation.
    if unit.recommended_pedagogy not in excluded:
        return _plan(
            unit, unit.recommended_pedagogy,
            "Using curriculum-recommended pedagogy.",
        )

    # Rule 3: fall back to any remaining method.
    remaining = [m for m in TeachingMethod if m not in excluded]
    if not remaining:
        raise RouterError(f"All methods excluded for unit {unit.id}")

    if llm_client is None:
        return _plan(
            unit, remaining[0],
            "Deterministic fallback: curriculum recommendation excluded; no LLM available.",
        )

    chosen, rationale = await _llm_choose(
        unit, state, remaining, llm_client, model, teaching_registry,
    )
    return _plan(unit, chosen, rationale)


def _plan(unit: LearningUnit, method: TeachingMethod, rationale: str) -> TeachingPlan:
    plan = TeachingPlan(
        unit_id=unit.id,
        methods=[MethodCall(method=method)],
        rationale=rationale,
    )
    _logger.info(
        "pedagogy_router",
        extra={"unit_id": unit.id, "method": method.value, "rationale": rationale},
    )
    return plan


async def _llm_choose(
    unit: LearningUnit,
    state: LearnerState,
    remaining: list[TeachingMethod],
    client: AnthropicLike,
    model: str,
    teaching_registry: SkillRegistry | None,
) -> tuple[TeachingMethod, str]:
    user_message = (
        f"Unit to teach:\n"
        f"- objective: {unit.objective}\n"
        f"- original recommendation: {unit.recommended_pedagogy.value} "
        f"(excluded — prior attempt failed)\n\n"
        f"Allowed methods: {[m.value for m in remaining]}\n\n"
        f"Learner state:\n"
        f"- mastered concepts: {state.mastered_concepts[:10]}\n"
        f"- previously struggled: {state.struggled_concepts[:10]}\n"
    )
    response = await client.messages.create(
        model=model,
        system=cached_system(build_router_system_prompt(teaching_registry)),
        messages=[{"role": "user", "content": user_message}],
        max_tokens=512,
    )
    record_usage(model=model, agent="pedagogy_router", response=response)
    try:
        text = extract_text(response)
        payload: dict[str, Any] = parse_json_payload(text)
    except StructuredOutputError as exc:
        raise RouterError(f"Router LLM output unparseable: {exc}") from exc
    method_value = payload.get("method")
    rationale = payload.get("rationale", "LLM-selected method.")
    try:
        chosen = TeachingMethod(method_value)
    except ValueError as exc:
        raise RouterError(f"Router LLM picked invalid method: {method_value!r}") from exc
    if chosen not in remaining:
        raise RouterError(
            f"Router LLM picked excluded method {chosen.value}; allowed: "
            f"{[m.value for m in remaining]}"
        )
    return chosen, rationale
