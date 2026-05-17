"""Teaching-skill registry.

Vision B (game-first): only Game and Worked Example are registered. Game is
the flagship; Worked Example is the silent fallback for concepts the game
agent cannot produce a passing artifact for. Other `TeachingMethod` enum
values exist for back-compat with serialized state and golden cases, but
are deliberately not routable.
"""
from __future__ import annotations

from backend.agents.teaching.game import GAME_SYSTEM_PROMPT, generate_game
from backend.agents.teaching.worked_example import (
    WORKED_EXAMPLE_SYSTEM_PROMPT,
    generate_worked_example,
)
from backend.orchestration.middleware import AnthropicLike
from backend.schemas.curriculum import LearningUnit
from backend.schemas.teaching import TeachingMethod
from backend.skills import Skill, SkillRegistry


def build_teaching_registry(
    *, client: AnthropicLike,
) -> SkillRegistry[LearningUnit, str]:
    """Construct a registry where keys match `TeachingMethod` enum values."""
    reg: SkillRegistry[LearningUnit, str] = SkillRegistry()

    # ---- real implementations ----

    async def _gen_game(unit: LearningUnit) -> str:
        spec = await generate_game(unit, client=client)
        return spec.html

    async def _gen_worked_example(unit: LearningUnit) -> str:
        return await generate_worked_example(unit, client=client)

    reg.register(Skill(
        name=TeachingMethod.GAME.value,
        description="Generate a self-contained interactive HTML game; learner plays to learn",
        applicable_when=(
            "Concepts that *are* games-at-heart: state machines, schedulers, "
            "queues, simulation, strategy, rules-with-feedback. Pick this over "
            "visual when the learner can drive the system, not just look at it."
        ),
        system_prompt=GAME_SYSTEM_PROMPT,
        generate=_gen_game,
    ))
    reg.register(Skill(
        name=TeachingMethod.WORKED_EXAMPLE.value,
        description=(
            "Markdown walkthrough: framing → fenced code → step-by-step trace "
            "→ why it matters"
        ),
        applicable_when=(
            "Step-by-step procedures with concrete code or math: algorithms, "
            "SQL queries, code patterns, mathematical derivations."
        ),
        system_prompt=WORKED_EXAMPLE_SYSTEM_PROMPT,
        generate=_gen_worked_example,
    ))

    return reg
