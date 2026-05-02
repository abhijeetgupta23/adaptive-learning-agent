"""Teaching-skill registry: maps each TeachingMethod to a Skill record.

Real generators are registered for the methods we actually implement (game,
worked_example). The rest are registered as **stubs** — frontmatter-only — so
they participate in router catalog text but fall back to demo placeholders at
teach time.

Why register stubs at all? Two reasons:
1. The pedagogy router still needs to *consider* them (curriculum may pick one).
2. The `catalog_text()` exposes them to the curriculum prompt so it can pick a
   pedagogy when planning. With stubs absent the curriculum agent would never
   suggest e.g. "feynman".
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

    # ---- stub skills (frontmatter only, no generator yet) ----
    # These let the pedagogy router catalog them and let curriculum pick them;
    # the dispatcher falls back to a demo placeholder when invoked.

    reg.register(Skill(
        name=TeachingMethod.VISUAL.value,
        description="Diagram or visualization of structure (Venn, tree, topology)",
        applicable_when=(
            "Topological / relational / spatial concepts best *seen* rather "
            "than *driven*: tree shapes, set relationships, layouts."
        ),
    ))
    reg.register(Skill(
        name=TeachingMethod.ANALOGY.value,
        description="Map a novel mechanism to a familiar one",
        applicable_when=(
            "Novel abstractions where the learner already understands a "
            "structurally-similar familiar system."
        ),
    ))
    reg.register(Skill(
        name=TeachingMethod.SOCRATIC.value,
        description="Guide via questions; learner constructs the understanding",
        applicable_when=(
            "Fuzzy conceptual understanding the learner must build themselves; "
            "ideas where direct telling sticks worse than self-derivation."
        ),
    ))
    reg.register(Skill(
        name=TeachingMethod.FEYNMAN.value,
        description="Have the learner teach it back; expose gaps",
        applicable_when=(
            "Capstone or self-test: the learner should be able to explain it "
            "in their own words to verify deep understanding."
        ),
    ))
    reg.register(Skill(
        name=TeachingMethod.RETRIEVAL.value,
        description="Spaced-recall drill of vocabulary or facts",
        applicable_when=(
            "Vocabulary, named entities, dates, formulas — content that just "
            "needs to be remembered, not derived."
        ),
    ))

    return reg
