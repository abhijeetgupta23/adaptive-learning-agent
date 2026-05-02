"""Skill registry: progressive token disclosure for orchestrators.

The point of skills here is **token-budget economics**:

- Tier 1 — frontmatter (`name`, `description`, `applicable_when`): always visible to
  the orchestrator. Cheap. ~50 tokens per skill.
- Tier 2 — full `system_prompt`: loaded into the LLM context **only when this skill
  is selected**. Expensive but bounded. ~500-1500 tokens.
- Tier 3 — tool `functions`: registered with the LLM only when the skill is active.

A naïve dispatcher loads every method's prompt every turn. With N=32 skills × 1k
tokens, that's 32k input tokens per call before user message. Progressive disclosure
keeps the orchestrator's catalog at ~350 tokens (7 × 50) and only inflates context
on the actual chosen path.
"""
from backend.skills.registry import (
    Skill,
    SkillRegistry,
    StubSkillError,
)

__all__ = ["Skill", "SkillRegistry", "StubSkillError"]
