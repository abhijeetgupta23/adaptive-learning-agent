"""Generic Skill primitive: frontmatter + lazy-loaded prompt + generator fn."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class StubSkillError(RuntimeError):
    """Raised when a stub skill (frontmatter-only) is invoked."""


@dataclass
class Skill(Generic[T, R]):
    """A unit of progressive disclosure.

    Tier 1 (frontmatter): name + description + applicable_when. Always visible.
    Tier 2 (system_prompt): full prompt; loaded only when this skill runs.
    Tier 3 (generate): the async callable that does the work. None for stubs
        (frontmatter-only registrations that participate in routing but don't
        produce output yet).
    """

    name: str
    description: str
    applicable_when: str
    system_prompt: str = ""
    generate: Callable[[T], Awaitable[R]] | None = None

    def frontmatter(self) -> str:
        """Tier-1 catalog entry. Cheap to include in any orchestrator prompt."""
        return (
            f"- {self.name}: {self.description}\n"
            f"  When to use: {self.applicable_when}"
        )

    def is_stub(self) -> bool:
        return self.generate is None

    async def run(self, payload: T) -> R:
        if self.generate is None:
            raise StubSkillError(
                f"Skill {self.name!r} is registered but has no generator."
            )
        return await self.generate(payload)


class SkillRegistry(Generic[T, R]):
    """Holds skills keyed by name; produces a frontmatter catalog on demand."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill[T, R]] = {}

    def register(self, skill: Skill[T, R]) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill {skill.name!r} already registered")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill[T, R] | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills)

    def catalog_text(self) -> str:
        """Tier-1 catalog: frontmatter for every skill, joined."""
        return "\n".join(s.frontmatter() for s in self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills


# Convenience: a Skill with no payload type at all (lambdas that ignore args).
AnyPayload = Any
AnySkill = Skill[Any, Any]
