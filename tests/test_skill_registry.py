"""Tests for backend.skills.SkillRegistry — progressive disclosure primitive."""
from __future__ import annotations

import pytest

from backend.skills import Skill, SkillRegistry, StubSkillError


def test_frontmatter_format():
    skill = Skill[None, str](
        name="game",
        description="Generate a playable micro-game",
        applicable_when="State machines, schedulers, simulation",
    )
    fm = skill.frontmatter()
    assert "game" in fm
    assert "Generate a playable" in fm
    assert "When to use:" in fm
    assert "State machines" in fm


def test_register_and_get():
    reg: SkillRegistry[None, str] = SkillRegistry()
    a = Skill[None, str](name="a", description="d", applicable_when="when")
    reg.register(a)
    assert reg.get("a") is a
    assert reg.get("missing") is None
    assert "a" in reg
    assert len(reg) == 1


def test_duplicate_registration_rejected():
    reg: SkillRegistry[None, str] = SkillRegistry()
    reg.register(Skill[None, str](name="a", description="d", applicable_when="w"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Skill[None, str](name="a", description="d", applicable_when="w"))


def test_catalog_text_joins_all_frontmatter():
    reg: SkillRegistry[None, str] = SkillRegistry()
    reg.register(Skill(name="game", description="play", applicable_when="state"))
    reg.register(Skill(name="visual", description="see", applicable_when="topology"))
    cat = reg.catalog_text()
    assert "- game: play" in cat
    assert "- visual: see" in cat
    # exactly two frontmatter blocks
    assert cat.count("When to use:") == 2


@pytest.mark.asyncio
async def test_run_invokes_generator():
    async def gen(payload: int) -> str:
        return f"got {payload}"
    skill = Skill[int, str](
        name="echo", description="echoes", applicable_when="ints",
        generate=gen,
    )
    assert await skill.run(7) == "got 7"
    assert not skill.is_stub()


@pytest.mark.asyncio
async def test_stub_skill_raises_on_run():
    skill = Skill[int, str](name="x", description="d", applicable_when="w")
    assert skill.is_stub()
    with pytest.raises(StubSkillError, match="x"):
        await skill.run(1)
