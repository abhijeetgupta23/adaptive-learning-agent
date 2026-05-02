from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from backend.agents._parsing import (
    StructuredOutputError,
    extract_text,
    parse_json_payload,
)
from backend.orchestration.ptc import AnthropicLike, run_ptc
from backend.orchestration.tools import Tool, ToolRegistry
from backend.schemas.curriculum import Curriculum, LearningUnit
from backend.schemas.goal import LearningGoal
from backend.skills import SkillRegistry

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

_CURRICULUM_SYSTEM_PROMPT_TEMPLATE = """\
You are the Curriculum Agent for an adaptive learning platform.

You will receive a LearningGoal. Your job:
1. Research the domain using the web_search tool. Find 2-4 credible sources that \
cover the user's sub_goal at their desired_depth. Prefer primary sources (official \
documentation, well-known tutorials, academic material).
2. Plan an ordered curriculum of 3-7 learning units that take the user from their \
current_level to their desired_depth within their time_budget_hours.
3. Each unit MUST cite at least one retrievable source from your research.
4. For each unit, pick one `recommended_pedagogy` from the catalog below:

{PEDAGOGY_CATALOG}

Bias rule: when a unit could plausibly be `game` OR `visual`, pick `game`. At \
least one unit per curriculum should be `game` when the topic admits one (state \
machines, schedulers, queues, simulation, strategy).

Constraints:
- Total estimated time across units should roughly match time_budget_hours — do not \
plan a semester when the user has a weekend.
- Order matters: later units should build on earlier ones. Use the prerequisites \
field to declare dependencies (list of unit ids).
- Unit ids should be short and sequential: "u1", "u2", "u3", ...

After your research is done, output EXACTLY this JSON and nothing else. No prose, \
no code fences.

{"units": [
  {
    "id": "u1",
    "objective": "short, action-oriented learning objective",
    "prerequisites": [],
    "recommended_pedagogy": "visual",
    "grounding_sources": [
      {"url": "https://...", "title": "...", "snippet": "..."}
    ]
  },
  ...
]}
"""


def build_curriculum_system_prompt(registry: SkillRegistry) -> str:
    """Build the curriculum agent's system prompt with the live skill catalog.

    Progressive disclosure: only frontmatter (~50 tokens × N skills) goes into
    the prompt — the full per-skill teaching prompts are loaded only at teach
    time by the dispatcher in session.py.
    """
    return _CURRICULUM_SYSTEM_PROMPT_TEMPLATE.replace(
        "{PEDAGOGY_CATALOG}", registry.catalog_text(),
    )


class CurriculumAgentError(Exception):
    """Raised when the Curriculum Agent's output cannot be parsed."""


def _build_user_message(goal: LearningGoal) -> str:
    return (
        "Plan a curriculum for this learner. Research first using web_search, "
        "then output the final JSON.\n\n"
        f"{goal.model_dump_json(indent=2)}"
    )


def _parse_units(payload: dict[str, Any]) -> list[LearningUnit]:
    if "units" not in payload or not isinstance(payload["units"], list):
        raise CurriculumAgentError(f"Expected 'units' list in output, got: {payload}")
    try:
        return [LearningUnit.model_validate(u) for u in payload["units"]]
    except ValidationError as exc:
        raise CurriculumAgentError(f"Unit validation failed: {exc}") from exc


async def build_curriculum(
    goal: LearningGoal,
    *,
    client: AnthropicLike,
    search_tool: Tool,
    teaching_registry: SkillRegistry,
    additional_tools: list[Tool] | None = None,
    model: str = DEFAULT_MODEL,
    max_iterations: int = 10,
    max_tokens: int = 4096,
) -> Curriculum:
    """Research + plan a grounded curriculum for the given goal.

    `search_tool` is the primary grounding source (Tavily). `additional_tools`
    can supply complementary backends (e.g. Wikipedia); the curriculum LLM
    sees all of them and picks per-query based on each tool's description.
    """
    tool_registry = ToolRegistry()
    tool_registry.register(search_tool)
    for tool in additional_tools or []:
        tool_registry.register(tool)

    result = await run_ptc(
        client,
        model=model,
        system=build_curriculum_system_prompt(teaching_registry),
        messages=[{"role": "user", "content": _build_user_message(goal)}],
        tools=tool_registry,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        agent="curriculum",
    )

    try:
        final_text = extract_text(result.final_message)
        payload = parse_json_payload(final_text)
    except StructuredOutputError as exc:
        raise CurriculumAgentError(str(exc)) from exc
    units = _parse_units(payload)
    curriculum = Curriculum(goal=goal, units=units)
    _logger.info(
        "curriculum_built",
        extra={
            "unit_count": len(units),
            "iterations": result.iterations,
            "input_tokens": result.input_tokens_total,
            "output_tokens": result.output_tokens_total,
        },
    )
    return curriculum
