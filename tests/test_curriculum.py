import json
from pathlib import Path

import pytest

from backend.agents.curriculum import CurriculumAgentError, build_curriculum
from backend.agents.teaching.registry import build_teaching_registry
from backend.orchestration.tools import Tool
from backend.retrieval.search_tool import build_web_search_tool
from backend.retrieval.tavily import search_web
from backend.schemas import (
    Curriculum,
    DesiredDepth,
    KnowledgeLevel,
    LearningGoal,
    TeachingMethod,
)
from evals.evaluators.curriculum_coherence import (
    CurriculumCoherenceEvalError,
    evaluate_curriculum_coherence,
)
from evals.evaluators.grounding import check_grounding
from tests._fakes import FakeAnthropic, text_response, tool_use_response

GOLDEN_DIR = Path(__file__).parent.parent / "evals" / "golden" / "curriculum"


def _goal() -> LearningGoal:
    return LearningGoal(
        domain="databases",
        sub_goal="understand SQL joins",
        current_level=KnowledgeLevel.BEGINNER,
        desired_depth=DesiredDepth.WORKING,
        time_budget_hours=4.0,
    )


def _curriculum_payload() -> dict:
    return {
        "units": [
            {
                "id": "u1",
                "objective": "Understand relational tables and primary/foreign keys",
                "prerequisites": [],
                "recommended_pedagogy": "visual",
                "grounding_sources": [{
                    "url": "https://www.postgresql.org/docs/current/tutorial-join.html",
                    "title": "PostgreSQL tutorial: joins",
                    "snippet": "A join combines rows from two or more tables.",
                }],
            },
            {
                "id": "u2",
                "objective": "Distinguish INNER vs LEFT joins via worked examples",
                "prerequisites": ["u1"],
                "recommended_pedagogy": "worked_example",
                "grounding_sources": [{
                    "url": "https://sqlzoo.net/wiki/SELECT_from_Nobel_Tutorial",
                    "title": "SQLZoo: joins tutorial",
                    "snippet": "INNER JOIN matches keys; LEFT JOIN keeps unmatched left rows.",
                }],
            },
            {
                "id": "u3",
                "objective": "Apply joins in a scripted mini-game",
                "prerequisites": ["u2"],
                "recommended_pedagogy": "game",
                "grounding_sources": [{
                    "url": "https://mystery.knightlab.com/",
                    "title": "SQL Murder Mystery",
                    "snippet": "A detective game teaching SQL through investigation.",
                }],
            },
        ]
    }


# -------- Curriculum agent --------

async def test_build_curriculum_happy_path() -> None:
    async def fake_search(inp: dict) -> list[dict]:
        return [{"url": "https://example.com", "title": "t", "snippet": "s"}]

    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )
    client = FakeAnthropic([
        tool_use_response("t1", "web_search", {"query": "sql joins tutorial"}),
        text_response(json.dumps(_curriculum_payload())),
    ])

    curriculum = await build_curriculum(
        _goal(), client=client, search_tool=search_tool,
        teaching_registry=build_teaching_registry(client=client),
    )

    assert isinstance(curriculum, Curriculum)
    assert len(curriculum.units) == 3
    assert curriculum.units[0].id == "u1"
    assert curriculum.units[1].recommended_pedagogy is TeachingMethod.WORKED_EXAMPLE
    assert curriculum.goal == _goal()


async def test_build_curriculum_rejects_bad_json() -> None:
    async def fake_search(inp: dict) -> list[dict]: return []
    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )
    client = FakeAnthropic([text_response("no json here, just prose")])
    with pytest.raises(CurriculumAgentError):
        await build_curriculum(
        _goal(), client=client, search_tool=search_tool,
        teaching_registry=build_teaching_registry(client=client),
    )


async def test_build_curriculum_rejects_malformed_unit() -> None:
    async def fake_search(inp: dict) -> list[dict]: return []
    search_tool = Tool(
        name="web_search", description="",
        input_schema={"type": "object"}, handler=fake_search,
    )
    bad_payload = {"units": [{"id": "u1"}]}  # missing required fields
    client = FakeAnthropic([text_response(json.dumps(bad_payload))])
    with pytest.raises(CurriculumAgentError):
        await build_curriculum(
        _goal(), client=client, search_tool=search_tool,
        teaching_registry=build_teaching_registry(client=client),
    )


# -------- Search tool --------

async def test_web_search_tool_returns_structured_results() -> None:
    class FakeTavily:
        async def search(self, **kwargs):
            return {"results": [
                {"url": "https://a.com", "title": "A", "content": "alpha content"},
                {"url": "https://b.com", "title": "B", "content": "beta content"},
            ]}

    results = await search_web("x", client=FakeTavily())
    assert len(results) == 2
    assert results[0] == {"url": "https://a.com", "title": "A", "snippet": "alpha content"}


async def test_build_web_search_tool_callable_via_registry() -> None:
    class FakeTavily:
        async def search(self, **kwargs):
            return {"results": [{"url": "https://a.com", "title": "A", "content": "c"}]}

    tool = build_web_search_tool(client=FakeTavily())
    result = await tool.handler({"query": "joins"})
    assert result[0]["url"] == "https://a.com"


# -------- Grounding evaluator --------

def _make_curriculum(units_sources: list[list[str]]) -> Curriculum:
    from backend.schemas.curriculum import GroundingSource, LearningUnit
    units = []
    for idx, urls in enumerate(units_sources, start=1):
        units.append(LearningUnit(
            id=f"u{idx}",
            objective=f"unit {idx}",
            recommended_pedagogy=TeachingMethod.VISUAL,
            grounding_sources=[
                GroundingSource(url=url, title=f"src{i}")
                for i, url in enumerate(urls)
            ],
        ))
    return Curriculum(goal=_goal(), units=units)


async def test_grounding_passes_with_diverse_sources() -> None:
    curriculum = _make_curriculum([
        ["https://a.com/1"],
        ["https://b.com/2"],
        ["https://c.com/3"],
    ])
    result = await check_grounding(curriculum)
    assert result.passed is True
    assert result.unique_source_count == 3
    assert result.overall_score == 1.0


async def test_grounding_penalizes_duplicate_sources() -> None:
    curriculum = _make_curriculum([
        ["https://same.com/1"],
        ["https://same.com/1"],
        ["https://same.com/1"],
        ["https://same.com/1"],
    ])
    result = await check_grounding(curriculum)
    # 4 units, need >= 2 unique; only 1 → diversity penalty
    assert result.unique_source_count == 1
    assert result.overall_score < 1.0


async def test_grounding_reachable_check_with_fake_http() -> None:
    class FakeResp:
        def __init__(self, status_code): self.status_code = status_code

    class FakeHttp:
        def __init__(self, mapping): self.mapping = mapping
        async def head(self, url, **kwargs):
            return FakeResp(self.mapping.get(url, 500))
        async def get(self, url, **kwargs):
            return FakeResp(self.mapping.get(url, 500))

    curriculum = _make_curriculum([
        ["https://good.com/1"],
        ["https://bad.com/1"],
    ])
    http = FakeHttp({"https://good.com/1": 200, "https://bad.com/1": 404})
    result = await check_grounding(curriculum, verify_reachable=True, http_client=http)
    assert result.per_unit[0].sources[0].reachable is True
    assert result.per_unit[1].sources[0].reachable is False


# -------- Curriculum coherence evaluator --------

def _judge_payload(passed: bool = True, score: float = 0.9) -> dict:
    return {
        "axis_scores": [
            {"axis": "ordering", "score": 1.0, "rationale": "units build on each other"},
            {"axis": "completeness", "score": 0.9, "rationale": "covers core joins"},
            {"axis": "scope", "score": 0.85, "rationale": "fits 4-hour budget"},
            {"axis": "pedagogy_fit", "score": 0.85, "rationale": "visual + game fit"},
        ],
        "overall_score": score,
        "passed": passed,
        "rationale": "Coherent progression for a beginner.",
    }


async def test_coherence_evaluator_parses_pass() -> None:
    client = FakeAnthropic([text_response(json.dumps(_judge_payload(True, 0.9)))])
    curriculum = _make_curriculum([["https://a.com"], ["https://b.com"], ["https://c.com"]])
    result = await evaluate_curriculum_coherence(curriculum, judge=client)
    assert result.passed is True
    assert result.overall_score == 0.9
    assert len(result.axis_scores) == 4


async def test_coherence_evaluator_rejects_malformed_output() -> None:
    client = FakeAnthropic([text_response(json.dumps({"passed": True}))])
    curriculum = _make_curriculum([["https://a.com"]])
    with pytest.raises(CurriculumCoherenceEvalError):
        await evaluate_curriculum_coherence(curriculum, judge=client)


async def test_coherence_evaluator_sends_goal_and_units() -> None:
    client = FakeAnthropic([text_response(json.dumps(_judge_payload()))])
    curriculum = _make_curriculum([["https://a.com"]])
    await evaluate_curriculum_coherence(curriculum, judge=client)
    call = client.messages.calls[0]
    user_content = call["messages"][0]["content"]
    assert "databases" in user_content  # goal domain
    assert "u1" in user_content  # unit id from curriculum


# -------- Golden cases --------

def test_five_curriculum_golden_cases_exist_and_validate() -> None:
    files = sorted(GOLDEN_DIR.glob("*.json"))
    assert len(files) == 5
    ids = set()
    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        ids.add(case["id"])
        LearningGoal.model_validate(case["goal"])
        assert 1 <= case["min_units"] <= case["max_units"] <= 12
        assert isinstance(case["must_cover_keywords"], list)
    assert len(ids) == 5  # unique


def test_curriculum_golden_cases_cover_three_domains() -> None:
    files = sorted(GOLDEN_DIR.glob("*.json"))
    domains = {
        json.loads(p.read_text(encoding="utf-8"))["goal"]["domain"]
        for p in files
    }
    assert any("database" in d for d in domains)
    assert any("italian" in d.lower() for d in domains)
    assert any("machine learning" in d.lower() for d in domains)
