import json
import os
from pathlib import Path

import pytest

from backend.schemas import DesiredDepth, KnowledgeLevel, LearningGoal, Modality
from evals.evaluators.intent_fidelity import (
    IntentFidelityEvalError,
    IntentFidelityResult,
    evaluate_intent_fidelity,
    load_golden_intent_cases,
)
from tests._fakes import FakeAnthropic, text_response

GOLDEN_DIR = Path(__file__).parent.parent / "evals" / "golden" / "intent"


def _goal(**overrides) -> LearningGoal:
    base = {
        "domain": "databases",
        "sub_goal": "understand SQL joins",
        "current_level": KnowledgeLevel.BEGINNER,
        "desired_depth": DesiredDepth.WORKING,
        "time_budget_hours": 4.0,
        "preferred_modalities": [Modality.VISUAL],
    }
    base.update(overrides)
    return LearningGoal(**base)


def _judge_payload(passed: bool = True, score: float = 0.95) -> dict:
    return {
        "field_scores": [
            {"field": "domain", "score": 1.0, "rationale": "exact"},
            {"field": "sub_goal", "score": 0.9, "rationale": "same scope"},
            {"field": "current_level", "score": 1.0, "rationale": "exact match"},
            {"field": "desired_depth", "score": 1.0, "rationale": "exact match"},
            {"field": "time_budget_hours", "score": 1.0, "rationale": "within 30%"},
            {"field": "preferred_modalities", "score": 0.8, "rationale": "jaccard 0.5"},
        ],
        "overall_score": score,
        "passed": passed,
        "rationale": "Goals substantially equivalent." if passed else "Wrong domain.",
    }


async def test_evaluator_parses_pass_verdict() -> None:
    client = FakeAnthropic([text_response(json.dumps(_judge_payload(True, 0.95)))])
    result = await evaluate_intent_fidelity(
        expected=_goal(),
        predicted=_goal(),
        judge=client,
    )
    assert isinstance(result, IntentFidelityResult)
    assert result.passed is True
    assert result.overall_score == 0.95
    assert len(result.field_scores) == 6
    assert {fs.field for fs in result.field_scores} == {
        "domain", "sub_goal", "current_level",
        "desired_depth", "time_budget_hours", "preferred_modalities",
    }


async def test_evaluator_parses_fail_verdict() -> None:
    client = FakeAnthropic([text_response(json.dumps(_judge_payload(False, 0.4)))])
    result = await evaluate_intent_fidelity(
        expected=_goal(),
        predicted=_goal(domain="cooking"),
        judge=client,
    )
    assert result.passed is False
    assert result.overall_score == 0.4


async def test_evaluator_rejects_malformed_judge_output() -> None:
    client = FakeAnthropic([text_response(json.dumps({"passed": True}))])  # missing fields
    with pytest.raises(IntentFidelityEvalError):
        await evaluate_intent_fidelity(
            expected=_goal(),
            predicted=_goal(),
            judge=client,
        )


async def test_evaluator_sends_both_goals_in_user_message() -> None:
    client = FakeAnthropic([text_response(json.dumps(_judge_payload()))])
    await evaluate_intent_fidelity(
        expected=_goal(sub_goal="learn inner joins"),
        predicted=_goal(sub_goal="understand LEFT joins"),
        judge=client,
    )
    call = client.messages.calls[0]
    user_content = call["messages"][0]["content"]
    assert "learn inner joins" in user_content
    assert "understand LEFT joins" in user_content
    assert "Intent Agent" in call["system"][0]["text"]


def test_five_golden_intent_cases_exist_and_validate() -> None:
    cases = load_golden_intent_cases(str(GOLDEN_DIR))
    assert len(cases) == 5
    ids = {c["id"] for c in cases}
    assert len(ids) == 5  # all unique
    for case in cases:
        assert case["conversation"], f"case {case['id']} has empty conversation"
        assert case["conversation"][-1]["role"] == "user", (
            f"case {case['id']} conversation should end on user turn"
        )
        # expected_goal must be a valid LearningGoal
        LearningGoal.model_validate(case["expected_goal"])


def test_golden_cases_cover_all_three_domains() -> None:
    cases = load_golden_intent_cases(str(GOLDEN_DIR))
    domains = {c["expected_goal"]["domain"] for c in cases}
    # loose check: at least one case per target domain area
    assert any("database" in d or "sql" in d.lower() for d in domains)
    assert any("italian" in d.lower() for d in domains)
    assert any(
        "machine learning" in d.lower() or "neural" in d.lower() for d in domains
    )


def test_load_golden_intent_cases_raises_on_missing_directory() -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        load_golden_intent_cases(os.path.join(str(GOLDEN_DIR), "does_not_exist"))
