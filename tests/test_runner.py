import json

from backend.schemas.assessment import Verdict
from backend.schemas.teaching import TeachingMethod
from evals.evaluators.assessment_accuracy import evaluate_assessment_accuracy
from evals.evaluators.learning_gain import evaluate_learning_gain
from evals.evaluators.pedagogy_fit import evaluate_pedagogy_fit
from evals.runner import run_assessment, run_pedagogy_fit, summarize
from tests._fakes import FakeAnthropic, text_response

# -------- Deterministic evaluators --------

def test_pedagogy_fit_pass_and_fail() -> None:
    passed = evaluate_pedagogy_fit(
        case_id="x", chosen=TeachingMethod.GAME,
        acceptable=[TeachingMethod.GAME, TeachingMethod.VISUAL],
    )
    assert passed.passed is True

    failed = evaluate_pedagogy_fit(
        case_id="y", chosen=TeachingMethod.RETRIEVAL,
        acceptable=[TeachingMethod.GAME, TeachingMethod.VISUAL],
    )
    assert failed.passed is False
    assert "not in acceptable" in failed.rationale


def test_assessment_accuracy_exact_match() -> None:
    r = evaluate_assessment_accuracy(
        case_id="x", predicted=Verdict.PASS, expected=Verdict.PASS,
    )
    assert r.passed is True
    assert "Exact match" in r.rationale


def test_assessment_accuracy_false_pass_flagged() -> None:
    r = evaluate_assessment_accuracy(
        case_id="x", predicted=Verdict.PASS,
        expected=Verdict.NEEDS_DIFFERENT_APPROACH,
    )
    assert r.passed is False
    assert "False pass" in r.rationale


def test_assessment_accuracy_false_fail_flagged() -> None:
    r = evaluate_assessment_accuracy(
        case_id="x", predicted=Verdict.NEEDS_REINFORCEMENT, expected=Verdict.PASS,
    )
    assert r.passed is False
    assert "False fail" in r.rationale


def test_assessment_accuracy_near_miss_within_fail_family() -> None:
    r = evaluate_assessment_accuracy(
        case_id="x",
        predicted=Verdict.NEEDS_REINFORCEMENT,
        expected=Verdict.NEEDS_DIFFERENT_APPROACH,
    )
    assert r.passed is False
    assert "Near-miss" in r.rationale


def test_learning_gain_passes_above_threshold() -> None:
    r = evaluate_learning_gain(case_id="x", pre_score=0.2, post_score=0.7)
    assert r.passed is True
    assert abs(r.delta - 0.5) < 1e-9


def test_learning_gain_fails_below_threshold() -> None:
    r = evaluate_learning_gain(case_id="x", pre_score=0.5, post_score=0.55)
    assert r.passed is False


# -------- Runner integration with fakes --------

def _pedagogy_case(acceptable: list[str], recommended: str = "visual") -> dict:
    return {
        "id": "test_case",
        "unit": {
            "id": "u1",
            "objective": "test objective",
            "prerequisites": [],
            "recommended_pedagogy": recommended,
            "grounding_sources": [{"url": "https://x.com", "title": "t"}],
        },
        "learner_state": {"user_id": "demo"},
        "acceptable_methods": acceptable,
    }


async def test_run_pedagogy_fit_uses_curriculum_recommendation() -> None:
    # Rules-first router with no LLM: recommended is "visual", acceptable includes visual → pass.
    results = await run_pedagogy_fit(
        client=None,
        cases=[_pedagogy_case(["visual", "game"], recommended="visual")],
    )
    assert results[0]["passed"] is True
    assert results[0]["chosen"] == "visual"


async def test_run_pedagogy_fit_fails_when_recommendation_not_acceptable() -> None:
    results = await run_pedagogy_fit(
        client=None,
        cases=[_pedagogy_case(["game"], recommended="retrieval")],
    )
    assert results[0]["passed"] is False
    assert results[0]["chosen"] == "retrieval"


async def test_run_assessment_integration() -> None:
    case = {
        "id": "t1",
        "unit": {
            "id": "u1",
            "objective": "distinguish joins",
            "prerequisites": [],
            "recommended_pedagogy": "visual",
            "grounding_sources": [{"url": "https://x.com", "title": "t"}],
        },
        "method": "visual",
        "transcript": [{"role": "user", "content": "I get it."}],
        "expected_verdict": "pass",
    }
    client = FakeAnthropic([text_response(json.dumps({
        "verdict": "pass",
        "diagnostic_notes": "clear transfer",
        "confidence": 0.9,
    }))])
    results = await run_assessment(client, cases=[case])
    assert results[0]["passed"] is True
    assert results[0]["predicted"] == "pass"


async def test_run_assessment_flags_mismatch() -> None:
    case = {
        "id": "t2",
        "unit": {
            "id": "u1",
            "objective": "distinguish joins",
            "prerequisites": [],
            "recommended_pedagogy": "visual",
            "grounding_sources": [{"url": "https://x.com", "title": "t"}],
        },
        "method": "visual",
        "transcript": [{"role": "user", "content": "I'm lost"}],
        "expected_verdict": "needs_different_approach",
    }
    client = FakeAnthropic([text_response(json.dumps({
        "verdict": "pass",
        "diagnostic_notes": "seemed ok",
        "confidence": 0.7,
    }))])
    results = await run_assessment(client, cases=[case])
    assert results[0]["passed"] is False


def test_summarize_formats_report() -> None:
    report = {
        "pedagogy": [
            {"case_id": "p1", "passed": True, "chosen": "visual"},
            {"case_id": "p2", "passed": False, "chosen": "retrieval"},
        ],
        "assessment": [
            {"case_id": "a1", "passed": True, "predicted": "pass"},
        ],
    }
    text = summarize(report)
    assert "pedagogy: 1/2 passed" in text
    assert "assessment: 1/1 passed" in text
    assert "TOTAL: 2/3" in text
    assert "[PASS] p1" in text
    assert "[FAIL] p2" in text
