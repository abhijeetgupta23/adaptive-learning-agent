"""CLI runner for the eval harness.

Usage:
    uv run python -m evals.runner --all
    uv run python -m evals.runner --evaluator intent
    uv run python -m evals.runner --evaluator pedagogy --output report.json

The runner expects ANTHROPIC_API_KEY and TAVILY_API_KEY to be set when running
evaluators that need live calls (intent, curriculum_coherence, grounding,
game_quality, assessment_accuracy). Deterministic evaluators (pedagogy_fit,
learning_gain stub) do not require keys.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from backend.agents.assessment import assess_understanding
from backend.agents.curriculum import build_curriculum
from backend.agents.intent import IntentCompleteResponse, intent_turn
from backend.agents.pedagogy_router import route_pedagogy
from backend.orchestration.anthropic_client import get_anthropic_client
from backend.orchestration.logging import configure_logging
from backend.orchestration.ptc import AnthropicLike
from backend.retrieval.search_tool import build_web_search_tool
from backend.schemas.assessment import Verdict
from backend.schemas.curriculum import LearningUnit
from backend.schemas.goal import LearningGoal
from backend.schemas.state import LearnerState
from backend.schemas.teaching import TeachingMethod
from evals.evaluators.assessment_accuracy import evaluate_assessment_accuracy
from evals.evaluators.curriculum_coherence import evaluate_curriculum_coherence
from evals.evaluators.grounding import check_grounding
from evals.evaluators.intent_fidelity import (
    evaluate_intent_fidelity,
    load_golden_intent_cases,
)
from evals.evaluators.learning_gain import evaluate_learning_gain
from evals.evaluators.pedagogy_fit import evaluate_pedagogy_fit

_logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
GOLDEN_ROOT = REPO_ROOT / "evals" / "golden"

EVALUATORS = [
    "intent",
    "curriculum_coherence",
    "grounding",
    "pedagogy",
    "assessment",
    "learning_gain",
]


def _load_cases(subdir: str) -> list[dict[str, Any]]:
    directory = GOLDEN_ROOT / subdir
    if not directory.is_dir():
        return []
    cases: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


async def run_intent(
    client: AnthropicLike, cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else load_golden_intent_cases(str(GOLDEN_ROOT / "intent"))
    results: list[dict[str, Any]] = []
    for case in cases:
        expected = LearningGoal.model_validate(case["expected_goal"])
        try:
            turn = await intent_turn(client, conversation=case["conversation"])
        except Exception as exc:  # noqa: BLE001
            results.append({"case_id": case["id"], "passed": False, "error": str(exc)})
            continue
        if not isinstance(turn, IntentCompleteResponse):
            results.append({
                "case_id": case["id"], "passed": False,
                "error": "Agent did not finalize",
            })
            continue
        eval_res = await evaluate_intent_fidelity(
            expected=expected, predicted=turn.goal, judge=client,
        )
        results.append({
            "case_id": case["id"],
            "passed": eval_res.passed,
            "score": eval_res.overall_score,
        })
    return results


async def run_curriculum_coherence(
    client: AnthropicLike,
    search_tool,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("curriculum")
    results: list[dict[str, Any]] = []
    for case in cases:
        goal = LearningGoal.model_validate(case["goal"])
        try:
            curriculum = await build_curriculum(goal, client=client, search_tool=search_tool)
            eval_res = await evaluate_curriculum_coherence(curriculum, judge=client)
        except Exception as exc:  # noqa: BLE001
            results.append({"case_id": case["id"], "passed": False, "error": str(exc)})
            continue
        results.append({
            "case_id": case["id"],
            "passed": eval_res.passed,
            "score": eval_res.overall_score,
            "unit_count": len(curriculum.units),
        })
    return results


async def run_grounding(
    client: AnthropicLike,
    search_tool,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("curriculum")
    results: list[dict[str, Any]] = []
    for case in cases:
        goal = LearningGoal.model_validate(case["goal"])
        try:
            curriculum = await build_curriculum(goal, client=client, search_tool=search_tool)
            eval_res = await check_grounding(curriculum)
        except Exception as exc:  # noqa: BLE001
            results.append({"case_id": case["id"], "passed": False, "error": str(exc)})
            continue
        results.append({
            "case_id": case["id"],
            "passed": eval_res.passed,
            "score": eval_res.overall_score,
            "unique_sources": eval_res.unique_source_count,
        })
    return results


async def run_pedagogy_fit(
    client: AnthropicLike | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("pedagogy_fit")
    results: list[dict[str, Any]] = []
    for case in cases:
        unit = LearningUnit.model_validate(case["unit"])
        state = LearnerState.model_validate(case["learner_state"])
        acceptable = [TeachingMethod(m) for m in case["acceptable_methods"]]
        plan = await route_pedagogy(unit, state, llm_client=client)
        chosen = plan.methods[0].method
        eval_res = evaluate_pedagogy_fit(
            case_id=case["id"], chosen=chosen, acceptable=acceptable,
        )
        results.append({
            "case_id": case["id"],
            "passed": eval_res.passed,
            "chosen": chosen.value,
            "acceptable": [m.value for m in acceptable],
        })
    return results


async def run_assessment(
    client: AnthropicLike, cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("assessment")
    results: list[dict[str, Any]] = []
    for case in cases:
        unit = LearningUnit.model_validate(case["unit"])
        method = TeachingMethod(case["method"])
        expected = Verdict(case["expected_verdict"])
        try:
            predicted = await assess_understanding(
                unit, method, case["transcript"], client=client,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"case_id": case["id"], "passed": False, "error": str(exc)})
            continue
        eval_res = evaluate_assessment_accuracy(
            case_id=case["id"],
            predicted=predicted.verdict,
            expected=expected,
        )
        results.append({
            "case_id": case["id"],
            "passed": eval_res.passed,
            "predicted": predicted.verdict.value,
            "expected": expected.value,
            "rationale": eval_res.rationale,
        })
    return results


def run_learning_gain(cases: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("learning_gain")
    results: list[dict[str, Any]] = []
    for case in cases:
        eval_res = evaluate_learning_gain(
            case_id=case["id"],
            pre_score=case["pre_score"],
            post_score=case["post_score"],
            min_delta=case.get("min_delta", 0.2),
        )
        results.append({
            "case_id": case["id"],
            "passed": eval_res.passed,
            "delta": eval_res.delta,
        })
    return results


def summarize(report: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["=" * 60, "Eval Report", "=" * 60]
    total_passed = 0
    total_cases = 0
    for name, results in report.items():
        if not results:
            lines.append(f"{name}: (no cases)")
            continue
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        total_passed += passed
        total_cases += total
        lines.append(f"{name}: {passed}/{total} passed")
        for r in results:
            status = "[PASS]" if r.get("passed") else "[FAIL]"
            extras = ", ".join(
                f"{k}={v}" for k, v in r.items() if k not in ("case_id", "passed")
            )
            lines.append(f"  {status} {r['case_id']} {extras}")
    if total_cases:
        lines.append("-" * 60)
        lines.append(f"TOTAL: {total_passed}/{total_cases} passed")
    return "\n".join(lines)


async def run_selected(
    names: list[str],
    *,
    client: AnthropicLike | None,
    search_tool,
) -> dict[str, list[dict[str, Any]]]:
    report: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        if name == "intent":
            report[name] = await run_intent(client) if client else []
        elif name == "curriculum_coherence":
            if client and search_tool:
                report[name] = await run_curriculum_coherence(client, search_tool)
            else:
                report[name] = []
        elif name == "grounding":
            if client and search_tool:
                report[name] = await run_grounding(client, search_tool)
            else:
                report[name] = []
        elif name == "pedagogy":
            report[name] = await run_pedagogy_fit(client=client)
        elif name == "assessment":
            report[name] = await run_assessment(client) if client else []
        elif name == "learning_gain":
            report[name] = run_learning_gain()
        else:
            raise ValueError(f"Unknown evaluator: {name}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the adaptive-learning eval harness.")
    parser.add_argument("--all", action="store_true", help="Run all evaluators")
    parser.add_argument(
        "--evaluator", choices=EVALUATORS,
        help="Run a single evaluator",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args(argv)

    configure_logging()

    if not args.all and not args.evaluator:
        parser.print_help()
        return 2

    selected = EVALUATORS if args.all else [args.evaluator]

    client = None
    search_tool = None
    if os.getenv("ANTHROPIC_API_KEY"):
        client = get_anthropic_client()
    if os.getenv("TAVILY_API_KEY"):
        search_tool = build_web_search_tool()

    report = asyncio.run(run_selected(selected, client=client, search_tool=search_tool))

    print(summarize(report))

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total_passed = sum(
        sum(1 for r in results if r.get("passed")) for results in report.values()
    )
    total_cases = sum(len(results) for results in report.values())
    return 0 if total_passed == total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
