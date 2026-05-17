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
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from backend.agents.assessment import assess_understanding
from backend.agents.curriculum import build_curriculum
from backend.agents.intent import IntentCompleteResponse, intent_turn
from backend.agents.pedagogy_router import route_pedagogy
from backend.orchestration.anthropic_client import get_anthropic_client
from backend.orchestration.cost import session_accumulator
from backend.orchestration.logging import configure_logging
from backend.orchestration.ptc import AnthropicLike
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


async def _run_case_tracked(
    case_id: str, body: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Run one case under cost + latency tracking.

    `body` returns a result dict (without case_id/latency_s/cost_usd). We
    inject those three keys here so every evaluator's output has a uniform
    shape and aggregation in `summarize` works without per-evaluator special-casing.
    """
    t0 = time.monotonic()
    with session_accumulator() as acc:
        try:
            result = await body()
        except Exception as exc:  # noqa: BLE001
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        cost = acc.total_usd
    result.setdefault("case_id", case_id)
    result["latency_s"] = round(time.monotonic() - t0, 3)
    result["cost_usd"] = round(cost, 4)
    return result


async def run_intent(
    client: AnthropicLike, cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else load_golden_intent_cases(str(GOLDEN_ROOT / "intent"))
    results: list[dict[str, Any]] = []
    for case in cases:
        async def body(case=case) -> dict[str, Any]:
            expected = LearningGoal.model_validate(case["expected_goal"])
            turn = await intent_turn(client, conversation=case["conversation"])
            if not isinstance(turn, IntentCompleteResponse):
                return {"passed": False, "error": "Agent did not finalize"}
            eval_res = await evaluate_intent_fidelity(
                expected=expected, predicted=turn.goal, judge=client,
            )
            return {"passed": eval_res.passed, "score": eval_res.overall_score}
        results.append(await _run_case_tracked(case["id"], body))
    return results


async def run_curriculum_coherence(
    client: AnthropicLike,
    search_tool,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("curriculum")
    results: list[dict[str, Any]] = []
    for case in cases:
        async def body(case=case) -> dict[str, Any]:
            goal = LearningGoal.model_validate(case["goal"])
            curriculum = await build_curriculum(goal, client=client, search_tool=search_tool)
            eval_res = await evaluate_curriculum_coherence(curriculum, judge=client)
            return {
                "passed": eval_res.passed,
                "score": eval_res.overall_score,
                "unit_count": len(curriculum.units),
            }
        results.append(await _run_case_tracked(case["id"], body))
    return results


async def run_grounding(
    client: AnthropicLike,
    search_tool,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("curriculum")
    results: list[dict[str, Any]] = []
    for case in cases:
        async def body(case=case) -> dict[str, Any]:
            goal = LearningGoal.model_validate(case["goal"])
            curriculum = await build_curriculum(goal, client=client, search_tool=search_tool)
            eval_res = await check_grounding(curriculum)
            return {
                "passed": eval_res.passed,
                "score": eval_res.overall_score,
                "unique_sources": eval_res.unique_source_count,
            }
        results.append(await _run_case_tracked(case["id"], body))
    return results


async def run_pedagogy_fit(
    client: AnthropicLike | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("pedagogy_fit")
    results: list[dict[str, Any]] = []
    for case in cases:
        async def body(case=case) -> dict[str, Any]:
            unit = LearningUnit.model_validate(case["unit"])
            state = LearnerState.model_validate(case["learner_state"])
            acceptable = [TeachingMethod(m) for m in case["acceptable_methods"]]
            plan = await route_pedagogy(unit, state, llm_client=client)
            chosen = plan.methods[0].method
            eval_res = evaluate_pedagogy_fit(
                case_id=case["id"], chosen=chosen, acceptable=acceptable,
            )
            return {
                "passed": eval_res.passed,
                "chosen": chosen.value,
                "acceptable": [m.value for m in acceptable],
            }
        results.append(await _run_case_tracked(case["id"], body))
    return results


async def run_assessment(
    client: AnthropicLike, cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("assessment")
    results: list[dict[str, Any]] = []
    for case in cases:
        async def body(case=case) -> dict[str, Any]:
            unit = LearningUnit.model_validate(case["unit"])
            method = TeachingMethod(case["method"])
            expected = Verdict(case["expected_verdict"])
            predicted = await assess_understanding(
                unit, method, case["transcript"], client=client,
            )
            eval_res = evaluate_assessment_accuracy(
                case_id=case["id"],
                predicted=predicted.verdict,
                expected=expected,
            )
            return {
                "passed": eval_res.passed,
                "predicted": predicted.verdict.value,
                "expected": expected.value,
                "rationale": eval_res.rationale,
            }
        results.append(await _run_case_tracked(case["id"], body))
    return results


def run_learning_gain(cases: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    cases = cases if cases is not None else _load_cases("learning_gain")
    results: list[dict[str, Any]] = []
    for case in cases:
        t0 = time.monotonic()
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
            "latency_s": round(time.monotonic() - t0, 3),
            "cost_usd": 0.0,
        })
    return results


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-evaluator aggregates: pass-rate, latency p50/p99, cost total/mean."""
    n = len(results)
    if n == 0:
        return {
            "n_cases": 0, "n_passed": 0, "pass_rate": 0.0,
            "latency_p50_s": None, "latency_p99_s": None,
            "cost_total_usd": 0.0, "cost_mean_usd": 0.0,
        }
    passed = sum(1 for r in results if r.get("passed"))
    latencies = [r["latency_s"] for r in results if "latency_s" in r]
    costs = [r.get("cost_usd", 0.0) for r in results]
    return {
        "n_cases": n,
        "n_passed": passed,
        "pass_rate": round(passed / n, 3),
        "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p99_s": (
            round(statistics.quantiles(latencies, n=100)[98], 3)
            if len(latencies) >= 2 else (round(latencies[0], 3) if latencies else None)
        ),
        "cost_total_usd": round(sum(costs), 4),
        "cost_mean_usd": round(sum(costs) / n, 4),
    }


def _grand_total(aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_passed = 0
    total_cases = 0
    total_cost = 0.0
    for agg in aggregates.values():
        total_passed += agg["n_passed"]
        total_cases += agg["n_cases"]
        total_cost += agg["cost_total_usd"]
    return {
        "n_cases": total_cases,
        "n_passed": total_passed,
        "pass_rate": round(total_passed / total_cases, 3) if total_cases else 0.0,
        "cost_total_usd": round(total_cost, 4),
    }


def summarize(report: dict[str, list[dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    """Return (human-readable text, structured aggregates dict)."""
    lines = ["=" * 72, "Eval Report", "=" * 72]
    aggregates: dict[str, dict[str, Any]] = {}
    for name, results in report.items():
        agg = _aggregate(results)
        aggregates[name] = agg
        if agg["n_cases"] == 0:
            lines.append(f"{name}: (no cases)")
            continue
        lines.append(
            f"{name}: {agg['n_passed']}/{agg['n_cases']} passed  "
            f"|  p50 {agg['latency_p50_s']}s  p99 {agg['latency_p99_s']}s  "
            f"|  cost ${agg['cost_total_usd']:.4f} "
            f"(mean ${agg['cost_mean_usd']:.4f}/case)"
        )
        for r in results:
            status = "[PASS]" if r.get("passed") else "[FAIL]"
            extras = ", ".join(
                f"{k}={v}" for k, v in r.items()
                if k not in ("case_id", "passed", "latency_s", "cost_usd")
            )
            lines.append(
                f"  {status} {r['case_id']}  "
                f"{r.get('latency_s', '?')}s  ${r.get('cost_usd', 0):.4f}  "
                f"{extras}"
            )
    total = _grand_total(aggregates)
    if total["n_cases"]:
        lines.append("-" * 72)
        lines.append(
            f"TOTAL: {total['n_passed']}/{total['n_cases']} passed  "
            f"|  cost ${total['cost_total_usd']:.4f}"
        )
    return "\n".join(lines), {"per_evaluator": aggregates, "total": total}


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
    if os.getenv("ANTHROPIC_API_KEY"):
        client = get_anthropic_client()
    # `search_tool=None` makes the curriculum agent fall back to Anthropic's
    # server-side web_search — no separate vendor key needed.
    report = asyncio.run(run_selected(selected, client=client, search_tool=None))

    text, aggregates = summarize(report)
    print(text)

    if args.output:
        args.output.write_text(
            json.dumps(
                {"summary": aggregates, "results": report},
                indent=2,
            ),
            encoding="utf-8",
        )

    total = aggregates["total"]
    return 0 if total["n_passed"] == total["n_cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
