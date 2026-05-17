"""Compare game-generation quality, latency, and cost across two models.

Standalone experiment script for the SLM-routing decision. Runs
`generate_game` once per (unit, model) pair, scores each output with the
existing `game_quality` evaluator (judge = Sonnet, same for both, to keep
comparison fair), records per-call cost via the session_accumulator, and
prints a markdown comparison table.

Usage:
    uv run python -m evals.game_model_compare \\
        --models claude-sonnet-4-6,claude-haiku-4-5-20251001 \\
        --output evals/game_model_compare_report.json

Requires ANTHROPIC_API_KEY. Cost estimate: ~$2-5 total for the default
3-unit run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pick up ANTHROPIC_API_KEY from .env before backend imports

from backend.agents.teaching.game import GameAgentError, generate_game  # noqa: E402
from backend.orchestration.anthropic_client import get_anthropic_client  # noqa: E402
from backend.orchestration.cost import session_accumulator  # noqa: E402
from backend.orchestration.logging import configure_logging  # noqa: E402
from backend.schemas.curriculum import LearningUnit  # noqa: E402
from evals.evaluators.game_quality import (  # noqa: E402
    GameQualityEvalError,
    evaluate_game_quality,
)

_logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
GOLDEN_ROOT = REPO_ROOT / "evals" / "golden" / "pedagogy_fit"

# Haiku-first: prove the default holds before considering Sonnet as fallback.
DEFAULT_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]
JUDGE_MODEL = "claude-sonnet-4-6"  # same judge across runs for fairness


@dataclass
class RunRecord:
    case_id: str
    model: str
    unit_objective: str
    success: bool
    passed: bool | None
    overall_score: float | None
    playable: bool | None
    latency_s: float
    cost_usd: float
    iterations: int | None
    rationale: str | None
    error: str | None


def _load_game_acceptable_cases() -> list[dict]:
    cases: list[dict] = []
    for path in sorted(GOLDEN_ROOT.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if "game" in case.get("acceptable_methods", []):
            cases.append(case)
    return cases


async def _one_run(
    *,
    case: dict,
    model: str,
    client,
) -> RunRecord:
    unit = LearningUnit.model_validate(case["unit"])
    case_id = case["id"]

    t0 = time.monotonic()
    cost_usd = 0.0
    spec = None
    err: str | None = None
    iterations: int | None = None

    with session_accumulator() as acc:
        try:
            spec = await generate_game(unit, client=client, model=model)
            iterations = len(spec.tool_call_log)
        except GameAgentError as exc:
            err = f"generate_game failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            err = f"unexpected: {type(exc).__name__}: {exc}"
        cost_usd = acc.total_usd

    latency_s = time.monotonic() - t0

    if spec is None:
        return RunRecord(
            case_id=case_id, model=model,
            unit_objective=unit.objective, success=False,
            passed=None, overall_score=None, playable=None,
            latency_s=round(latency_s, 2), cost_usd=round(cost_usd, 4),
            iterations=iterations, rationale=None, error=err,
        )

    try:
        eval_res = await evaluate_game_quality(
            spec, unit, judge=client, model=JUDGE_MODEL,
        )
    except GameQualityEvalError as exc:
        return RunRecord(
            case_id=case_id, model=model,
            unit_objective=unit.objective, success=True,
            passed=False, overall_score=None, playable=None,
            latency_s=round(latency_s, 2), cost_usd=round(cost_usd, 4),
            iterations=iterations, rationale=None,
            error=f"judge_parse: {exc}",
        )

    return RunRecord(
        case_id=case_id, model=model,
        unit_objective=unit.objective, success=True,
        passed=eval_res.passed,
        overall_score=round(eval_res.overall_score, 3),
        playable=eval_res.playable,
        latency_s=round(latency_s, 2),
        cost_usd=round(cost_usd, 4),
        iterations=iterations,
        rationale=eval_res.rationale,
        error=None,
    )


def _summarize(records: list[RunRecord]) -> dict:
    by_model: dict[str, list[RunRecord]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)

    summary: dict = {}
    for model, rs in by_model.items():
        passed = sum(1 for r in rs if r.passed)
        completed = sum(1 for r in rs if r.success)
        latencies = [r.latency_s for r in rs if r.success]
        costs = [r.cost_usd for r in rs if r.success]
        summary[model] = {
            "n_cases": len(rs),
            "n_completed": completed,
            "n_passed": passed,
            "pass_rate": round(passed / len(rs), 3) if rs else 0.0,
            "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
            "latency_p99_s": round(
                statistics.quantiles(latencies, n=100)[98], 2,
            ) if len(latencies) >= 2 else (
                round(latencies[0], 2) if latencies else None
            ),
            "cost_total_usd": round(sum(costs), 4),
            "cost_mean_usd": round(sum(costs) / len(costs), 4) if costs else 0.0,
        }
    return summary


def _markdown_table(summary: dict) -> str:
    headers = [
        "Model", "Cases", "Completed", "Passed", "Pass-rate",
        "Latency p50 (s)", "Latency p99 (s)", "Mean cost ($)", "Total cost ($)",
    ]
    rows = []
    for model, s in summary.items():
        rows.append([
            model, s["n_cases"], s["n_completed"], s["n_passed"],
            f"{s['pass_rate']:.0%}",
            s["latency_p50_s"] if s["latency_p50_s"] is not None else "-",
            s["latency_p99_s"] if s["latency_p99_s"] is not None else "-",
            f"{s['cost_mean_usd']:.4f}",
            f"{s['cost_total_usd']:.4f}",
        ])
    sep = "|".join(["---"] * len(headers))
    out = ["|" + "|".join(headers) + "|", "|" + sep + "|"]
    for row in rows:
        out.append("|" + "|".join(str(c) for c in row) + "|")
    return "\n".join(out)


async def main_async(models: list[str], output: Path | None) -> int:
    client = get_anthropic_client()
    cases = _load_game_acceptable_cases()
    if not cases:
        print("No game-acceptable cases found in", GOLDEN_ROOT, file=sys.stderr)
        return 2

    print(f"Comparing {len(models)} models across {len(cases)} units:")
    for m in models:
        print(f"  - {m}")
    for c in cases:
        print(f"  - {c['id']}: {c['unit']['objective']}")
    print()

    records: list[RunRecord] = []
    for case in cases:
        for model in models:
            print(f"running {case['id']} / {model} ...", flush=True)
            rec = await _one_run(case=case, model=model, client=client)
            records.append(rec)
            status = "PASS" if rec.passed else (
                "FAIL" if rec.success else "ERROR"
            )
            print(
                f"  {status}  score={rec.overall_score}  "
                f"latency={rec.latency_s}s  cost=${rec.cost_usd}  "
                f"iters={rec.iterations}"
            )

    summary = _summarize(records)
    table = _markdown_table(summary)

    print()
    print("=" * 72)
    print("Comparison summary")
    print("=" * 72)
    print(table)
    print()

    if output is not None:
        output.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "records": [asdict(r) for r in records],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {output}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare game-generation across models.",
    )
    parser.add_argument(
        "--models", type=str, default=",".join(DEFAULT_MODELS),
        help="Comma-separated model IDs to compare.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write a JSON report.",
    )
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    configure_logging()
    return asyncio.run(main_async(models, args.output))


if __name__ == "__main__":
    sys.exit(main())
