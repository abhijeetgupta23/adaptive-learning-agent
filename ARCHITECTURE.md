# Architecture

## Shape

A **workflow with agentic nodes**. The outer graph is deterministic. The nodes are agentic LLM calls with schema-grounded inputs and outputs. No agent framework — hand-rolled programmatic tool calling (PTC).

```
                                    ┌──────────────── backedges ───────────────┐
                                    │                                           │
                                    ▼                                           │
┌──────────┐    ┌────────────┐    ┌───────────┐    ┌─────────┐    ┌──────────┐  │
│  Intent  │───▶│ Curriculum │───▶│ Pedagogy  │───▶│  Teach  │───▶│  Assess  │──┤
└──────────┘    └────────────┘    │  Router   │    └─────────┘    └──────────┘  │
     ▲               ▲            └───────────┘         ▲             │         │
     │               │                                  │             │         │
     │               │                                  └── retry ────┘         │
     │               │                                                          │
     │               └───── revise-curriculum (N failures in a row) ────────────┘
     │
     └─── LearnerState ──▶ Memory ──▶ persisted to SQLite
```

### v1 backedges

1. **assess-fail-retry** — Router picks a different `TeachingMethod` for the same unit when the Assessment Agent returns `needs_different_approach`. Previously-tried methods are excluded.
2. **curriculum-revise-on-repeated-failure** — After N consecutive failures across units, Curriculum can re-plan from the current unit. *(Hook exists in the router; automatic re-plan deferred past v1.)*

Out of scope for v1: mid-unit prerequisite detours, cross-unit retrieval, re-intent mid-session.

## Agent roster

| Agent              | Input                            | Output                      | Model     | Role                                           |
| ------------------ | -------------------------------- | --------------------------- | --------- | ---------------------------------------------- |
| Intent             | free-form conversation           | `LearningGoal` or ask       | Sonnet    | structured goal extraction                     |
| Curriculum         | `LearningGoal`                   | `Curriculum` (grounded)     | Opus      | research + plan, every unit cites a source     |
| Pedagogy Router    | `LearningUnit` + `LearnerState`  | `TeachingPlan`              | rules → Opus fallback | pick method for this unit            |
| Game (flagship)    | `LearningUnit`                   | `GameSpec` (HTML+JS)        | Sonnet    | concept-as-interaction micro-game              |
| Analogy/Visual/etc.| `LearningUnit`                   | streamed explanation        | Sonnet    | *stubs in v1*                                  |
| Assessment         | transcript of teaching           | `AssessmentResult`          | Sonnet    | pass / retry-same / retry-different            |
| Memory             | `(state, unit, method, result)`  | updated `LearnerState`      | pure fn   | three-tier persistence                         |

## Principles

- **Schema-grounded I/O.** Every agent boundary is a Pydantic model. See [backend/schemas/](backend/schemas/). No free-form dict passing.
- **Grounded generation.** Every factual claim in a curriculum traces to a retrievable source. Validated by the Grounding evaluator ([evals/evaluators/grounding.py](evals/evaluators/grounding.py)).
- **Evals first-class.** Seven evaluators ship in-repo ([evals/evaluators/](evals/evaluators/)) with golden cases ([evals/golden/](evals/golden/)). Runner: `uv run python -m evals.runner --all`.
- **Streaming by default.** SSE events for every stage of the session ([backend/schemas/events.py](backend/schemas/events.py), [backend/orchestration/sse.py](backend/orchestration/sse.py)).
- **Observability.** Structured JSON logs with request IDs and per-agent timing ([backend/orchestration/logging.py](backend/orchestration/logging.py)).

## Directory layout

```
adaptive-learning-agent/
├── backend/
│   ├── agents/              # Intent, Curriculum, PedagogyRouter, Teaching/Game, Assessment, Memory
│   ├── schemas/             # Pydantic models + SSE event union
│   ├── orchestration/       # PTC harness, ToolRegistry, AnthropicClient, SSE, logging
│   ├── retrieval/           # Tavily wrapper + web_search Tool
│   ├── persistence/         # SQLAlchemy async + LearnerState store
│   ├── api/                 # FastAPI /api/session/run (SSE endpoint)
│   └── main.py              # lifespan: engine, client, search tool
├── frontend/
│   └── src/app/page.tsx     # minimal demo UI consuming SSE
├── evals/
│   ├── evaluators/          # 7 evaluators
│   ├── golden/              # golden cases per evaluator × domain
│   └── runner.py            # CLI: --all | --evaluator NAME
└── tests/                   # pytest, all agents/evaluators/persistence covered
```

## Seven evaluators

| Evaluator             | Kind               | Signal                                                   |
| --------------------- | ------------------ | -------------------------------------------------------- |
| Intent Fidelity       | LLM judge          | predicted `LearningGoal` matches gold per-field rubric   |
| Curriculum Coherence  | LLM judge          | ordering, completeness, scope, pedagogy_fit axes         |
| Grounding             | deterministic (+ optional HTTP) | structural + source diversity + reachability |
| Pedagogy Fit          | deterministic      | router's method ∈ gold `acceptable_methods`              |
| Game Quality          | LLM judge + gates  | solvable, on_concept, engagement + playability gate      |
| Assessment Accuracy   | deterministic      | agent verdict matches human gold                         |
| Learning Gain         | deterministic (stub) | post_score − pre_score ≥ threshold                     |

Deterministic evaluators run without an API key. Judge-based evaluators need `ANTHROPIC_API_KEY`.

## What to read next

- [HANDOFF.md](HANDOFF.md) — original project brief, task breakdown, scope
- [backend/orchestration/ptc.py](backend/orchestration/ptc.py) — the tool-use loop
- [backend/api/session.py](backend/api/session.py) — the full session flow, end-to-end
