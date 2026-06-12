> **⚠️ Superseded by → [ai-agent-to-learn-faster-and-better](https://github.com/abhijeetgupta23/ai-agent-to-learn-faster-and-better)** — all development continues there.

# adaptive-learning-agent

> An agent that generates a Civ-3-style learning moment for any concept, on demand. Tell it what you want to learn, get a playable, grounded micro-game in ~30 seconds.

![demo](v2-split-game.png)

---

## Thesis

I learned more history from Civ 3 than from any textbook. Kerbal taught me orbital mechanics; Factorio taught me systems. **Games are how humans actually internalize models** — they just take years to build. Squire & Barab documented this exact effect for Civ III at the University of Wisconsin¹; a 2026 three-level meta-analysis across 1,029 effect sizes confirms moderate-to-large gains in cognition, motivation, and engagement from game-based learning².

Meanwhile, attention is collapsing. MIT/Stanford's longitudinal study (45K participants, 13 years) shows the average attention span has fallen 36.7% since 2000, to **7.6 seconds**³. **52% of people skip videos longer than a minute** even on topics they care about. Long prose and three-hour podcasts are losing the war.

Hand-crafted educational games are amazing (Civ, Kerbal, Factorio, DragonBox, Brilliant) — but they take years per topic. **LLMs make on-demand generation finally economic.** This project is the smallest credible version of "every concept gets a Civ-3-class learning moment, generated when you ask for it."

---

## What's in the box

The infrastructure, top-down:

- **Hand-rolled multi-agent workflow.** Intent → Curriculum → Router → Teach → Assess → Memory, with backedges for retry and curriculum revision. Not a monolithic prompt. Not an agent framework.
- **Programmatic Tool Calling (PTC).** ~130-line orchestration loop ([backend/orchestration/ptc.py](backend/orchestration/ptc.py)). Every agent boundary is a [Pydantic schema](backend/schemas/) — no free-form dict passing.
- **Flagship: on-the-fly game generation.** Concept-as-interaction micro-games rendered in a sandboxed iframe. The Game Agent runs its own PTC loop with six forced-order tools (design → validate → submit) grounded in the MDA framework⁴ ([backend/agents/teaching/game.py](backend/agents/teaching/game.py), [game_design.py](backend/agents/teaching/game_design.py)).
- **Worked-example fallback.** When a concept resists game-ification, [worked_example.py](backend/agents/teaching/worked_example.py) takes over. The router's only decision is "game or fall back."
- **Grounded curricula.** Every unit cites a retrievable source. Validated by the [grounding evaluator](evals/evaluators/grounding.py).
- **Haiku-first model routing, validated by eval.** All agents default to Claude Haiku 4.5. The game agent — the most demanding generator — hits **3/3 (100%) pass-rate** under the existing `game_quality` evaluator, mean score 0.79, p50 latency 62s, mean cost **$0.11/game** ([evals/game_haiku_only_report.json](evals/game_haiku_only_report.json)). Sonnet remains available as a fallback the eval would gate to if pass-rate slipped ([evals/game_model_compare.py](evals/game_model_compare.py)).
- **SSE streaming** with a typed event union ([backend/schemas/events.py](backend/schemas/events.py)).
- **Evals first-class.** 7 evaluators × 21 golden cases × 3 domains, with **latency p50/p99 + cost-per-case surfaced in the report** ([evals/runner.py](evals/runner.py)).
- **187 tests, ruff clean, frontend typed.**

¹ Squire & Barab, *Replaying History: Learning World History Through Playing Civilization III*, LearnTechLib.
² Xu & Dai, *Exploring the Efficacy of Game-Based Pedagogy: A Three-Level Meta-Analysis*, 2026.
³ NPR, Feb 2026: *Researchers say when it comes to our attention spans, we are at war.*
⁴ Hunicke, LeBlanc & Zubek, *MDA: A Formal Approach to Game Design and Game Research.*

---

## Architecture at a glance

```
                                ┌──────────────── backedges ───────────────┐
                                │                                          │
                                ▼                                          │
┌────────┐   ┌────────────┐   ┌──────────┐   ┌───────┐   ┌──────────┐      │
│ Intent │──▶│ Curriculum │──▶│ Pedagogy │──▶│ Teach │──▶│  Assess  │──────┤
└────────┘   └────────────┘   │  Router  │   └───────┘   └──────────┘      │
     ▲             ▲          └──────────┘        ▲           │            │
     │             │                               └── retry ─┘            │
     │             └──── revise-curriculum (N consecutive failures) ───────┘
     │
     └── LearnerState ──▶ Memory ──▶ SQLite
```

Outer graph is a deterministic workflow. Nodes are agentic LLM calls with schema-grounded I/O. Full topology in [ARCHITECTURE.md](ARCHITECTURE.md).

### Agent roster

| Agent              | Model    | Role                                                   |
| ------------------ | -------- | ------------------------------------------------------ |
| Intent             | Haiku 4.5 | Conversational goal extraction → `LearningGoal`        |
| Curriculum         | Haiku 4.5 | Web research + ordered plan; every unit grounded       |
| Pedagogy Router    | rules → Haiku 4.5 | Default: route to Game. Fall back to Worked Example only for un-gameable concepts. |
| Game *(flagship)*  | Haiku 4.5 | Generate self-contained HTML/JS micro-game in a sandboxed iframe |
| Worked Example     | Haiku 4.5 | Silent fallback when a concept resists game-ification  |
| Assessment         | Haiku 4.5 | pass / retry-same / retry-different                    |
| Memory             | pure fn  | LearnerState, persisted to SQLite                      |

---

## The three domains

The runtime is domain-general (`LearningGoal.domain` is a free-form string). Eval coverage spans three domains picked to stress the game agent on different shapes of content:

| Domain                    | Stresses                                  |
| ------------------------- | ----------------------------------------- |
| **SQL joins**             | Set-theoretic, spatial intuition          |
| **Italian for travel**    | Vocabulary recall via game mechanic       |
| **How neural nets learn** | Abstract conceptual reasoning             |

---

## The seven evaluators

| Evaluator             | Kind                            | Signal                                                |
| --------------------- | ------------------------------- | ----------------------------------------------------- |
| Intent Fidelity       | LLM judge                       | Predicted goal matches gold per-field rubric          |
| Curriculum Coherence  | LLM judge                       | Ordering, completeness, scope, pedagogy fit           |
| Grounding             | deterministic (+ optional HTTP) | Structural + source diversity + reachability          |
| Pedagogy Fit          | deterministic                   | Router's method ∈ gold `acceptable_methods`           |
| Game Quality          | LLM judge + gates               | Solvable, on-concept, engagement + playability gate   |
| Assessment Accuracy   | deterministic                   | Verdict matches human gold                            |
| Learning Gain         | deterministic                   | post_score − pre_score ≥ threshold                    |

Deterministic evaluators run with no API key. Judge-based evaluators need `ANTHROPIC_API_KEY`.

---

## Setup

Prereqs: Python 3.11+, Node.js 20+, [`uv`](https://docs.astral.sh/uv/).

```bash
# backend
uv venv
uv pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY + TAVILY_API_KEY

# frontend
cd frontend && npm install
```

## Run

```bash
# backend (from repo root)
uv run uvicorn backend.main:app --reload

# frontend (in a second terminal)
cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and click **Start session**. The frontend consumes the SSE stream from `/api/session/run` and renders each event type — `intent_update`, `curriculum_ready`, `unit_started`, `teaching_chunk`, `assessment_result`, `complete`. Games render in sandboxed iframes.

Headless verification:

```bash
curl -N -X POST http://127.0.0.1:8000/api/session/run \
  -H "Content-Type: application/json" \
  -d '{"goal":{"domain":"databases","sub_goal":"INNER vs LEFT joins",
       "current_level":"beginner","desired_depth":"working",
       "time_budget_hours":2.0,"preferred_modalities":["visual"]},
       "user_id":"demo"}'
```

## Test

```bash
uv run pytest                  # 187 tests
uv run ruff check .            # lint
cd frontend && npm run build   # frontend typecheck + build
```

## Evals

```bash
# deterministic only — no API key needed
uv run python -m evals.runner --evaluator pedagogy
uv run python -m evals.runner --evaluator assessment --output report.json

# full suite (judge-based evaluators included)
uv run python -m evals.runner --all --output eval_report.json

# model bake-off for the game agent (Haiku vs Sonnet, eval-gated)
uv run python -m evals.game_model_compare --output evals/game_model_compare_report.json
```

Each report includes **pass-rate, latency p50/p99, and cost per case** ([cost.py](backend/orchestration/cost.py) records both regular and cache-tier pricing). The CLI exits non-zero if any case fails so it composes cleanly with CI.

### Game-agent eval results (Haiku 4.5, May 2026)

Live run against the 3 game-acceptable golden cases ([evals/game_haiku_only_report.json](evals/game_haiku_only_report.json)):

| Case | Concept | Score | Latency | Cost | UI archetype |
| --- | --- | --- | --- | --- | --- |
| `pedagogy_sql_01` | INNER vs LEFT JOIN | 0.770 | 62.3s | $0.082 | drag_drop |
| `pedagogy_sql_02` | Multi-join query | 0.767 | 60.3s | $0.075 | drag_drop |
| `pedagogy_italian_01` | 50 Italian travel phrases | 0.833 | 130.6s | $0.167 | button_array |
| **Aggregate** | | **3/3 pass (100%)** | **p50 62.3s · p99 196.2s** | **mean $0.108 · total $0.324** | |

Pass threshold: `game_quality` requires mean ≥ 0.70 AND every axis ≥ 0.5. Judge: Sonnet 4.6 (kept on Sonnet for grader stability across model swaps).

Reading the numbers:

- **Cost is small enough that the "Civ-3 moment on demand" pitch is real economics, not aspiration.** ~$0.11 per generated game.
- **Latency is the soft spot:** 60-130s wall-clock per game. The SSE stream surfaces design progress (PTC tool calls) so the wait is not opaque, but it's the obvious target for any future "speed up" work.
- The 50-phrase Italian case is the outlier — bigger HTML output (20 KB vs ~13 KB), more interactive elements, ~2x latency and cost. Haiku spends more tokens when the concept demands it.

---

## Scope

**v1 is:** single-user, local-first, domain-general (eval coverage across SQL joins, Italian travel, neural-net attention). One flagship pedagogy — **on-the-fly game generation** — done excellently. Worked-example as silent fallback for concepts that resist game-ification.

**v1 is not:** auth, payments, mobile, multimedia, fine-tuning, multi-goal juggling, full synthetic-learner simulation for learning-gain.

---

## Repo map

```
adaptive-learning-agent/
├── backend/
│   ├── agents/          Intent, Curriculum, Router, Teaching/Game, Assessment, Memory
│   ├── schemas/         Pydantic models + SSE event union
│   ├── orchestration/   PTC harness, ToolRegistry, Anthropic client, SSE, logging
│   ├── retrieval/       Tavily wrapper + web_search Tool
│   ├── persistence/     SQLAlchemy async + LearnerState store
│   ├── api/             FastAPI /api/session/run (SSE)
│   └── main.py
├── frontend/            Next.js 14 + Tailwind, SSE consumer + sandboxed game iframe
├── evals/               7 evaluators, 21 golden cases, runner CLI
└── tests/               pytest, all agents/evaluators/persistence covered
```

## Deploy

Configs ship for the most common PaaS combos; pick one for the backend, Vercel for the frontend.

**Backend on Fly.io** (recommended):

```bash
fly launch --no-deploy           # claim the app name
fly secrets set ANTHROPIC_API_KEY=sk-ant-... TAVILY_API_KEY=tvly-...
fly secrets set ADAPTIVE_LEARNING_FRONTEND_ORIGIN=https://your-app.vercel.app
fly deploy                       # uses backend/Dockerfile + fly.toml
```

**Backend on Railway**: connect the GitHub repo; Railway auto-detects `railway.json` (Dockerfile build). Set the same env vars in the dashboard.

**Frontend on Vercel**: connect the GitHub repo, root = `frontend/`. Vercel auto-detects Next.js. Add one env var:

```
NEXT_PUBLIC_API_BASE_URL=https://<your-fly-or-railway-url>
```

After both are deployed, set `ADAPTIVE_LEARNING_FRONTEND_ORIGIN` on the backend to the Vercel URL so CORS allows it. Done — visit the Vercel URL.

**Cost-free demo**: append `?fixture=1` to the deployed frontend URL. Replays a recorded session locally without hitting the backend at all.

## Further reading

- [WRITEUP.md](WRITEUP.md) — design notes: PTC vs LangGraph, the MDA-grounded game agent, the prompt-cache gotcha, interactive assessment, roadmap
- [ARCHITECTURE.md](ARCHITECTURE.md) — full agent topology, principles, evaluator details
- [HANDOFF.md](HANDOFF.md) — original project brief and scope
- [backend/orchestration/ptc.py](backend/orchestration/ptc.py) — the tool-use loop
- [backend/api/session.py](backend/api/session.py) — the full session, end-to-end