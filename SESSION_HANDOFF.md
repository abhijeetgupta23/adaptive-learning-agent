# Session Handoff — adaptive-learning-agent

> Successor doc to [HANDOFF.md](HANDOFF.md) (the original project brief). This captures
> the in-flight state of the build so the next agent can pick up without re-deriving context.

**Last session:** 2026-04-26 (built Tasks 1–10, wired prompt caching, verified end-to-end with real LLMs).

---

## TL;DR

Tasks 1–10 from [HANDOFF.md §9](HANDOFF.md#9-first-task-breakdown-for-claude-code) are substantially complete. **125/125 pytest pass, ruff clean, frontend builds.** Backend boots, SSE endpoint streamed a real curriculum (Opus + Tavily, 6 grounded units, ~30s) successfully. Three deploy/UI things remain (see §6).

Nothing is committed yet. Single `master` branch, no commits since project init.

---

## 1. Status by task

| Task | Status | Notes |
|------|--------|-------|
| 1. Scaffold | done | pre-existing |
| 2. Schemas | done | [backend/schemas/](backend/schemas/) — Goal, Curriculum, LearnerState, TeachingPlan, Assessment, SSE event union |
| 3. Orchestration | done | PTC harness, ToolRegistry, Anthropic wrapper, SSE encoder, JSON logger |
| 4. Intent Agent + eval | done | + 5 golden cases |
| 5. Curriculum Agent + Grounding + Coherence evals | done | + 5 golden cases |
| 6. Pedagogy Router + Game Agent + Game Quality eval | done | game has hard gates (playability + no-fetch) |
| 7. Assessment + Memory + persistence | done | SQLAlchemy async + SQLite |
| 8. Session loop + UI | done | FastAPI SSE endpoint, Next.js demo page renders every event type |
| 9. Eval harness | done | runner CLI, 23 golden cases (target was 30) |
| 10. Polish | partial | ARCHITECTURE.md and README updated; demo video + deploys not done |

---

## 2. What's verified live

Real-world tested in this session:

- ✅ Backend boots via `lifespan` (loads `.env`, creates SQLite, instantiates client+search tool)
- ✅ `GET /health` returns ok
- ✅ `POST /api/session/run` streams SSE: `intent_update` → `curriculum_ready` → per-unit (`unit_started`, `teaching_chunk`, `assessment_result`) → `complete`
- ✅ Curriculum quality on inspection: 6 units with real grounding URLs (codinghorror, sqlshack, w3schools, datalemur, etc.)
- ✅ Error path: missing API keys produce a clean `ErrorEvent`, not a 500
- ✅ Frontend `npm run build` succeeds
- ✅ Eval runner CLI prints summary on Windows (after fixing Unicode → `[PASS]`/`[FAIL]`)

Not yet verified live:
- 🔲 **Playwright UI drive** — MCP server was registered ([§6](#6-pending-follow-ups)) but Claude Code needs a restart to activate it
- 🔲 Frontend rendering of a live game iframe (only structurally tested via mocks)
- 🔲 Game Agent firing in a real session — first live curriculum picked visual/worked_example/socratic, not game. Pick a "game-friendly" goal (e.g., binary search visualization, Game of Life, negotiation strategy) to exercise it.

---

## 3. Critical files

```
backend/
  agents/
    _parsing.py          shared structured-output helpers (extract_text, parse_json_payload)
    intent.py            Intent agent — discriminated ask/complete union
    curriculum.py        Curriculum agent — uses run_ptc with web_search tool
    pedagogy_router.py   Rules-first router with LLM fallback
    teaching/game.py     Flagship Game agent — outputs GameSpec with self-contained HTML
    assessment.py        Pass/needs_reinforcement/needs_different_approach verdicts
    memory.py            Pure state-update fn (apply_assessment)
  orchestration/
    ptc.py               run_ptc — the tool-use loop (auto-wraps system w/ cache_control)
    cache.py             cached_system / normalize_system helpers (NEW this session)
    sse.py               encode_sse_event / encode_sse_stream
    tools.py             ToolRegistry
    anthropic_client.py  get_anthropic_client + log_messages_call
    logging.py           JSONFormatter + ContextVar-based request_id
  api/session.py         FastAPI /api/session/run — orchestrates the full session loop
  persistence/           SQLAlchemy async (LearnerStateRow), in-memory + file-backed
  retrieval/             Tavily wrapper + web_search Tool builder
  main.py                lifespan: load_dotenv, engine, client, search tool, CORS, router

frontend/
  src/app/page.tsx       demo UI: Start session button, EventCard renderers, sandboxed iframe for games
  src/lib/events.ts      TypeScript mirror of backend SSE union + readSseStream() async generator
  next.config.mjs        proxies /api/* to localhost:8000 (or NEXT_PUBLIC_API_BASE_URL)

evals/
  evaluators/            7 evaluators (intent_fidelity, curriculum_coherence, grounding,
                         pedagogy_fit, game_quality, assessment_accuracy, learning_gain)
  golden/                23 golden cases across 5 directories (intent, curriculum,
                         pedagogy_fit, assessment, learning_gain)
  runner.py              CLI: `python -m evals.runner --all` or `--evaluator NAME`

tests/
  _fakes.py              FakeAnthropic + text_response/tool_use_response helpers
  test_*.py              unit + integration tests, all green (125/125)
```

---

## 4. Decisions made this session (not in HANDOFF.md)

- **HANDOFF.md "no LangGraph" rule softened.** Now reads "prefer hand-rolled PTC for v1; hybrid is fine if loop plumbing hurts." See [HANDOFF.md:20](HANDOFF.md#L20) and [HANDOFF.md:260](HANDOFF.md#L260).
- **Prompt caching wired into all 8 LLM call sites** ([backend/orchestration/cache.py](backend/orchestration/cache.py)). System prompts get `cache_control: ephemeral`. Biggest win is the curriculum PTC loop where the same system prompt is replayed every iteration.
- **`load_dotenv()` added to [backend/main.py](backend/main.py)** so `.env` works without manual export.
- **Runner output uses `[PASS]`/`[FAIL]`** not Unicode ✓/✗ — Windows cp1252 codepage couldn't encode the symbols.
- **Session endpoint guards against missing keys** with a clean `ErrorEvent` instead of `AttributeError`.
- **23 golden cases shipped vs 30 target.** Concentrated gap: pedagogy_fit / assessment / learning_gain breadth. Adding 7 more cases would hit the target.

---

## 5. Cost / caching status

- **All system prompts cached** (ephemeral, 5-min TTL).
- One real curriculum build cost ~$0.50–$1.00 (Opus, ~25–50K input + ~1.5–2.5K output, no cache hits since it was the first call).
- Subsequent runs within 5 minutes: input tokens at the cached prefix drop to 10% of base ($1.50/Mtok instead of $15/Mtok). Expected savings on PTC inner-loop iterations: ~70%.
- **Cost knobs not yet pulled:**
  - Curriculum could swap Opus→Sonnet (5× cheaper, probably acceptable)
  - `max_iterations` is 10; rarely needs more than 5 in practice
  - Tool definitions could also have `cache_control` (small win, easy to add)

To get exact token counts on a live run: change uvicorn to `--log-level info`. The PTC loop emits `ptc_iteration` JSON logs with `input_tokens` and `output_tokens` per call.

---

## 6. Pending follow-ups

**Playwright UI drive (in flight):**
- MCP server `playwright` was added to local-scope config: `claude mcp add playwright -- npx @playwright/mcp@latest`
- **Action: restart Claude Code** so the server registers. Then ToolSearch for `browser_navigate`/`browser_click`/etc. and a successor agent can drive the demo flow end-to-end.

**Game Agent live exercise:**
- Pick a goal where games are the natural fit (Conway's Game of Life, hash collision visualization, rock-paper-scissors strategy, binary search). The router will then route to game pedagogy, the Game Agent will generate HTML, and the frontend will render it in the sandboxed iframe.

**Security:**
- 🔴 **Both API keys leaked.** They were pasted into `.env.example` (which conventionally gets committed) and now exist in this session's transcript. **Rotate both before any push.** I restored `.env.example` to placeholder values and put working keys in `.env` (gitignored).

**Task 10 leftovers:**
- 30-second demo GIF / mp4 (manual screen capture)
- Deploy backend to Fly.io, frontend to Vercel
- 7 more golden cases to hit 30

**Cleanup:**
- Nothing committed yet — the whole build is uncommitted on `master`. Ready to stage when the user gives a signal. Suggested split: per-task commits (~10 commits) for clean history, or a single squash if speed matters more.

**Known minor issues:**
- Smoke test creates `./adaptive_learning.db` as a side effect of the lifespan. It's gitignored. Could swap to `IN_MEMORY_DB_URL` via env var in the test fixture if it gets annoying.
- The frontend's `DEMO_GOAL` is hardcoded SQL joins. To exercise other domains/goals, edit [frontend/src/app/page.tsx](frontend/src/app/page.tsx) or add a goal-input form.

---

## 7. How to resume

```bash
# from repo root
uv run pytest -q                              # expect 125 passed
uv run ruff check backend tests evals          # expect "All checks passed!"
uv run python -m evals.runner --evaluator pedagogy   # deterministic, no keys needed

# live (needs ANTHROPIC_API_KEY + TAVILY_API_KEY in .env):
uv run uvicorn backend.main:app --reload --port 8000
# in another terminal:
cd frontend && npm run dev
# open http://localhost:3000 → click "Start session"
```

To verify SSE without a browser:
```bash
curl -N -X POST http://127.0.0.1:8000/api/session/run \
  -H "Content-Type: application/json" \
  -d '{"goal":{"domain":"databases","sub_goal":"INNER vs LEFT joins","current_level":"beginner","desired_depth":"working","time_budget_hours":2.0,"preferred_modalities":["visual"]},"user_id":"demo"}'
```

---

## 8. Memory files written this session

Persisted under `C:\Users\abhij\.claude\projects\<project>\memory\`:
- `user_role.md` — author context; this is a personal POW project
- `feedback_pacing.md` — when working a pre-approved task list, roll through; don't gate each task with "proceed?"

A successor agent that reads `MEMORY.md` will see these and apply them.
