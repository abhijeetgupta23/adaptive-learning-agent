# Resume here — handoff brief

**Last commit on `master`:** `b3e12de` — "Vision B + Haiku-first + observability + middleware + claim verifier" (pushed to `origin`).
**Branch state:** clean.
**Tests:** 216 pass · ruff clean · frontend tsc clean.

---

## What's already done (don't redo)

1. **Vision B locked in.** Game = flagship, Worked Example = silent fallback. The 5 stub pedagogies (Analogy, Visual, Socratic, Feynman, Retrieval) are removed from the registry, router prompt, README, ARCHITECTURE, and frontend type union. `TeachingMethod` enum still has all 7 values for back-compat with serialized state; only `game` and `worked_example` are routable.
2. **Haiku-first across all 6 agents.** Validated by a live eval: 3/3 game-generation cases PASS, mean score 0.79, mean $0.108/game, p50 62s. Sonnet remains the fallback the eval would gate to if Haiku slipped. See `evals/game_haiku_only_report.json`.
3. **Eval runner surfaces per-evaluator latency p50/p99 + per-case cost.** JSON shape is now `{summary: {per_evaluator, total}, results}`.
4. **CircuitBreaker on the Game agent** — trips on 3 consecutive `GameAgentError`s OR $0.50 cumulative game-agent spend; sticky; falls back to `worked_example` for the rest of the session. New `circuit_breaker_tripped` SSE event.
5. **Second-axis claim verifier** on `game_quality`: LLM judge ∩ keyword-regex on visible HTML text. Quorum gating with two thresholds at top of `evals/evaluators/game_quality.py`.
6. **/ops/dashboard observability endpoint** — JSON + dark-theme HTML. Backed by new `session_runs` SQLite table. `CallTimer` + `latency_ms` plumbed through every agent call.
7. **README reframed around the Civ-3 thesis** with academic citations (Squire & Barab, Plass 2015 meta-analysis, Bjork & Bjork, MIT attention-span data, MDA framework).
8. **Asyncio fixture rewritten** Vision-B coherent + Haiku-recalibrated. Cost trajectory now $0.193 vs old $0.516.

---

## Pending: Phase B Graph RAG (paused mid-build)

Goal: a universal **game-design knowledge graph** (Graphiti + Kuzu) that augments the Game agent's prompt with cited mechanics, principles, and misconception patterns at generation time. This is NOT a per-domain concept graph — the graph holds *transferable game-design intelligence* so the system stays domain-general.

### Schema (finalized)

Soft-schema, Vester-style — Pydantic node types constrain extraction; edge labels are emergent.

| Node type | Purpose |
|---|---|
| `Mechanic` | drag_drop, matching, simulation, spatial_arrange, button_array, canvas, role_play, puzzle, prediction_reveal |
| `ConceptType` | relational, procedural, conceptual, definitional, spatial, causal, classificatory |
| `Principle` | desirable_difficulty, immediate_feedback, scaffolded_complexity, achievement, etc. |
| `FeedbackPattern` | reveal_truth_on_error, correct_in_place, score_only, narrative_consequence |
| `MisconceptionPattern` | **universal** patterns — reversal_error, overgeneralization, fragmented_application, equality_vs_assignment |
| `Source` | citation: title, authors, year, url |

### What's seeded already (in repo)

`data/game_design_corpus/` has 5 of 7 distilled markdown summaries committed:
- `mda_framework.md` (Hunicke, LeBlanc, Zubek 2004)
- `plass_foundations.md` (Plass, Homer, Kinzer 2015)
- `gee_36_principles.md` (Gee 2003)
- `bjork_desirable_difficulties.md` (Bjork & Bjork 2011)
- `sitzmann_meta_analysis.md` (Sitzmann 2011)

### Still to write (do this first when you resume)

- `qian_lehman_programming_misconceptions.md` — Qian & Lehman 2017, ACM TOCE 18(1). Distill: variables-remain-linked (= reversal_error), `score = score - 1` read as equality not assignment, natural-language interference, content-level vs epistemic-level misconceptions. ~250-400 words.
- `perkins_simmons_patterns_of_misunderstanding.md` — Perkins & Simmons 1988, Review of Educational Research. Distill: integrative model of misconceptions across science, math, and programming; content/problem-solving/epistemic/inquiry levels; cross-domain pattern transfer. ~250-400 words.

### Blockers when Phase B agent was paused

1. **`uv add graphiti-core kuzu` was denied** in the agent's sandbox. Resolve by either:
   - Running locally yourself: `uv add graphiti-core kuzu` from repo root.
   - Or letting the new agent edit `pyproject.toml` directly and then running `uv lock && uv sync`.
2. **Graphiti needs an embedding model.** Anthropic has no embeddings API, so Graphiti uses Gemini (per Vester's pattern in `C:/Users/abhij/Documents/VestorAI/Vester_DeepAgents_PTC_Crypto_Stocks/vester_da_ptc/user_memory_graph.py`). Will need `GOOGLE_API_KEY` in `.env`. Make this optional — graph should degrade to no-op-embedder if key missing.

### Phase B build plan (resume from here)

1. Write the 2 missing corpus files (above).
2. Add deps to `pyproject.toml` and lock.
3. `backend/schemas/concept_graph.py` — Pydantic node models from the table above.
4. `backend/persistence/concept_graph.py` — `ConceptGraph` class wrapping Graphiti+Kuzu, methods: `add_episode(text, source)` and `query_for_game(concept_type) -> GraphContext`. DB file at `data/concept_graph/graph.kuzu` (already gitignored).
5. `backend/agents/graph_extractor.py` — walks the corpus, calls `add_episode` per chunk (~500 words). Uses Haiku.
6. `backend/agents/concept_classifier.py` — tiny Haiku call: `LearningUnit` → one of the 7 `ConceptType.kind` values.
7. `backend/agents/teaching/game.py` — new `await build_game_design_context(unit, graph, client) -> str`. Inject into the **user message** (not system prompt — preserves cache).
8. `scripts/bootstrap_concept_graph.py` — runnable bootstrap. Prints node/edge counts per source. **Costs ~$0.12 to run.**
9. Tests in `tests/test_concept_graph.py` (~150 LOC). Mock Graphiti if real install is heavy.
10. README addition: "Concept graph (game-design knowledge)" subsection citing the 7 sources.

### Critical constraints (don't violate)

- **Vision B**: graph informs game-coding decisions only. No new teaching methods.
- **Haiku-first**: every new LLM call uses `claude-haiku-4-5-20251001`.
- **Optional everywhere**: every read path degrades gracefully if graph is empty/unavailable. Game agent must still work without it.
- **Soft edge schema**: don't enumerate edge types — let Graphiti's emergent labels stand.
- **Don't break 216 tests, ruff, tsc.**

---

## Phase A (REMEMBER — do after Phase B works)

Full PDF extraction of all 7 verified sources to expand the corpus from short distilled summaries to ~50-100 nodes / ~150-300 edges. Sources + verified URLs are in the prior session transcript (search "verified, free PDFs" or check the recent web searches). Re-run bootstrap script after.

---

## Other parked work

- **Session-summary memory rollup** — after each session, a Haiku consolidator writes a 200-token "what you learned" string to `LearnerState`. Next session's Intent agent reads it. Mirrors Vester's `autoDream` consolidation.
- **5 utility skills** (analogy_finder, prerequisite_checker, assessment_question_generator, code_sandbox, citation_lookup). Brings registry from 2 → 7 real skills. Some of these become more useful once Graph RAG is in (`analogy_finder` reads the graph).

---

## Vester deep-audit summary (for context)

Vester's 60% of agent-engineering surface is already mirrored here: PTC, schemas, evals, streaming, cost tracking, A/B branch comparison, skill registry with tiers, middleware (now with CircuitBreaker), observability (now with /ops/dashboard).

The four still-missing patterns I identified:
1. **Graph memory** (Graphiti+Kuzu) — covered by Phase B above
2. **Memory consolidation** (autoDream-style) — covered by Session-summary above
3. **Cross-architecture benchmarks** — partially covered by `evals/game_model_compare.py`; could extend
4. **Multiple-source claim verification** — partially covered by the second-axis verifier; could add a 3rd axis (e.g., a different model as the 3rd judge)

---

## Quickstart commands

```bash
# from repo root
uv venv && uv pip install -e ".[dev]"   # first-time setup
uv run pytest --no-header -q             # should show: 216 passed
uv run ruff check .                      # should show: All checks passed!

# run the live Haiku game-gen eval (costs ~$0.32)
uv run python -m evals.game_model_compare --models claude-haiku-4-5-20251001 \
  --output evals/game_haiku_only_report.json

# frontend
cd frontend && npm install && npm run dev   # http://localhost:3000?fixture=1
```

For the new browser session: clone the repo (commit `b3e12de` is on `master`), read this file, then start Phase B from the build plan above.
