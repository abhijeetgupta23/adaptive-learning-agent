# Adaptive Learning Agent — design notes

A multi-agent system that takes a learning goal, plans a curriculum grounded in
real sources, teaches each unit using a pedagogy chosen for the concept (worked
example, analogy, generated micro-game), and assesses understanding by
*conversation* — not multiple choice. Built as a personal proof-of-work piece;
the architecture mirrors patterns I ship at VesterAI but in a different domain
so the code is fully shareable.

The interesting parts aren't *that* it teaches — plenty of LLM tutors exist —
but the engineering choices behind it. This post is about those choices.

## Why hand-rolled PTC instead of LangGraph

LangGraph is a fine choice for orchestration. I deliberately wrote PTC
(Programmatic Tool Calling) by hand because the win-condition for this project
was *"can a senior reviewer read the orchestration code in fifteen minutes."*
A 130-line `run_ptc` loop hits that bar; a graph framework doesn't.

The trade I'm making explicit:

- **What I lose**: opinionated retry / cycle-detection / tool-result
  serialization, all of which LangGraph gives you for free.
- **What I gain**: every iteration of the loop is one inspectable function call
  with predictable side effects. When the game-agent JSON parser broke, I read
  the loop and fixed it in one place. When prompt caching needed a rolling
  breakpoint, I added it in 20 lines without touching graph state.

The cost of "not invented here" syndrome is the maintenance load if this grew
to 50 nodes. At 6 agents and ~300 lines of orchestration, it hasn't.

## Game generation as PTC tool-use, not one-shot

The first version of the game agent was one-shot: "here's a unit objective,
write me self-contained HTML/CSS/JS in a single ~15K-token JSON response."
About 30 % of generations had a JSON parse error, a JS syntax error, or got
truncated mid-`<script>`. I added retry-on-failure, but each retry was another
full one-shot — expensive and prone to the same failure mode.

The fix: game generation became its own PTC loop with six tools, ordered:

1. `commit_learning_link` — articulate the *learning mechanic* and the *game
   mechanic* that delivers it. Forced first call. This is the bridge that makes
   the game *educational*, not just fun. The serious-games literature (Plass,
   Kapp et al.) names this as the #1 failure mode of educational games.
2. `commit_core_loop` — the four-element loop: player action, system response,
   feedback, win condition. Per game-jam GDD practice, *one precise question*.
3. `commit_aesthetic` — what the player should *feel*, plus a `ui_archetype`
   from a five-element enum (`grid | drag_drop | button_array | canvas |
   list`). Constraining the layout space is what dropped variance.
4. `validate_html` — tag balance + archetype-specific element check.
5. `validate_js` — wraps `node --check` on the inline `<script>`.
6. `submit_game` — final commit; refuses to ship unless 1–5 all succeeded.

This decomposition is grounded in Hunicke et al.'s [MDA framework][mda]
(Mechanics → Dynamics → Aesthetics) plus the serious-games literature on the
learning-mechanic ↔ game-mechanic bridge. It's *more* expensive in tokens than
the one-shot approach (~$0.37 vs ~$0.07 per game on Sonnet 4.6) and much more
reliable. The validators run *inside* the loop, so a failed JS draft never
reaches the iframe.

This ports the Vester pattern that I described in the resume: *"single-loop
agent that writes against tools, with validation tools available so the loop
self-corrects."*

[mda]: https://users.cs.northwestern.edu/~hunicke/MDA.pdf

## The prompt-caching gotcha that bit us

Prompt caching is the obvious cost lever for any multi-iteration agent loop.
We wired `cache_control: ephemeral` on every system prompt — the resume claim
that *"caching is wired across all 8 LLM call sites"* is technically true.

It was also economically false for most agents.

Anthropic's docs spell out the constraint: **the cached prefix must be at
least 2,048 tokens for Sonnet 4.6** (4,096 for Opus / Haiku). Smaller prefixes
silently no-op — no error, no warning. Most of our agent system prompts are
400–800 tokens. Caching only kicked in when a PTC loop's growing conversation
crossed 2,048 tokens organically — usually around iteration 3 of the
curriculum or game agent.

Fix: rolling per-turn `cache_control` on the most recent user / tool-result
message, on top of the system-prompt breakpoint. Each iteration writes a new
cache entry at the latest message; the next iteration reads it via Anthropic's
20-block lookback window. ~20 lines of code in `run_ptc`. Game-agent cost
drops from $0.37 to a projected ~$0.10 with proper cache hits.

The deeper architectural answer is the **DeepAgents single-agent pattern**:
collapse the six per-agent prompts into one ~3,500-token "Adaptive Tutor"
prompt under a single PTC loop, with all current agents becoming skills the
tutor calls. The single big prompt is over the threshold from the very first
call, and amortizes across every session that uses the same agent. That
refactor is the next pass; this writeup describes its scope but I haven't
shipped it.

## Concurrency and prompt caching: the part I had to look up

A real question came up during the build: *if user A is mid-conversation and
user B starts a different one, does B "break" A's cache?*

The mechanics, after reading the docs:

- Cache entries are **content-addressed**, not session-addressed. A's prefix
  hashes to one slot, B's prefix hashes to a different slot. They coexist.
- Concurrent **reads** of the same entry (e.g. the shared system prompt) are
  safe. There's no lock, no contention.
- The one real constraint: *"a cache entry only becomes available after the
  first response begins."* So if A and B fire at the exact same instant with
  byte-identical prefixes, both write — neither gets the cached read until
  the first response begins streaming.

So at production scale, the system prompt cache amortizes across all users
(shared bytes → shared slot), but each user's conversation tail lives in its
own slot. The failure mode under multi-tenancy isn't *interference* — it's
**TTL expiry during HITL pauses**: if the learner stares at a question for six
minutes and the default 5-minute TTL expires, their conversation cache is gone
regardless of how many other users are active. Mitigation: Anthropic's 1-hour
cache product (2× write cost) for HITL flows where pauses are expected.

## Interactive assessment closes the adaptive promise

The original architecture had an `assess_understanding` agent in the codebase
but it wasn't wired into the session loop — assessment auto-passed every unit.
The "adaptive" in the project name was a false promise.

The fix is one of the smaller code changes in the project but the largest
product change: after each `teaching_chunk`, the assessment agent generates a
probing question (`generate_assessment_question`); the SSE stream emits an
`assessment_question` event; the session pauses on a queue waiting for the
learner's free-text answer; when it arrives, `assess_understanding` evaluates
the answer against the unit's objective and emits a verdict
(pass / needs\_reinforcement / needs\_different\_approach).

The verdict drives the same per-unit loop the click-button verdict drove
earlier:

- `pass` → advance to next unit
- `needs_reinforcement` → re-teach with the same method
- `needs_different_approach` → router excludes the current method, picks
  another, re-teaches

Capped at three attempts per unit so a stuck loop can't burn money forever.
The whole flow is ~80 lines added; closes the adaptive loop properly. The
click-button verdict path is preserved as a fallback (run without an
`assessment_queue` and it's auto-pass or click-driven).

## Eval harness, not an afterthought

Seven evaluators, twenty-three golden cases across three domains (databases,
italian travel, ML basics). The four interesting ones:

- `intent_fidelity` — does the intent agent extract goal fields the learner
  actually meant?
- `curriculum_coherence` — does the planned curriculum make sense for the
  goal? LLM-as-judge with axis-by-axis rubric.
- `grounding` — every unit must cite at least one retrievable URL; the
  evaluator HEADs each URL to verify reachability.
- `game_quality` — hard gates first (any interactive elements? no `fetch`,
  no external CDN? script parses?), then LLM-as-judge.

The point isn't that the evals are exhaustive — it's that the eval harness
exists and the rubric is *visible*. Reviewers in the agent-engineering hiring
guides I read consistently flag eval-harness presence as the #1 differentiator
between a candidate who's shipped and one who's done tutorials.

## What's deliberately not here

- **No multi-tier model routing** (Haiku for triage, Sonnet for substance) —
  Vester does this but the audience here is small enough that Sonnet
  everywhere is defensible at <$0.50/session.
- **No data-provenance auto-capture** — would matter at the analyst-tool scale
  Vester operates at; for a tutor app, every grounding source is in the
  curriculum_ready event already.
- **No deployment-side cache pre-warming**, **no scheduled eval Cloud Run
  job** — those are infra niceties that distract from the orchestration story
  the repo is trying to tell.

## Roadmap

In priority order, the items I'd ship next:

1. **DeepAgents single-agent pivot.** Single ~3,500-token system prompt
   (always above the cache threshold), all current agents become skills under
   one PTC loop. ~70 % cache hit rate is structural; matches Vester's
   production architecture honestly.
2. **Lazy per-unit teaching.** Currently the curriculum agent plans 6 units
   and the session teaches all 6. With verdict-driven flow, lazy generation
   means we only pay for content the learner actually consumes.
3. **Multi-tier model routing.** Haiku for intent, Sonnet for substance,
   reserve Opus for cases where the curriculum agent's grounding is
   ambiguous. ~30 % cost reduction at the system level.
4. **Recorded-session replay** as a first-class evaluation primitive.
   Capture a session's full SSE stream, replay against a different model or a
   modified prompt to A/B without re-paying.

## Numbers, current

- 189 backend tests, ruff clean
- 6 agents (intent, curriculum, pedagogy router, game, worked example,
  assessment) over ~3,200 lines of Python
- Frontend: Next.js 14 / React 18 / Tailwind — single-page split layout,
  curriculum stepper, focus-mode unit view, sandboxed iframe for games,
  custom markdown renderer for worked examples
- ~$0.50/session on Sonnet 4.6 with the rolling-cache fix; projected ~$0.15
  after the DeepAgents pivot
- Local-only by default; Dockerfile + fly.toml + railway.json + vercel.json
  shipped for deploy

## Repo map

- [`backend/orchestration/ptc.py`](backend/orchestration/ptc.py) — 130-line
  PTC loop, rolling cache, cost record_usage
- [`backend/agents/teaching/game_design.py`](backend/agents/teaching/game_design.py) —
  the six MDA-grounded game-design tools
- [`backend/agents/teaching/game_validators.py`](backend/agents/teaching/game_validators.py) —
  HTML tag-balancer + archetype-element check, JS validator wrapping
  `node --check`
- [`backend/api/session.py`](backend/api/session.py) — SSE orchestrator;
  HITL intent loop, per-unit interactive-assessment loop, three verdict
  branches with attempt cap
- [`backend/skills/registry.py`](backend/skills/registry.py) — generic
  `Skill` primitive for progressive token disclosure
- [`evals/`](evals/) — 7 evaluators, 23 golden cases, runnable via
  `python -m evals.runner --all`
- [`frontend/src/app/page.tsx`](frontend/src/app/page.tsx) — single-page
  React app; derived state, left-rail stepper, focus-mode main pane,
  fixture mode for cost-free UI iteration

## Trying it

```bash
# Backend
uv sync
uv run uvicorn backend.main:app --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev

# Open http://localhost:3000
# Or zero-cost replay of a recorded session: ?fixture=1
```

Required env: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`. Both have free tiers
generous enough to run several full sessions.

---

*Built personally outside of work hours. Patterns are the same as VesterAI's
production agent platform, but the code here is fully shareable. If you're
hiring for agent engineering, the [repo](.) is the artifact; this writeup is
the commentary.*
