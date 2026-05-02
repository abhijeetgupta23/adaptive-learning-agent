# Claude Code Handoff: `adaptive-learning-agent`

## 1. Project Summary

A personal POW (proof-of-work) project. An adaptive learning agent that helps a user learn anything — the user states a goal, the agent constructs a path, teaches each concept using the pedagogy best suited to it (games, analogies, visuals, Socratic dialogue, retrieval practice, etc.), and assesses understanding before advancing.

**Thesis:** In the AI era, depth in one skill is brittle. Range and the ability to reskill quickly are the durable assets. This agent is infrastructure for that ability — learn anything, fast, joyfully.

Built in personal time on personal hardware with a personal Anthropic API key.

**Repo name:** `adaptive-learning-agent` (will rebrand later if it becomes a startup).

---

## 2. Architectural Pattern

Core patterns:

- **Multi-agent chatbot system.** A small society of specialized agents orchestrated together, not a single monolithic prompt.
- **Programmatic tool calling (PTC).** Hand-rolled orchestration with explicit tool definitions — preserves observability and evaluation precision. Framework use (LangGraph etc.) is a judgment call, not banned; see Guardrails §10.
- **Schema-grounded generation.** Every agent output is grounded in a defined schema. No free-form LLM output that downstream components have to re-parse.
- **SSE streaming parser.** Stream structured events to the frontend, parse them on the client for progressive rendering.
- **Evaluation harness from day one.** Not bolted on later. Build the eval runner and golden dataset alongside the agent.

---

## 3. Agent Topology

Multi-agent topology specialized for learning.

### Intent Agent
- **Input:** user's stated learning goal (free-form natural language)
- **Job:** interview the user to extract a structured `LearningGoal` — domain, specific sub-goal, current knowledge level, desired depth (cocktail / working / expert), time budget, preferred modalities (visual / verbal / interactive)
- **Output schema:** `LearningGoal` (strict JSON)
- **Notes:** Should be conversational, not a form. Probes only the minimum needed. Users struggle to articulate their own goals — this agent helps them.

### Curriculum Agent
- **Input:** `LearningGoal`
- **Job:** research the domain (via web search + retrieval), construct an ordered list of `LearningUnits` that progress the user from current state to goal. Each unit has: objective, prerequisite check, recommended pedagogy tag, grounding sources.
- **Output schema:** `Curriculum` (ordered list of `LearningUnit` objects)
- **Notes:** This is where grounding matters most. Every claim in a unit should trace to a retrievable source. Hallucinated curricula are a failure mode.

### Pedagogy Router
- **Input:** a `LearningUnit` + current `LearnerState`
- **Job:** decide which teaching method(s) to use for this unit. Options from the toolkit: worked example, analogy, visual diagram, game / simulation, Socratic dialogue, Feynman technique (explain-back), retrieval practice, productive failure, storytelling.
- **Output schema:** `TeachingPlan` (sequence of method calls with parameters)
- **Notes:** This is the distinctive IP. The router is what makes the agent *adaptive* rather than one-trick.

### Teaching Agents (one per method)
- **Worked-Example Agent** — steps through a concrete example with fading.
- **Analogy Agent** — constructs a mapping from new concept to something the user already knows. Requires LearnerState to know what the user knows.
- **Visual Agent** — generates diagrams, flowcharts, or annotated images. Dual-coding principle: pair every verbal explanation with a visual when possible.
- **Game Agent** — generates a playable micro-game (text-based or simple HTML/JS) that encodes the concept as an interaction. *This is the flagship method for v1.*
- **Socratic Agent** — never asserts, only asks. Leads user to construct the understanding.
- **Feynman Agent** — user explains back, agent plays confused student, probes for gaps.
- **Retrieval Agent** — active recall via questions. Spaced per session.

### Assessment Agent
- **Input:** user's performance in a teaching interaction
- **Job:** decide whether the user has demonstrated understanding sufficient to advance. Not just "did they get the right answer" but "does their reasoning indicate they actually understand."
- **Output schema:** `AssessmentResult` (pass / needs-reinforcement / needs-different-approach + diagnostic notes)
- **Notes:** Hardest eval problem in the system. Graders are LLMs; grade the graders with a holdout set.

### Memory / State Agent
- **Input:** all events in the session + historical state
- **Job:** maintain `LearnerState` — what the user knows, what they struggled with, what analogies land for them, what pedagogies worked.
- **Output schema:** `LearnerState` (persisted across sessions)
- **Notes:** Three-layer memory: session-local memory, user-level memory (persistent), shared content memory (curated concept graph, populated offline).

---

## 4. Session Flow (the v1 killer loop)

```
1. User opens app. State: "What do you want to learn?"
2. Intent Agent interviews → LearningGoal locked.
3. Curriculum Agent researches + plans → Curriculum displayed. User can edit.
4. Session begins on Unit 1.
5. For each Unit:
   a. Pedagogy Router picks TeachingPlan for this unit.
   b. Teaching Agent(s) execute the plan (stream to UI via SSE).
   c. User interacts (plays game, explains back, answers retrieval questions, etc).
   d. Assessment Agent judges.
   e. If pass → advance. If needs-reinforcement → retry with adjusted plan. If needs-different-approach → route to different Teaching Agent.
6. End of session: Memory Agent persists LearnerState. Next session resumes from LearnerState.
```

---

## 5. Evaluation Harness (build alongside, not after)

Seven evaluators for v1:

1. **Intent Fidelity** — did Intent Agent correctly extract the user's goal? (LLM judge vs. human-labeled gold set)
2. **Curriculum Coherence** — is the curriculum logically ordered and complete for the goal? (LLM judge + rubric)
3. **Grounding** — every factual claim in curriculum/teaching traces to a retrievable source (automated check: retrieve claim's cited source and verify support)
4. **Pedagogy Fit** — did the router pick a reasonable method for this unit? (LLM judge vs. gold set of (unit, appropriate methods) pairs)
5. **Game Quality** — if a game was generated, is it playable, solvable, and on-concept? (multi-dim rubric + LLM judge)
6. **Assessment Accuracy** — does the Assessment Agent's pass/fail judgment match a human grader on a gold set?
7. **End-to-End Learning Gain** — synthetic learner with known baseline runs through a unit; measure knowledge delta pre/post via a structured quiz.

**Golden dataset target for v1:** 30 test cases across 3 domains (see §7).

---

## 6. Tech Stack

- **Backend:** Python 3.11+, FastAPI, SSE for streaming.
- **Model:** Claude (Sonnet for teaching agents, Opus for router/curriculum where reasoning depth matters). Single provider for now; abstract the client so swapping is trivial.
- **Retrieval:** start with Tavily or web search for open-domain research. Add a local vector store (Chroma) later if needed.
- **Frontend:** Next.js 14 + TypeScript + Tailwind. Minimal — chat interface, curriculum sidebar, session progress, game rendering area.
- **Persistence:** SQLite for v1. LearnerState + sessions + curricula. Upgrade to Postgres if it becomes a startup.
- **Eval runner:** standalone Python CLI that can run any evaluator against any test case.
- **Observability:** structured JSON logging, request IDs, per-agent timing. Honeycomb / Axiom later; local file logs for now.
- **Hosting:** Vercel for frontend, Fly.io or Render for backend. Personal accounts.

---

## 7. v1 Scope

**In scope for v1:**
- Single-user, local-first (login optional, no multi-tenancy yet)
- Three hardcoded test domains to prove the loop:
  1. **SQL joins** (technical, visual-friendly, game-friendly)
  2. **Basic Italian for travel** (language, spaced-repetition-friendly)
  3. **How neural networks learn** (conceptual, analogy-heavy)
- One killer pedagogy done excellently: **on-the-fly game generation**. Other methods (analogy, retrieval, Socratic) done competently as fallbacks.
- End-to-end flow working: intent → curriculum → session → assessment → state persistence.
- Eval harness with 30 test cases and 7 evaluators running.

**Out of scope for v1:**
- Multi-tenancy / user accounts / payments
- Multimedia (audio, video)
- Mobile app
- Multi-goal juggling per user
- Social features / sharing
- Custom models / fine-tuning

---

## 8. Directory Layout

```
adaptive-learning-agent/
├── README.md
├── HANDOFF.md                      # this file
├── ARCHITECTURE.md                 # expand on §3 with diagrams
├── backend/
│   ├── agents/
│   │   ├── intent.py
│   │   ├── curriculum.py
│   │   ├── pedagogy_router.py
│   │   ├── teaching/
│   │   │   ├── worked_example.py
│   │   │   ├── analogy.py
│   │   │   ├── visual.py
│   │   │   ├── game.py
│   │   │   ├── socratic.py
│   │   │   ├── feynman.py
│   │   │   └── retrieval.py
│   │   ├── assessment.py
│   │   └── memory.py
│   ├── schemas/                    # Pydantic models for all agent I/O
│   ├── orchestration/              # PTC harness, SSE, tool registry
│   ├── retrieval/                  # web search + grounding
│   ├── persistence/                # SQLite + SQLAlchemy
│   ├── api/                        # FastAPI routes
│   └── main.py
├── frontend/
│   ├── (Next.js app)
├── evals/
│   ├── runner.py
│   ├── evaluators/
│   │   ├── intent_fidelity.py
│   │   ├── curriculum_coherence.py
│   │   ├── grounding.py
│   │   ├── pedagogy_fit.py
│   │   ├── game_quality.py
│   │   ├── assessment_accuracy.py
│   │   └── learning_gain.py
│   └── golden/
│       ├── sql_joins/
│       ├── italian_travel/
│       └── neural_networks/
├── scripts/
│   └── (bootstrap, seeding, etc.)
├── pyproject.toml
└── .env.example
```

---

## 9. First-Task Breakdown (for Claude Code)

Work through these in order. Each task should result in a commit. Do not skip ahead.

### Task 1: Project scaffold
- Initialize the directory layout above
- `pyproject.toml` with deps: `anthropic`, `fastapi`, `uvicorn`, `pydantic`, `sqlalchemy`, `aiosqlite`, `httpx`, `sse-starlette`, `tavily-python`, `pytest`, `ruff`
- Next.js app initialized in `frontend/` with Tailwind
- `.env.example` with `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`
- `README.md` with setup instructions
- Verify: `uv run pytest` and `cd frontend && npm run dev` both succeed

### Task 2: Core schemas
- Define Pydantic models in `backend/schemas/`:
  - `LearningGoal`, `LearningUnit`, `Curriculum`
  - `LearnerState`
  - `TeachingPlan`, `TeachingMethod` enum
  - `AssessmentResult`
  - SSE event types
- Tests for schema validation

### Task 3: Orchestration primitives
- PTC harness: tool registry, tool execution loop, structured logging
- SSE streaming wrapper: emit typed events (`unit_started`, `teaching_chunk`, `assessment_result`, etc.)
- Anthropic client wrapper with retries and logging

### Task 4: Intent Agent (first vertical slice)
- Implement Intent Agent end-to-end
- Conversational flow that produces a valid `LearningGoal` from free-form input
- Golden test cases (5 examples)
- Eval: Intent Fidelity evaluator + its judge prompt

### Task 5: Curriculum Agent
- Web search integration (Tavily)
- Schema-grounded generation: Curriculum Agent produces a validated `Curriculum`
- Grounding pass: every unit cites at least one retrievable source
- Tests + 5 golden cases

### Task 6: Pedagogy Router + Game Teaching Agent (flagship method)
- Pedagogy Router: rules-first (deterministic) with LLM fallback for ambiguous cases
- Game Agent: generates a playable text-or-HTML micro-game for a given unit
- Game Quality evaluator
- Tests

### Task 7: Assessment + Memory Agents
- Assessment Agent: judge user responses with rubric
- Memory Agent: persist and load LearnerState across sessions

### Task 8: Session Loop
- FastAPI endpoint that runs the full flow and streams SSE
- Frontend: minimal UI that connects to SSE and renders each event type

### Task 9: Eval Harness
- CLI runner for all 7 evaluators
- Golden dataset with 30 cases across 3 domains
- `make eval` command that runs the full suite and outputs a report

### Task 10: Polish for POW
- Architecture diagram in ARCHITECTURE.md
- 30-second demo video (GIF or mp4) embedded in README
- Deploy backend (Fly.io) + frontend (Vercel)
- Clean commit history, meaningful messages

---

## 10. Guardrails for Claude Code

- **Prefer hand-rolled PTC for v1.** Keeps observability and eval precision tight while the surface area is small. A LangGraph+PTC hybrid is acceptable later if the hand-rolled loop plumbing starts to hurt. The real constraint is **data handoff via persistent namespace, not serialization across subagent boundaries**.
- **Every agent I/O is schema-validated.** No free-form dict passing between agents.
- **Every factual claim is grounded.** If a teaching agent states something factual, the source must be retrievable and cited.
- **Evals are first-class, not afterthought.** Each agent ships with its evaluator in the same PR.
- **Observability by default.** Structured logs with request IDs, per-agent latency and token counts.
- **No hardcoded secrets.** Everything via `.env`.
- **Commit granularity: one logical unit per commit.** No mega-commits. The commit history is part of the POW artifact.
- **Tests before demos.** If a feature isn't tested, it doesn't exist.

---

## 11. What "Done" Looks Like for v1

A reader landing on the repo sees:
1. A clear README with thesis, demo GIF, architecture overview, setup instructions.
2. A live-deployed demo they can use.
3. An ARCHITECTURE.md that walks through the multi-agent topology with diagrams.
4. An evals/ directory with a runnable harness and 30 test cases passing.
5. A clean commit history showing the project built up in logical increments.
6. Three domains working end-to-end (SQL joins, Italian travel, neural networks).

That's the POW artifact. Rebrand to Tagore later if it becomes a startup.
