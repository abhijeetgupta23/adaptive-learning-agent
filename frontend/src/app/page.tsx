"use client";

import { useMemo, useRef, useState } from "react";
import {
  CostUpdate,
  Curriculum,
  LearningGoal,
  SSEEvent,
  TeachingMethod,
  TeachingPlan,
  AssessmentResult,
  Verdict,
  readSseStream,
} from "@/lib/events";

const DEFAULT_GOAL_PROMPT =
  "how Python's asyncio event loop schedules coroutines: await suspension, the ready queue, and when control returns";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

interface FixtureFile {
  events: SSEEvent[];
  delays_ms?: Record<string, number>;
}

function getFixtureName(): string | null {
  if (typeof window === "undefined") return null;
  const v = new URLSearchParams(window.location.search).get("fixture");
  if (!v) return null;
  return v === "1" ? "asyncio_session" : v;
}

// ============================================================================
// Derived state
// ============================================================================

type UnitStatus =
  | "pending"
  | "active"
  | "done"
  | "needs_reinforcement"
  | "needs_different_approach";

interface DerivedUnit {
  id: string;
  index: number;
  objective: string;
  recommendedPedagogy: TeachingMethod;
  status: UnitStatus;
  attemptCount: number;
  awaitingVerdict: boolean;
  plan?: TeachingPlan;
  teachingMethod?: TeachingMethod;
  teachingContent?: string;
  assessment?: AssessmentResult;
}

interface DerivedSession {
  goal?: LearningGoal;
  pendingQuestion: string | null;
  intentHistory: { question: string; answer?: string }[];
  curriculum?: Curriculum;
  units: DerivedUnit[];
  activeUnitIdx: number; // -1 if none active
  complete: boolean;
  errorMessage?: string;
}

function deriveSession(
  events: SSEEvent[],
  pendingQuestion: string | null,
): DerivedSession {
  const intentAsks = events.filter((e) => e.event === "intent_ask");
  const intentUpdate = events.find(
    (e) => e.event === "intent_update" && (e as { goal?: unknown }).goal,
  );
  const curriculumEv = events.find((e) => e.event === "curriculum_ready");
  const completeEv = events.find((e) => e.event === "complete");
  const errorEv = events.find((e) => e.event === "error");

  const curriculum =
    curriculumEv && curriculumEv.event === "curriculum_ready"
      ? curriculumEv.curriculum
      : undefined;

  const units: DerivedUnit[] = [];
  if (curriculum) {
    curriculum.units.forEach((u, i) => {
      // A unit can have MULTIPLE teaching attempts (re-teach / different way).
      // Use the LATEST event of each kind for the displayed view. Type
      // predicates narrow the union so TS knows the per-event shape.
      type StartedEv = Extract<SSEEvent, { event: "unit_started" }>;
      type TeachEv = Extract<SSEEvent, { event: "teaching_chunk" }>;
      type AssessEv = Extract<SSEEvent, { event: "assessment_result" }>;
      const startedEvs = events.filter(
        (e): e is StartedEv => e.event === "unit_started" && e.unit_id === u.id,
      );
      const teachingEvs = events.filter(
        (e): e is TeachEv =>
          e.event === "teaching_chunk" && e.unit_id === u.id,
      );
      const assessEvs = events.filter(
        (e): e is AssessEv =>
          e.event === "assessment_result" && e.result.unit_id === u.id,
      );
      const lastStarted = startedEvs.at(-1);
      const lastTeaching = teachingEvs.at(-1);
      const lastAssess = assessEvs.at(-1);

      // Status logic:
      // - If any assessment is pass → done (terminal-positive).
      // - Else if a new unit_started has fired without a matching assessment
      //   yet → active (currently teaching this attempt).
      // - Else if last assessment is non-pass → reflect that verdict.
      // - Else if started but no assessment → active.
      // - Else pending.
      const anyPass = assessEvs.some((e) => e.result.verdict === "pass");
      let status: UnitStatus = "pending";
      if (anyPass) {
        status = "done";
      } else if (
        startedEvs.length > 0 &&
        startedEvs.length > assessEvs.length
      ) {
        status = "active";
      } else if (lastAssess) {
        const v = lastAssess.result.verdict;
        status =
          v === "pass"
            ? "done"
            : v === "needs_reinforcement"
              ? "needs_reinforcement"
              : "needs_different_approach";
      } else if (lastStarted) {
        status = "active";
      }

      units.push({
        id: u.id,
        index: i + 1,
        objective: u.objective,
        recommendedPedagogy: u.recommended_pedagogy,
        status,
        attemptCount: startedEvs.length,
        plan: lastStarted?.plan,
        teachingMethod: lastTeaching?.method,
        teachingContent: lastTeaching?.content,
        assessment: lastAssess?.result,
        // Awaiting verdict: latest teaching has rendered but no matching
        // assessment yet. Drives the verdict-button gate in the UI.
        awaitingVerdict: teachingEvs.length > assessEvs.length,
      });
    });
  }
  const activeUnitIdx = units.findIndex((u) => u.status === "active");

  return {
    goal:
      intentUpdate && intentUpdate.event === "intent_update"
        ? intentUpdate.goal ?? undefined
        : undefined,
    pendingQuestion,
    intentHistory: intentAsks
      .filter((e): e is Extract<SSEEvent, { event: "intent_ask" }> =>
        e.event === "intent_ask",
      )
      .map((e) => ({ question: e.question })),
    curriculum,
    units,
    activeUnitIdx,
    complete: !!completeEv,
    errorMessage:
      errorEv && errorEv.event === "error" ? errorEv.message : undefined,
  };
}

// ============================================================================
// Page
// ============================================================================

export default function Home() {
  const [goalText, setGoalText] = useState(DEFAULT_GOAL_PROMPT);
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [cost, setCost] = useState<CostUpdate | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [focusedIdx, setFocusedIdx] = useState<number | null>(null);
  // Per-unit verdict the user picked (visual; backend wiring is separate work).
  const [verdictByUnit, setVerdictByUnit] = useState<Record<string, Verdict>>(
    {},
  );

  const fixtureAnswerResolverRef = useRef<((v: string) => void) | null>(null);
  const fixtureVerdictResolverRef = useRef<((v: Verdict) => void) | null>(null);

  const session = useMemo(
    () => deriveSession(events, pendingQuestion),
    [events, pendingQuestion],
  );
  const effectiveFocus =
    focusedIdx !== null
      ? focusedIdx
      : session.activeUnitIdx >= 0
        ? session.activeUnitIdx
        : Math.max(0, session.units.length - 1);

  const submitAnswer = async (answer: string) => {
    if (!answer.trim()) return;
    setPendingQuestion(null);
    if (fixtureAnswerResolverRef.current) {
      fixtureAnswerResolverRef.current(answer);
      fixtureAnswerResolverRef.current = null;
      return;
    }
    if (!sessionId) return;
    try {
      const res = await fetch(`${API_BASE}/api/session/${sessionId}/respond`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} on /respond`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const recordVerdict = async (unitId: string, verdict: Verdict) => {
    setVerdictByUnit((prev) => ({ ...prev, [unitId]: verdict }));
    // Fixture mode: resolve the awaited promise so the replay loop synthesizes
    // an assessment event and continues. Real mode: POST /verdict so the
    // backend's per-unit loop branches on the choice (pass advances; reinforce
    // re-teaches the same method; different_approach excludes the method).
    if (fixtureVerdictResolverRef.current) {
      fixtureVerdictResolverRef.current(verdict);
      fixtureVerdictResolverRef.current = null;
      return;
    }
    if (!sessionId) return;
    try {
      const res = await fetch(`${API_BASE}/api/session/${sessionId}/verdict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} on /verdict`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const runFixtureSession = async (name: string) => {
    const res = await fetch(`/fixtures/${name}.json`);
    if (!res.ok) throw new Error(`fixture ${name} → HTTP ${res.status}`);
    const fixture: FixtureFile = await res.json();
    const delays = fixture.delays_ms ?? {};
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    // After each teaching_chunk for active unit we pause for a verdict click,
    // synthesize the assessment from it, then SKIP the next event in the
    // fixture if it's an assessment_result for the same unit. Lets the
    // verdict UX engage even though the fixture is a static recording.
    let skipNextAssessmentFor: string | null = null;

    for (const ev of fixture.events) {
      if (
        skipNextAssessmentFor &&
        ev.event === "assessment_result" &&
        ev.result.unit_id === skipNextAssessmentFor
      ) {
        skipNextAssessmentFor = null;
        continue;
      }
      if (ev.event === "session_started") {
        setSessionId(ev.session_id);
      } else if (ev.event === "cost_update") {
        setCost({
          total_usd: ev.total_usd,
          by_agent: ev.by_agent,
          cache_hit_rate: ev.cache_hit_rate,
          call_count: ev.call_count,
        });
      } else if (ev.event === "intent_ask") {
        setPendingQuestion(ev.question);
        setEvents((prev) => [...prev, ev]);
        await new Promise<string>((resolve) => {
          fixtureAnswerResolverRef.current = resolve;
        });
      } else if (ev.event === "teaching_chunk") {
        setEvents((prev) => [...prev, ev]);
        // Wait for user to click a verdict button.
        const verdict = await new Promise<Verdict>((resolve) => {
          fixtureVerdictResolverRef.current = resolve;
        });
        // Synthesize an assessment_result that mirrors the user's verdict,
        // and skip the fixture's pre-recorded one for this unit.
        const synthesized: SSEEvent = {
          event: "assessment_result",
          result: {
            unit_id: ev.unit_id,
            verdict,
            diagnostic_notes:
              "User verdict (fixture-replay synthesis — backend wiring runs in real sessions).",
            confidence: 1.0,
          },
        };
        setEvents((prev) => [...prev, synthesized]);
        skipNextAssessmentFor = ev.unit_id;
      } else {
        setEvents((prev) => [...prev, ev]);
        if (ev.event === "complete" || ev.event === "error") break;
      }
      await sleep(delays[ev.event] ?? 200);
    }
  };

  const runRealSession = async () => {
    const res = await fetch(`${API_BASE}/api/session/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        initial_message: goalText.trim(),
        user_id: "demo",
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    for await (const ev of readSseStream(res)) {
      if (ev.event === "session_started") {
        setSessionId(ev.session_id);
        continue;
      }
      if (ev.event === "cost_update") {
        setCost({
          total_usd: ev.total_usd,
          by_agent: ev.by_agent,
          cache_hit_rate: ev.cache_hit_rate,
          call_count: ev.call_count,
        });
        continue;
      }
      if (ev.event === "intent_ask") {
        setPendingQuestion(ev.question);
        setEvents((prev) => [...prev, ev]);
        continue;
      }
      setEvents((prev) => [...prev, ev]);
      if (ev.event === "complete" || ev.event === "error") break;
    }
  };

  const startSession = async () => {
    if (!goalText.trim()) return;
    setRunning(true);
    setEvents([]);
    setErrorMsg(null);
    setCost(null);
    setSessionId(null);
    setPendingQuestion(null);
    setFocusedIdx(null);
    setVerdictByUnit({});
    fixtureAnswerResolverRef.current = null;
    fixtureVerdictResolverRef.current = null;
    const fixtureName = getFixtureName();
    try {
      if (fixtureName) {
        await runFixtureSession(fixtureName);
      } else {
        await runRealSession();
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
      setPendingQuestion(null);
      fixtureAnswerResolverRef.current = null;
      fixtureVerdictResolverRef.current = null;
    }
  };

  const showSplit = session.units.length > 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <style>{ANIMATION_CSS}</style>
      <TopBar session={session} cost={cost} running={running} />

      {!showSplit && (
        <main className="mx-auto max-w-3xl px-6 pt-12 pb-24">
          {session.pendingQuestion ? (
            <IntentPanel
              question={session.pendingQuestion}
              onAnswer={submitAnswer}
            />
          ) : (
            <GoalForm
              value={goalText}
              onChange={setGoalText}
              running={running}
              onStart={startSession}
            />
          )}
          {errorMsg && <ErrorBanner msg={errorMsg} />}
          {running && !session.pendingQuestion && (
            <ThinkingShimmer label="Researching domain & planning curriculum…" />
          )}
        </main>
      )}

      {showSplit && (
        <main className="mx-auto max-w-7xl px-6 pt-6 pb-24">
          <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">
            <LeftRail
              units={session.units}
              activeIdx={session.activeUnitIdx}
              focusedIdx={effectiveFocus}
              onFocus={setFocusedIdx}
              goal={session.goal}
            />
            <MainPane
              session={session}
              focusedIdx={effectiveFocus}
              running={running}
              verdictByUnit={verdictByUnit}
              onVerdict={recordVerdict}
              onAnswer={submitAnswer}
            />
          </div>
          {errorMsg && <ErrorBanner msg={errorMsg} />}
        </main>
      )}
    </div>
  );
}

// ============================================================================
// CSS
// ============================================================================

const ANIMATION_CSS = `
  @keyframes cardIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  .card-in { animation: cardIn 0.35s ease-out both; }
  @keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
  .shimmer-bar { background: linear-gradient(90deg, rgba(56,189,248,0) 0%, rgba(56,189,248,0.35) 50%, rgba(56,189,248,0) 100%); background-size: 400px 100%; animation: shimmer 1.6s linear infinite; }
  @keyframes pulseRing { 0% { box-shadow: 0 0 0 0 rgba(56,189,248,0.55); } 70% { box-shadow: 0 0 0 12px rgba(56,189,248,0); } 100% { box-shadow: 0 0 0 0 rgba(56,189,248,0); } }
  .pulse-ring { animation: pulseRing 1.6s infinite; }
  @keyframes pulseDot { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .pulse-dot { animation: pulseDot 1.2s ease-in-out infinite; }
`;

// ============================================================================
// TopBar
// ============================================================================

function TopBar({
  session,
  cost,
  running,
}: {
  session: DerivedSession;
  cost: CostUpdate | null;
  running: boolean;
}) {
  const total = session.units.length;
  const done = session.units.filter((u) => u.status === "done").length;
  const phase = session.errorMessage
    ? "error"
    : session.complete
      ? "done"
      : !session.curriculum
        ? running
          ? "researching"
          : "idle"
        : `unit ${Math.min(done + 1, total)}/${total}`;
  const pct =
    total === 0 ? (running ? 6 : 0) : (done / Math.max(total, 1)) * 100;
  return (
    <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-6 py-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center text-sm font-bold shrink-0">
            A
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold tracking-tight">
              Adaptive Learning Agent
            </div>
            <div className="text-[11px] uppercase tracking-wider text-slate-500 truncate">
              {phase}
              {session.goal && (
                <span className="text-slate-600 normal-case font-normal tracking-normal">
                  {" · "}
                  <em className="text-slate-400">{session.goal.sub_goal}</em>
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4 shrink-0">
          {total > 0 && (
            <div className="hidden sm:flex items-center gap-2">
              <div className="relative h-1.5 w-32 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="absolute inset-y-0 left-0 bg-gradient-to-r from-sky-500 to-indigo-500 transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="text-[11px] tabular-nums text-slate-400">
                {done}/{total}
              </div>
            </div>
          )}
          {cost && <CostPill cost={cost} />}
        </div>
      </div>
    </header>
  );
}

function CostPill({ cost }: { cost: CostUpdate }) {
  const byAgent = Object.entries(cost.by_agent)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k}: $${v.toFixed(3)}`)
    .join(" · ");
  return (
    <div
      className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 font-mono text-[11px] text-slate-300"
      title={byAgent}
    >
      <span className="text-slate-500">spend</span>{" "}
      <span className="font-semibold text-emerald-400">
        ${cost.total_usd.toFixed(3)}
      </span>
      <span className="ml-2 text-slate-500">
        · {cost.call_count} calls · cache {(cost.cache_hit_rate * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// ============================================================================
// Pre-plan (idle / intent / researching)
// ============================================================================

function GoalForm({
  value,
  onChange,
  running,
  onStart,
}: {
  value: string;
  onChange: (v: string) => void;
  running: boolean;
  onStart: () => void;
}) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
      <label className="mb-2 block text-xs uppercase tracking-wider text-slate-500">
        What do you want to learn?
      </label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={running}
        rows={2}
        placeholder="e.g. how a B-tree handles inserts and splits"
        className="w-full resize-none rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-500/30 disabled:opacity-60"
      />
      <div className="mt-4 flex items-center justify-between">
        <div className="text-xs text-slate-500">
          The agent will research, plan a curriculum, and teach unit-by-unit.
        </div>
        <button
          onClick={onStart}
          disabled={running || !value.trim()}
          className={`rounded-lg px-5 py-2 text-sm font-semibold text-white transition ${
            running
              ? "bg-slate-700 cursor-not-allowed pulse-ring"
              : "bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 disabled:opacity-50"
          }`}
        >
          {running ? "Running…" : "Start session"}
        </button>
      </div>
    </section>
  );
}

function IntentPanel({
  question,
  onAnswer,
}: {
  question: string;
  onAnswer: (answer: string) => void;
}) {
  const [val, setVal] = useState("");
  const submit = () => {
    const v = val.trim();
    if (!v) return;
    onAnswer(v);
    setVal("");
  };
  return (
    <section className="rounded-2xl border border-sky-700/40 bg-sky-500/5 p-6 card-in">
      <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wider text-sky-400">
        <span className="h-1.5 w-1.5 rounded-full bg-sky-400 pulse-dot" />
        Intent agent — clarification
      </div>
      <p className="text-base text-slate-100 leading-relaxed mb-4">{question}</p>
      <div className="flex gap-2">
        <input
          type="text"
          value={val}
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="Type your answer…"
          autoFocus
          className="flex-1 rounded-lg border border-sky-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-500/30"
        />
        <button
          onClick={submit}
          disabled={!val.trim()}
          className="rounded-lg bg-gradient-to-r from-sky-500 to-indigo-600 px-5 py-2 text-sm font-semibold text-white disabled:opacity-50 hover:from-sky-400 hover:to-indigo-500"
        >
          Send
        </button>
      </div>
    </section>
  );
}

function ThinkingShimmer({ label }: { label: string }) {
  return (
    <div className="card-in mt-6 flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3 text-sm text-slate-400">
      <div className="h-2 w-2 animate-pulse rounded-full bg-sky-400" />
      <div className="flex-1">{label}</div>
      <div className="shimmer-bar h-1 w-32 rounded-full" />
    </div>
  );
}

function ErrorBanner({ msg }: { msg: string }) {
  return (
    <div className="mt-6 rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
      {msg}
    </div>
  );
}

// ============================================================================
// LeftRail
// ============================================================================

function LeftRail({
  units,
  activeIdx,
  focusedIdx,
  onFocus,
  goal,
}: {
  units: DerivedUnit[];
  activeIdx: number;
  focusedIdx: number;
  onFocus: (i: number | null) => void;
  goal?: LearningGoal;
}) {
  return (
    <aside className="lg:sticky lg:top-20 lg:self-start space-y-3">
      <div>
        <div className="text-[11px] uppercase tracking-wider text-slate-500 mb-1 px-1">
          Curriculum
        </div>
        {goal && (
          <div
            className="text-xs text-slate-400 line-clamp-2 italic px-1 mb-3"
            title={goal.sub_goal}
          >
            {goal.sub_goal}
          </div>
        )}
      </div>
      <div className="space-y-1.5">
        {units.map((u, i) => (
          <UnitListItem
            key={u.id}
            unit={u}
            isActive={i === activeIdx}
            isFocused={i === focusedIdx}
            onClick={() => onFocus(i)}
          />
        ))}
      </div>
    </aside>
  );
}

function UnitListItem({
  unit,
  isActive,
  isFocused,
  onClick,
}: {
  unit: DerivedUnit;
  isActive: boolean;
  isFocused: boolean;
  onClick: () => void;
}) {
  const ring =
    unit.status === "done"
      ? "border-emerald-500/40 bg-emerald-500/5"
      : unit.status === "active"
        ? "border-sky-500/60 bg-sky-500/10"
        : unit.status === "needs_reinforcement"
          ? "border-amber-500/40 bg-amber-500/5"
          : unit.status === "needs_different_approach"
            ? "border-rose-500/40 bg-rose-500/5"
            : "border-slate-800 bg-slate-900/40";
  const focusRing = isFocused
    ? "ring-2 ring-sky-500/40"
    : "hover:border-slate-700";
  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-lg border px-3 py-2.5 transition ${ring} ${focusRing}`}
    >
      <div className="flex items-start gap-3">
        <UnitStatusBadge status={unit.status} index={unit.index} active={isActive} />
        <div className="flex-1 min-w-0">
          <div className="text-xs leading-snug text-slate-200 line-clamp-2">
            {unit.objective}
          </div>
          <div className="mt-1.5">
            <PedagogyBadge method={unit.recommendedPedagogy} small />
          </div>
        </div>
      </div>
    </button>
  );
}

function UnitStatusBadge({
  status,
  index,
  active,
}: {
  status: UnitStatus;
  index: number;
  active: boolean;
}) {
  const ringCls =
    status === "done"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/50"
      : status === "active"
        ? "bg-sky-500/20 text-sky-300 border-sky-500/60"
        : status === "needs_reinforcement"
          ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
          : status === "needs_different_approach"
            ? "bg-rose-500/20 text-rose-300 border-rose-500/50"
            : "bg-slate-800/60 text-slate-500 border-slate-700";
  const content =
    status === "done" ? (
      "✓"
    ) : status === "needs_reinforcement" ? (
      "↺"
    ) : status === "needs_different_approach" ? (
      "↬"
    ) : (
      index
    );
  return (
    <div
      className={`shrink-0 h-7 w-7 rounded-full border flex items-center justify-center text-xs font-semibold ${ringCls} ${active ? "pulse-ring" : ""}`}
    >
      {content}
    </div>
  );
}

// ============================================================================
// MainPane
// ============================================================================

function MainPane({
  session,
  focusedIdx,
  running,
  verdictByUnit,
  onVerdict,
  onAnswer,
}: {
  session: DerivedSession;
  focusedIdx: number;
  running: boolean;
  verdictByUnit: Record<string, Verdict>;
  onVerdict: (unitId: string, verdict: Verdict) => void;
  onAnswer: (answer: string) => void;
}) {
  // HITL question takes priority over unit content
  if (session.pendingQuestion) {
    return (
      <div>
        <IntentPanel question={session.pendingQuestion} onAnswer={onAnswer} />
      </div>
    );
  }

  const unit = session.units[focusedIdx];
  if (!unit) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-6 text-sm text-slate-500">
        Pick a unit from the left.
      </div>
    );
  }

  const isActiveUnit = focusedIdx === session.activeUnitIdx;
  const showSummary = session.complete && unit.status === "done";

  return (
    <div className="space-y-4">
      <UnitHeader unit={unit} session={session} />
      {showSummary && session.complete && focusedIdx === session.units.length - 1 && (
        <SessionSummaryCallout session={session} />
      )}
      <UnitTeachingView unit={unit} running={running && isActiveUnit} />
      {unit.awaitingVerdict ? (
        <VerdictButtons
          unitId={unit.id}
          chosen={verdictByUnit[unit.id]}
          onPick={(v) => onVerdict(unit.id, v)}
        />
      ) : (
        unit.assessment && (
          <UnitAssessmentView
            assessment={unit.assessment}
            override={verdictByUnit[unit.id]}
          />
        )
      )}
    </div>
  );
}

function UnitHeader({
  unit,
  session,
}: {
  unit: DerivedUnit;
  session: DerivedSession;
}) {
  const teachingMethod = unit.teachingMethod ?? unit.recommendedPedagogy;
  return (
    <div>
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-slate-500">
        <span>Unit {unit.index} of {session.units.length}</span>
        <span>·</span>
        <PedagogyBadge method={teachingMethod} />
        {unit.attemptCount > 1 && (
          <>
            <span>·</span>
            <span className="text-amber-400">attempt {unit.attemptCount}</span>
          </>
        )}
      </div>
      <h2 className="mt-1 text-xl font-semibold leading-snug text-slate-50">
        {unit.objective}
      </h2>
    </div>
  );
}

function UnitTeachingView({
  unit,
  running,
}: {
  unit: DerivedUnit;
  running: boolean;
}) {
  if (!unit.teachingContent) {
    if (running && unit.status === "active") {
      const m = unit.plan?.methods[0]?.method;
      const label =
        m === "game"
          ? "Generating interactive game…"
          : m === "worked_example"
            ? "Writing worked example…"
            : m
              ? `Teaching via ${m}…`
              : "Preparing teaching content…";
      return <ThinkingShimmer label={label} />;
    }
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-6 text-sm text-slate-500">
        No teaching content yet.
      </div>
    );
  }
  const method = unit.teachingMethod ?? "retrieval";
  return (
    <section
      className={`rounded-xl border border-slate-800 bg-slate-900/40 ${
        method === "game" ? "p-3" : "p-5"
      }`}
    >
      <TeachingContent method={method} content={unit.teachingContent} />
    </section>
  );
}

function UnitAssessmentView({
  assessment,
  override,
}: {
  assessment: AssessmentResult;
  override?: Verdict;
}) {
  const v = override ?? assessment.verdict;
  const tone =
    v === "pass"
      ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-300"
      : v === "needs_reinforcement"
        ? "border-amber-500/30 bg-amber-500/5 text-amber-300"
        : "border-rose-500/30 bg-rose-500/5 text-rose-300";
  return (
    <div className={`rounded-xl border p-3 text-sm ${tone}`}>
      <span className="text-[11px] uppercase tracking-wider opacity-70">
        Assessment ·{" "}
      </span>
      <strong>{v}</strong>
      {override && (
        <span className="ml-2 text-xs opacity-70">(your verdict)</span>
      )}
      <div className="mt-1 text-xs text-slate-400">
        {assessment.diagnostic_notes}
      </div>
    </div>
  );
}

function VerdictButtons({
  unitId,
  chosen,
  onPick,
}: {
  unitId: string;
  chosen?: Verdict;
  onPick: (v: Verdict) => void;
}) {
  void unitId;
  const btn = (
    v: Verdict,
    label: string,
    tone: string,
  ) => {
    const isChosen = chosen === v;
    return (
      <button
        onClick={() => onPick(v)}
        disabled={!!chosen && !isChosen}
        className={`flex-1 rounded-xl border px-4 py-3 text-sm font-medium transition ${
          isChosen
            ? `${tone} ring-2 ring-offset-2 ring-offset-slate-950`
            : `border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800/60 ${
                !!chosen ? "opacity-40" : ""
              }`
        }`}
      >
        {label}
      </button>
    );
  };
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
        How did that land?
      </div>
      <div className="flex flex-wrap gap-2">
        {btn(
          "pass",
          "✓ Got it",
          "border-emerald-500/60 bg-emerald-500/10 text-emerald-300",
        )}
        {btn(
          "needs_reinforcement",
          "↺ Walk me through again",
          "border-amber-500/60 bg-amber-500/10 text-amber-300",
        )}
        {btn(
          "needs_different_approach",
          "↬ Try a different way",
          "border-rose-500/60 bg-rose-500/10 text-rose-300",
        )}
      </div>
      {chosen && (
        <div className="mt-3 text-xs text-slate-500">
          Verdict recorded. Backend wiring (re-teach / different-approach) is
          the next pass; for now this just visualizes your choice.
        </div>
      )}
    </div>
  );
}

function SessionSummaryCallout({ session }: { session: DerivedSession }) {
  const passed = session.units.filter((u) => u.status === "done").length;
  return (
    <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4">
      <div className="text-sm font-semibold text-emerald-300">
        Session complete · {passed} of {session.units.length} units mastered
      </div>
      <div className="mt-1 text-xs text-slate-400">
        Click any unit on the left to revisit its content.
      </div>
    </div>
  );
}

// ============================================================================
// TeachingContent (game iframe / worked-example markdown / stub)
// ============================================================================

function TeachingContent({
  method,
  content,
}: {
  method: TeachingMethod;
  content: string;
}) {
  if (method === "game") {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={content}
        className="h-[640px] w-full rounded-lg border border-slate-700 bg-white"
        title="micro-game"
      />
    );
  }
  if (method === "worked_example") {
    return <MarkdownLite text={content} />;
  }
  return (
    <div className="whitespace-pre-wrap text-sm italic text-slate-500">
      {content}
    </div>
  );
}

function MarkdownLite({ text }: { text: string }) {
  const segments: { kind: "code" | "prose"; lang?: string; body: string }[] =
    [];
  const fenceRe = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = fenceRe.exec(text)) !== null) {
    if (m.index > last)
      segments.push({ kind: "prose", body: text.slice(last, m.index) });
    segments.push({
      kind: "code",
      lang: m[1] || "text",
      body: m[2].replace(/\n$/, ""),
    });
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ kind: "prose", body: text.slice(last) });
  return (
    <div className="space-y-3 text-sm text-slate-300">
      {segments.map((s, i) =>
        s.kind === "code" ? (
          <pre
            key={i}
            className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs text-slate-100"
          >
            <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
              {s.lang}
            </div>
            <code>{s.body}</code>
          </pre>
        ) : (
          <ProseBlock key={i} text={s.body} />
        ),
      )}
    </div>
  );
}

function ProseBlock({ text }: { text: string }) {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let para: string[] = [];
  const flushPara = (key: string) => {
    if (para.length === 0) return;
    const joined = para.join(" ").trim();
    para = [];
    if (!joined) return;
    nodes.push(
      <p key={key} className="leading-relaxed text-slate-300">
        {renderInline(joined)}
      </p>,
    );
  };
  lines.forEach((line, idx) => {
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) {
      flushPara(`p-${idx}`);
      const level = h[1].length;
      const cls =
        level === 1
          ? "mt-3 text-base font-bold text-slate-100"
          : level === 2
            ? "mt-3 text-sm font-semibold text-slate-100"
            : "mt-3 text-sm font-medium text-slate-200";
      nodes.push(
        <div key={`h-${idx}`} className={cls}>
          {renderInline(h[2])}
        </div>,
      );
      return;
    }
    if (line.trim() === "") {
      flushPara(`p-${idx}`);
      return;
    }
    para.push(line);
  });
  flushPara("p-end");
  return <>{nodes}</>;
}

function renderInline(s: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(s)) !== null) {
    if (m.index > last) parts.push(s.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      parts.push(<strong key={key++}>{tok.slice(2, -2)}</strong>);
    } else {
      parts.push(
        <code
          key={key++}
          className="rounded bg-slate-800 px-1 text-[0.85em] text-amber-300"
        >
          {tok.slice(1, -1)}
        </code>,
      );
    }
    last = m.index + tok.length;
  }
  if (last < s.length) parts.push(s.slice(last));
  return parts;
}

// ============================================================================
// PedagogyBadge
// ============================================================================

const PEDAGOGY_TONE: Record<string, string> = {
  game: "bg-purple-500/15 text-purple-300 border-purple-500/30",
  visual: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  worked_example: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  analogy: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  socratic: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  feynman: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/30",
  retrieval: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

function PedagogyBadge({
  method,
  small,
}: {
  method: string;
  small?: boolean;
}) {
  const tone = PEDAGOGY_TONE[method] ?? PEDAGOGY_TONE.retrieval;
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 ${
        small ? "text-[9px]" : "text-[10px]"
      } font-medium uppercase tracking-wider ${tone}`}
    >
      {method}
    </span>
  );
}
