// Mirror of backend.schemas.events — keep in sync.

export type KnowledgeLevel = "novice" | "beginner" | "intermediate" | "advanced";
export type DesiredDepth = "cocktail" | "working" | "expert";
export type Modality = "visual" | "verbal" | "interactive";

export type TeachingMethod =
  | "worked_example"
  | "analogy"
  | "visual"
  | "game"
  | "socratic"
  | "feynman"
  | "retrieval";

export type Verdict = "pass" | "needs_reinforcement" | "needs_different_approach";

export interface LearningGoal {
  domain: string;
  sub_goal: string;
  current_level: KnowledgeLevel;
  desired_depth: DesiredDepth;
  time_budget_hours: number;
  preferred_modalities: Modality[];
}

export interface GroundingSource {
  url: string;
  title: string;
  snippet?: string | null;
}

export interface LearningUnit {
  id: string;
  objective: string;
  prerequisites: string[];
  recommended_pedagogy: TeachingMethod;
  grounding_sources: GroundingSource[];
}

export interface Curriculum {
  goal: LearningGoal;
  units: LearningUnit[];
}

export interface TeachingPlan {
  unit_id: string;
  methods: { method: TeachingMethod; params: Record<string, unknown> }[];
  rationale?: string | null;
}

export interface AssessmentResult {
  unit_id: string;
  verdict: Verdict;
  diagnostic_notes: string;
  confidence: number;
}

export interface CostUpdate {
  total_usd: number;
  by_agent: Record<string, number>;
  cache_hit_rate: number;
  call_count: number;
}

export type SSEEvent =
  | { event: "session_started"; session_id: string }
  | { event: "intent_update"; message: string; goal?: LearningGoal | null }
  | { event: "intent_ask"; session_id: string; question: string }
  | { event: "curriculum_ready"; curriculum: Curriculum }
  | { event: "unit_started"; unit_id: string; plan: TeachingPlan }
  | {
      event: "teaching_chunk";
      unit_id: string;
      method: TeachingMethod;
      content: string;
    }
  | { event: "assessment_result"; result: AssessmentResult }
  | ({ event: "cost_update" } & CostUpdate)
  | { event: "complete" }
  | { event: "error"; message: string };

export async function* readSseStream(
  response: Response,
): AsyncGenerator<SSEEvent, void, void> {
  if (!response.body) throw new Error("No response body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split(/\r?\n/).find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        yield JSON.parse(line.slice(6)) as SSEEvent;
      } catch {
        // skip malformed frames
      }
    }
  }
}
