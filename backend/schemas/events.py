from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from backend.schemas.assessment import AssessmentResult
from backend.schemas.curriculum import Curriculum
from backend.schemas.goal import LearningGoal
from backend.schemas.teaching import TeachingMethod, TeachingPlan


class SessionStartedEvent(BaseModel):
    event: Literal["session_started"] = "session_started"
    session_id: str


class IntentUpdateEvent(BaseModel):
    event: Literal["intent_update"] = "intent_update"
    message: str
    goal: LearningGoal | None = None


class IntentAskEvent(BaseModel):
    """Server pauses execution and asks the user a clarifying question."""
    event: Literal["intent_ask"] = "intent_ask"
    session_id: str
    question: str


class AssessmentQuestionEvent(BaseModel):
    """Assessment agent asks the learner a probing question after teaching."""
    event: Literal["assessment_question"] = "assessment_question"
    session_id: str
    unit_id: str
    question: str
    attempt: int


class CurriculumReadyEvent(BaseModel):
    event: Literal["curriculum_ready"] = "curriculum_ready"
    curriculum: Curriculum


class UnitStartedEvent(BaseModel):
    event: Literal["unit_started"] = "unit_started"
    unit_id: str
    plan: TeachingPlan


class TeachingChunkEvent(BaseModel):
    event: Literal["teaching_chunk"] = "teaching_chunk"
    unit_id: str
    method: TeachingMethod
    content: str


class AssessmentResultEvent(BaseModel):
    event: Literal["assessment_result"] = "assessment_result"
    result: AssessmentResult


class CostUpdateEvent(BaseModel):
    event: Literal["cost_update"] = "cost_update"
    total_usd: float
    by_agent: dict[str, float]
    cache_hit_rate: float
    call_count: int


class CircuitBreakerTrippedEvent(BaseModel):
    """Game-agent circuit breaker has tripped; remaining units fall back to
    worked_example for this session."""
    event: Literal["circuit_breaker_tripped"] = "circuit_breaker_tripped"
    reason: str
    fallback_method: TeachingMethod = TeachingMethod.WORKED_EXAMPLE


class CompleteEvent(BaseModel):
    event: Literal["complete"] = "complete"


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    message: str


SSEEvent = Annotated[
    SessionStartedEvent
    | IntentUpdateEvent
    | IntentAskEvent
    | CurriculumReadyEvent
    | UnitStartedEvent
    | TeachingChunkEvent
    | AssessmentQuestionEvent
    | AssessmentResultEvent
    | CostUpdateEvent
    | CircuitBreakerTrippedEvent
    | CompleteEvent
    | ErrorEvent,
    Field(discriminator="event"),
]

SSEEventAdapter: TypeAdapter[SSEEvent] = TypeAdapter(SSEEvent)
