from enum import StrEnum

from pydantic import BaseModel, Field


class Verdict(StrEnum):
    PASS = "pass"
    NEEDS_REINFORCEMENT = "needs_reinforcement"
    NEEDS_DIFFERENT_APPROACH = "needs_different_approach"


class AssessmentResult(BaseModel):
    unit_id: str = Field(min_length=1)
    verdict: Verdict
    diagnostic_notes: str
    confidence: float = Field(ge=0.0, le=1.0)
