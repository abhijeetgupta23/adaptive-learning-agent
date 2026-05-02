from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TeachingMethod(StrEnum):
    WORKED_EXAMPLE = "worked_example"
    ANALOGY = "analogy"
    VISUAL = "visual"
    GAME = "game"
    SOCRATIC = "socratic"
    FEYNMAN = "feynman"
    RETRIEVAL = "retrieval"


class MethodCall(BaseModel):
    method: TeachingMethod
    params: dict[str, Any] = Field(default_factory=dict)


class TeachingPlan(BaseModel):
    unit_id: str = Field(min_length=1)
    methods: list[MethodCall] = Field(min_length=1)
    rationale: str | None = None
