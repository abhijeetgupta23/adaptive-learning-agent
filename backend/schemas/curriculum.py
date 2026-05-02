from pydantic import BaseModel, Field, HttpUrl

from backend.schemas.goal import LearningGoal
from backend.schemas.teaching import TeachingMethod


class GroundingSource(BaseModel):
    url: HttpUrl
    title: str = Field(min_length=1)
    snippet: str | None = None


class LearningUnit(BaseModel):
    id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    prerequisites: list[str] = Field(default_factory=list)
    recommended_pedagogy: TeachingMethod
    grounding_sources: list[GroundingSource] = Field(min_length=1)


class Curriculum(BaseModel):
    goal: LearningGoal
    units: list[LearningUnit] = Field(min_length=1)
