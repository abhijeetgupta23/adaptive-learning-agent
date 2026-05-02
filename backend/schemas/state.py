from pydantic import BaseModel, Field

from backend.schemas.teaching import TeachingMethod


class LearnerState(BaseModel):
    user_id: str = Field(min_length=1)
    mastered_concepts: list[str] = Field(default_factory=list)
    struggled_concepts: list[str] = Field(default_factory=list)
    landed_analogies: dict[str, str] = Field(default_factory=dict)
    effective_pedagogies: dict[str, TeachingMethod] = Field(default_factory=dict)
