from enum import StrEnum

from pydantic import BaseModel, Field


class KnowledgeLevel(StrEnum):
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DesiredDepth(StrEnum):
    COCKTAIL = "cocktail"
    WORKING = "working"
    EXPERT = "expert"


class Modality(StrEnum):
    VISUAL = "visual"
    VERBAL = "verbal"
    INTERACTIVE = "interactive"


class LearningGoal(BaseModel):
    domain: str = Field(min_length=1)
    sub_goal: str = Field(min_length=1)
    current_level: KnowledgeLevel
    desired_depth: DesiredDepth
    time_budget_hours: float = Field(gt=0)
    preferred_modalities: list[Modality] = Field(default_factory=list)
