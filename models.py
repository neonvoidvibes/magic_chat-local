from pydantic import BaseModel
from typing import List, Optional

class ConversationPattern(BaseModel):
    pattern: str
    evidence: str
    timestamps: List[str]

class OrganizationalNeed(BaseModel):
    need: str
    urgency_score: float
    evidence: str

class EmergingTrajectory(BaseModel):
    trajectory: str
    implications: str

class LeveragePoint(BaseModel):
    point: str
    potential_impact: str

class LatentInsight(BaseModel):
    insight: str
    transformation_potential: str

class WeightedConclusion(BaseModel):
    weight: float
    category: str
    conclusion: str
    evidence: str

class InsightsOutput(BaseModel):
    conversation_patterns: List[ConversationPattern]
    organizational_needs: List[OrganizationalNeed]
    emerging_trajectories: List[EmergingTrajectory]
    leverage_points: List[LeveragePoint]
    latent_insights: List[LatentInsight]
    weighted_conclusions: List[WeightedConclusion]