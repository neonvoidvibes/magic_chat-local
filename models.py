from pydantic import BaseModel
from typing import List, Optional, Dict

class KeywordAnalysis(BaseModel):
    keyword: str
    frequency: int
    context: str

class ThemeAnalysis(BaseModel):
    theme: str
    supporting_evidence: str
    related_keywords: List[str]

class ConversationPattern(BaseModel):
    pattern: str
    evidence: str
    impact: str
    frequency: int
    timestamps: List[str]

class LatentNeed(BaseModel):
    need: str
    urgency_score: float
    evidence: str

class OrganizationalNeed(BaseModel):
    need: str
    urgency_score: float
    evidence: str

class LatentContent(BaseModel):
    content: str
    interpretation: str
    confidence_score: float

class EmergingInsight(BaseModel):
    insight: str
    evidence: str
    potential_impact: str

class TrajectoryForecast(BaseModel):
    trajectory: str
    likelihood: float
    implications: str
    timeframe: str

class LeverageOpportunity(BaseModel):
    opportunity: str
    potential_impact: str
    impact_score: float
    effort_score: float
    evidence: str

class SingleLoopLearning(BaseModel):
    explicit_learning: str
    hidden_learning: str
    implications: str

class DoubleLoopLearning(BaseModel):
    explicit_learning: str
    hidden_learning: str
    implications: str

class TripleLoopLearning(BaseModel):
    explicit_learning: str
    hidden_learning: str
    implications: str

class Sovereignty(BaseModel):
    sentience: str
    intelligence: str
    agency: str
    evolution: str

class Integral(BaseModel):
    subjective_perspective: str
    objective_perspective: str
    individual_domain: str
    collective_domain: str
    integral_capacity: float

class InsightsOutput(BaseModel):
    keywords: List[KeywordAnalysis]
    themes: List[ThemeAnalysis]
    conversation_patterns: List[ConversationPattern]
    latent_needs: List[LatentNeed]
    organizational_needs: List[OrganizationalNeed]
    latent_content: List[LatentContent]
    emerging_insights: List[EmergingInsight]
    trajectory_forecasts: List[TrajectoryForecast]
    leverage_opportunities: List[LeverageOpportunity]
    single_loop_learning: List[SingleLoopLearning]
    double_loop_learning: List[DoubleLoopLearning]
    triple_loop_learning: List[TripleLoopLearning]
    sovereignty_level: List[Sovereignty]
    integral_perspective: List[Integral]