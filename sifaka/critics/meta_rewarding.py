"""Meta-Rewarding critic implementation.

Based on: Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge
Paper: https://arxiv.org/abs/2407.19594
Authors: Wu et al. (2024)

Meta-Rewarding uses a two-stage judgment process where the model evaluates
its own evaluations to improve critique quality.

## Similarity to Original Paper:
- PRESERVED: Two-stage evaluation process
- PRESERVED: Meta-judgment of evaluation quality
- ENHANCED: Iterative critique refinement
- ENHANCED: Preference learning from meta-evaluation
- PRESERVED: Self-improving judgment

## Implementation Choices:
1. First stage: Generate initial critique
2. Second stage: Meta-evaluate the critique
3. Third stage: Refine critique based on meta-evaluation
4. Track improvement metrics for learning
5. Generate alternative suggestions with preferences

## Why This Approach:
- Matches paper's self-improvement concept
- Iterative refinement produces better critiques
- Preference learning improves over time
- Meta-rewards guide critique quality
"""

from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class CritiqueEvaluation(BaseModel):
    """Meta-evaluation of a critique aspect."""

    aspect: str = Field(
        ..., description="The aspect being evaluated (e.g., accuracy, helpfulness)"
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality score for this aspect"
    )
    reasoning: str = Field(..., description="Reasoning for the score")
    improvement_needed: bool = Field(
        ..., description="Whether this aspect needs improvement"
    )


class SuggestionPreference(BaseModel):
    """Preference ranking for alternative suggestions."""

    suggestion: str = Field(..., description="The suggestion text")
    preference_score: float = Field(..., ge=0.0, le=1.0, description="Preference score")
    rationale: str = Field(
        ..., description="Why this suggestion is preferred/not preferred"
    )


class MetaRewardingResponse(BaseModel):
    """Response model for Meta-Rewarding critic with iterative refinement."""

    # Initial critique
    feedback: str = Field(..., description="Refined critique after meta-evaluation")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Final improvement suggestions after refinement",
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Final confidence after meta-evaluation",
    )

    # Meta-evaluation fields
    initial_feedback: str = Field(..., description="Initial critique before refinement")
    meta_evaluation: str = Field(
        ..., description="Meta-evaluation of the initial critique"
    )
    critique_evaluations: list[CritiqueEvaluation] = Field(
        default_factory=list, description="Detailed evaluation of critique aspects"
    )

    # Refinement and learning
    refinement_rationale: str = Field(
        ..., description="How the critique was refined based on meta-evaluation"
    )
    suggestion_preferences: list[SuggestionPreference] = Field(
        default_factory=list,
        description="Ranked preferences for alternative suggestions",
    )

    # Quality metrics
    initial_quality_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Quality of initial critique"
    )
    final_quality_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Quality after refinement"
    )
    improvement_delta: float = Field(
        default=0.1, ge=-1.0, le=1.0, description="How much the critique improved"
    )

    # Learning metrics
    meta_reward: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Meta-reward signal for learning"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional meta-rewarding data"
    )


class MetaRewardingCritic(BaseCritic):
    """Implements Meta-Rewarding two-stage critique.

    ## When to Use This Critic:

    ✅ When to use:
    - High-stakes content requiring thorough review
    - Documents where critique quality itself matters
    - Complex evaluations needing self-reflection
    - Final quality assurance before publication

    ❌ When to avoid:
    - Quick iterative improvements
    - Simple text corrections
    - When single-pass evaluation is sufficient

    🎯 Best for:
    - Executive communications
    - Legal or compliance documents
    - Published articles or white papers
    - Critical business proposals
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "meta_rewarding"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom MetaRewardingResponse for structured output."""
        return MetaRewardingResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for two-stage meta-rewarding critique."""
        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""Apply the Meta-Rewarding framework with three stages: critique, meta-evaluate, and refine.

Text to evaluate:
{text}
{previous_context}

STAGE 1 - Initial Critique:
Generate your first critique considering:
- Content quality and accuracy
- Structure and organization
- Clarity and readability
- Completeness and depth
- Engagement and impact

STAGE 2 - Meta-Evaluation:
Evaluate your initial critique on these dimensions:
- Accuracy: Are your observations correct?
- Helpfulness: Will your suggestions improve the text?
- Coverage: Did you miss important issues?
- Fairness: Is your critique balanced?
- Specificity: Are your suggestions actionable?

Score each dimension and identify weaknesses in your critique.

STAGE 3 - Refined Critique:
Based on your meta-evaluation:
1. Refine your initial feedback to address identified weaknesses
2. Generate alternative suggestions and rank them by preference
3. Adjust confidence based on critique quality
4. Calculate improvement delta and meta-reward

Provide both initial and refined versions to show the improvement process."""

        return [
            {
                "role": "system",
                "content": "You are a Meta-Rewarding critic that uses three-stage evaluation: initial critique, meta-evaluation of that critique, then refined critique based on meta-rewards. You learn to improve your judgments through self-reflection.",
            },
            {"role": "user", "content": user_prompt},
        ]
