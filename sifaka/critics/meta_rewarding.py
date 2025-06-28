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


# Removed CritiqueEvaluation and SuggestionPreference classes since they're not used in generation.py


class MetaRewardingResponse(BaseModel):
    """Response model for Meta-Rewarding critic."""

    # Only include the standard fields since no metadata is used in generation.py
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

        user_prompt = f"""Apply the Meta-Rewarding framework: generate an initial critique, then meta-evaluate it, and finally provide a refined critique.

Text to evaluate:
{text}
{previous_context}

Process:
1. Generate your initial critique considering content quality, structure, clarity, completeness, and engagement
2. Meta-evaluate your initial critique for accuracy, helpfulness, coverage, fairness, and specificity
3. Provide a refined critique that addresses any weaknesses identified in the meta-evaluation

Focus on delivering actionable, high-quality feedback that has been improved through self-reflection."""

        return [
            {
                "role": "system",
                "content": "You are a Meta-Rewarding critic that uses three-stage evaluation: initial critique, meta-evaluation of that critique, then refined critique based on meta-rewards. You learn to improve your judgments through self-reflection.",
            },
            {"role": "user", "content": user_prompt},
        ]
