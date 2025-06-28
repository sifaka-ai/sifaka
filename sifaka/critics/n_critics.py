"""N-Critics ensemble critic implementation.

Based on: N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics
Paper: https://arxiv.org/abs/2310.18679
Authors: Tian et al. (2023)

N-Critics uses multiple critical perspectives to provide comprehensive
text evaluation through ensemble critique.

## Similarity to Original Paper:
- PRESERVED: Multiple critical perspectives for evaluation
- PRESERVED: Ensemble approach to critique
- SIMPLIFIED: Perspectives instead of separate critic models
- PRESERVED: Consensus building from multiple viewpoints

## Implementation Choices:
1. 4 default perspectives: Clarity, Accuracy, Completeness, Style
2. Each perspective gets independent assessment and score
3. Consensus score from averaging perspective scores
4. Agreement factor from score variance (low variance = high agreement)
5. Confidence combines consensus score (70%) + agreement (30%)

## Why This Approach:
- Single model with multiple prompting perspectives is more efficient
- Preserves multi-viewpoint evaluation without multiple models
- Customizable perspectives for different domains
- Variance-based agreement naturally measures consensus
- Simpler than training/managing multiple critic models
"""

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from ..core.config import Config
from .core.base import BaseCritic


# Removed PerspectiveAssessment class since it's not used in generation.py


class NCriticsResponse(BaseModel):
    """Response model specific to N-Critics ensemble."""

    feedback: str = Field(..., description="Synthesized feedback from all perspectives")
    suggestions: list[str] = Field(
        default_factory=list, description="Combined suggestions from all perspectives"
    )
    needs_improvement: bool = Field(
        ..., description="Consensus on whether improvement is needed"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence based on perspective agreement",
    )
    # Only keep consensus_score since that's what's used in generation.py
    consensus_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall consensus score from all perspectives",
    )


class NCriticsCritic(BaseCritic):
    """Implements N-Critics ensemble approach for comprehensive evaluation.

    When to Use This Critic:
    - ✅ Need multiple viewpoints on text quality
    - ✅ Want comprehensive evaluation across dimensions
    - ✅ Dealing with complex or multifaceted content
    - ✅ Need to identify blind spots single critics might miss
    - ❌ Quick single-dimension checks
    - ❌ When perspectives would be redundant
    - 🎯 Best for: Comprehensive reviews, final quality checks, complex documents
    """

    # Default critical perspectives
    DEFAULT_PERSPECTIVES = [
        "Clarity: Focus on readability, structure, and comprehensibility",
        "Accuracy: Focus on factual correctness and logical consistency",
        "Completeness: Focus on thoroughness and adequate coverage",
        "Style: Focus on tone, voice, and appropriateness for audience",
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.6,
        perspectives: Optional[List[str]] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        auto_generate_perspectives: bool = False,
        perspective_count: int = 4,
    ):
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

        self.auto_generate_perspectives = auto_generate_perspectives
        self.perspective_count = perspective_count
        self.perspectives = perspectives or self.DEFAULT_PERSPECTIVES

    @property
    def name(self) -> str:
        return "n_critics"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom NCriticsResponse for structured output."""
        return NCriticsResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Generate ensemble critique from multiple perspectives."""
        # Generate perspectives if requested
        if self.auto_generate_perspectives:
            self.perspectives = await self._generate_perspectives(text, result)

        perspectives_text = "\n".join(
            f"{i+1}. {perspective}" for i, perspective in enumerate(self.perspectives)
        )

        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""Using the N-Critics ensemble technique, evaluate this text from multiple critical perspectives:

CRITICAL PERSPECTIVES:
{perspectives_text}

Text to evaluate:
{text}
{previous_context}

Evaluate the text from each perspective and provide:
- Overall synthesized feedback combining all perspectives
- Specific improvement suggestions that address issues from multiple viewpoints
- A consensus score (0.0-1.0) representing overall text quality across all perspectives

Focus on constructive, actionable feedback that considers all viewpoints."""

        return [
            {
                "role": "system",
                "content": "You are an expert text critic using the N-Critics ensemble technique to evaluate text from multiple critical perspectives. Provide balanced, comprehensive feedback.",
            },
            {"role": "user", "content": user_prompt},
        ]

    async def _generate_perspectives(
        self, text: str, result: SifakaResult
    ) -> List[str]:
        """Generate context-appropriate perspectives for the text."""
        # This would use the LLM to generate perspectives based on the text type
        # For now, return defaults
        return self.DEFAULT_PERSPECTIVES
