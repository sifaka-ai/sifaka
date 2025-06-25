"""Self-Refine critic implementation.

Based on: Self-Refine: Iterative Refinement with Self-Feedback
Paper: https://arxiv.org/abs/2303.17651
Authors: Madaan et al. (2023)

Self-Refine uses iterative self-feedback to improve outputs without
additional training or external feedback.

## Similarity to Original Paper:
- PRESERVED: Core iterative self-feedback loop
- PRESERVED: Quality-focused refinement criteria
- SIMPLIFIED: Single feedback generation per iteration
- ADAPTED: Focus on general text improvement vs task-specific

## Implementation Choices:
1. Emphasis on quality dimensions: clarity, completeness, coherence
2. Concrete actionable feedback prioritized
3. Tracks refinement history to avoid loops
4. Balances between minor edits and substantial improvements

## Why This Approach:
- Self-refinement is natural for text improvement
- No need for external feedback sources
- Works well in iterative improvement loops
- Simple yet effective for quality enhancement
"""

from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class RefinementArea(BaseModel):
    """A specific area identified for refinement."""

    area: str = Field(..., description="The aspect needing refinement")
    current_state: str = Field(..., description="Current state of this area")
    target_state: str = Field(..., description="Desired improved state")
    priority: str = Field(
        default="medium", description="Priority level: high, medium, low"
    )


class SelfRefineResponse(BaseModel):
    """Response model specific to Self-Refine critic."""

    feedback: str = Field(..., description="Detailed refinement feedback")
    suggestions: list[str] = Field(
        default_factory=list, description="Specific refinement actions"
    )
    needs_improvement: bool = Field(
        ..., description="Whether further refinement would benefit the text"
    )
    confidence: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Confidence in refinement assessment"
    )
    refinement_areas: list[RefinementArea] = Field(
        default_factory=list, description="Specific areas needing refinement"
    )
    quality_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Overall quality score"
    )
    refinement_iterations_recommended: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Number of additional refinement iterations recommended",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional refinement data"
    )


class SelfRefineCritic(BaseCritic):
    """Implements Self-Refine iterative refinement approach.

    ## When to Use This Critic:

    ✅ When to use:
    - Improving general text quality and polish
    - Iterative refinement of existing content
    - Enhancing clarity, completeness, and coherence
    - Making text more engaging and professional

    ❌ When to avoid:
    - When specific domain expertise is required
    - For fact-checking or accuracy verification
    - When you need multiple perspectives simultaneously

    🎯 Best for:
    - General content improvement
    - Readability enhancement
    - Professional document polishing
    - Iterative quality improvement cycles
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
        return "self_refine"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom SelfRefineResponse for structured output."""
        return SelfRefineResponse

    def _get_system_prompt(self) -> str:
        """Get system prompt for Self-Refine critic."""
        return "You are a Self-Refine critic that provides iterative refinement feedback to improve text quality across multiple dimensions."

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for self-refinement evaluation."""
        # Get refinement history
        refinement_context = self._get_refinement_context(result)

        instructions = f"""You are tasked with providing self-refinement feedback for this text.

{refinement_context}

Evaluate the text across these quality dimensions:

1. **Clarity**: Is the text clear and easy to understand?
2. **Completeness**: Does it cover all necessary aspects?
3. **Coherence**: Is it well-structured and logically organized?
4. **Engagement**: Is it interesting and compelling?
5. **Accuracy**: Are claims supported and reasonable?
6. **Conciseness**: Is it appropriately concise without losing meaning?

Provide specific, actionable feedback for refinement. Focus on the most impactful improvements that would enhance the text quality."""

        return await self._simple_critique(text, result, instructions)

    def _get_refinement_context(self, result: SifakaResult) -> str:
        """Get context about the refinement process."""
        if not result.generations:
            return "This is the initial text requiring refinement."

        context_parts = [
            f"The text has undergone {len(result.generations)} refinement iterations."
        ]

        # Note text evolution
        if len(result.generations) >= 2:
            first_gen = result.generations[0]
            last_gen = result.generations[-1]

            # Calculate text growth
            growth = len(last_gen.text) - len(first_gen.text)
            growth_pct = (
                (growth / len(first_gen.text) * 100) if len(first_gen.text) > 0 else 0
            )

            context_parts.append(
                f"Text has {'grown' if growth > 0 else 'shrunk'} by {abs(growth)} characters ({abs(growth_pct):.1f}%)"
            )

        return "\n".join(context_parts)
