"""Self-Refine critic implementation for iterative text improvement.

Based on: Self-Refine: Iterative Refinement with Self-Feedback
Paper: https://arxiv.org/abs/2303.17651
Authors: Madaan et al. (2023)

Self-Refine uses iterative self-feedback to improve outputs without
additional training or external feedback. This implementation adapts
the approach for general text quality improvement.

## Similarity to Original Paper:

- PRESERVED: Core iterative self-feedback loop
- PRESERVED: Quality-focused refinement criteria
- PRESERVED: Self-directed improvement without external feedback
- SIMPLIFIED: Single feedback generation per iteration (vs. multiple rounds)
- ADAPTED: General text improvement vs. task-specific optimization

## Implementation Strategy:

1. **Multi-Dimensional Quality Assessment**: Evaluates text across 6 quality dimensions:
   - Clarity: Ease of understanding and comprehension
   - Completeness: Coverage of all necessary aspects
   - Coherence: Logical structure and organization
   - Engagement: Interest and compelling nature
   - Accuracy: Reasonableness and support for claims
   - Conciseness: Appropriate brevity without meaning loss

2. **Refinement History Tracking**: Monitors iteration count and text evolution
   to provide context-aware feedback and prevent unnecessary changes

3. **Actionable Feedback Generation**: Focuses on specific, implementable
   improvements rather than general observations

4. **Target State Definition**: Identifies specific areas for refinement
   with clear descriptions of desired improvements

## Why This Approach:

- **Natural Fit**: Self-refinement aligns perfectly with iterative text improvement
- **Self-Contained**: No external resources or feedback sources required
- **Scalable**: Works effectively across different content types and domains
- **Efficient**: Provides substantial quality improvements with minimal overhead
- **Transparent**: Clear feedback on what needs improvement and why

## Best Use Cases:

This critic excels at general-purpose text polishing and is particularly
effective for content that benefits from multiple revision cycles.
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.llm_client import Provider
from ..core.models import SifakaResult
from .core.base import BaseCritic


class RefinementArea(BaseModel):
    """Represents a specific area identified for refinement.

    Defines a targeted improvement opportunity with a clear description
    of the desired end state. Used by the text generator to understand
    exactly what changes are needed.

    Attributes:
        target_state: Clear description of how this area should be improved.
            Should be specific and actionable rather than vague.

    Example:
        >>> area = RefinementArea(
        ...     target_state="Make the introduction more engaging by adding a hook"
        ... )
    """

    target_state: str = Field(..., description="Desired improved state")


class SelfRefineResponse(BaseModel):
    """Structured response model for Self-Refine critique.

    Provides comprehensive feedback on text quality across multiple dimensions,
    with specific refinement areas and actionable suggestions for improvement.

    Attributes:
        feedback: Detailed assessment of current text quality with specific
            observations about what works well and what needs improvement.
        suggestions: List of specific, actionable refinement steps that can
            be implemented to improve the text.
        needs_improvement: Boolean indicating whether the text would benefit
            from further refinement iterations.
        confidence: Confidence level in the refinement assessment (0.0-1.0).
        refinement_areas: Specific areas identified for targeted improvement,
            each with a clear target state description.
    """

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
    # Only keep refinement_areas since that's what's used in generation.py
    refinement_areas: list[RefinementArea] = Field(
        default_factory=list, description="Specific areas needing refinement"
    )


class SelfRefineCritic(BaseCritic):
    """Implements Self-Refine iterative refinement approach.

    Provides comprehensive text quality assessment and refinement guidance
    across multiple dimensions, with emphasis on iterative improvement
    through self-feedback loops.

    ## When to Use This Critic:

    ✅ **Ideal for:**
    - General text quality improvement and polishing
    - Iterative refinement of existing content over multiple cycles
    - Enhancing clarity, completeness, and coherence
    - Making text more engaging and professionally written
    - Content that benefits from multiple revision rounds
    - Situations where you want comprehensive quality feedback

    ❌ **Avoid when:**
    - Specific domain expertise or specialized knowledge is required
    - Primary need is fact-checking or accuracy verification
    - You need multiple perspectives or consensus-building
    - Content is already highly polished and near-final
    - Time constraints don't allow for iterative improvement

    🎯 **Optimal applications:**
    - Blog posts and articles going through multiple drafts
    - Professional documents requiring high polish
    - Educational content needing clarity optimization
    - Marketing copy requiring engagement improvement
    - Technical documentation needing readability enhancement
    - Creative writing benefiting from structural refinement

    ## Quality Dimensions Evaluated:

    1. **Clarity**: Ease of understanding and comprehension
    2. **Completeness**: Coverage of all necessary aspects
    3. **Coherence**: Logical structure and organization
    4. **Engagement**: Interest and compelling nature
    5. **Accuracy**: Reasonableness and support for claims
    6. **Conciseness**: Appropriate brevity without meaning loss

    ## Refinement Strategy:

    The critic tracks refinement history to provide context-aware feedback,
    preventing over-refinement and focusing on the most impactful improvements
    at each iteration.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """Initialize Self-Refine critic with quality assessment capabilities.

        Creates a critic focused on iterative text refinement across multiple
        quality dimensions, with tracking of refinement history for context.

        Args:
            model: LLM model for refinement assessment. GPT-4o-mini provides
                good balance of quality and cost for iterative refinement.
            temperature: Generation temperature (0.0-1.0). Moderate values
                (0.6-0.8) work well for balanced refinement feedback.
            provider: LLM provider (OpenAI, Anthropic, etc.)
            api_key: API key override if not using environment variables
            config: Full Sifaka configuration object

        Example:
            >>> # Standard refinement critic
            >>> critic = SelfRefineCritic()
            >>>
            >>> # More creative refinement suggestions
            >>> critic = SelfRefineCritic(temperature=0.8)
            >>>
            >>> # Conservative, consistent refinement
            >>> critic = SelfRefineCritic(temperature=0.5)

        Note:
            The critic automatically tracks refinement iterations and provides
            context-aware feedback to prevent over-refinement.
        """
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        """Return the unique identifier for this critic.

        Returns:
            "self_refine" - used in configuration, logging, and metadata
        """
        return "self_refine"

    def _get_response_type(self) -> type[BaseModel]:
        """Specify the structured response format for refinement feedback.

        Returns:
            SelfRefineResponse class providing detailed refinement analysis
            with specific areas for improvement and target states.
        """
        return SelfRefineResponse

    def _get_system_prompt(self) -> str:
        """Generate system prompt for Self-Refine evaluation.

        Creates a system prompt that establishes the critic's role as an
        iterative refinement specialist focused on quality improvement
        across multiple dimensions.

        Returns:
            System prompt emphasizing iterative refinement and quality focus
        """
        return "You are a Self-Refine critic that provides iterative refinement feedback to improve text quality across multiple dimensions."

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create comprehensive messages for self-refinement evaluation.

        Builds detailed evaluation instructions that include refinement context,
        quality dimensions to assess, and guidance for providing actionable
        feedback.

        Args:
            text: Text to evaluate for refinement opportunities
            result: SifakaResult containing refinement history and context

        Returns:
            List of message dictionaries for LLM conversation, including
            system context and detailed evaluation instructions

        Note:
            Incorporates refinement history to provide context-aware feedback
            and prevent over-refinement in later iterations.
        """
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
        """Generate context about the refinement process and history.

        Analyzes the refinement history to provide context about how many
        iterations have occurred and how the text has evolved, helping
        guide appropriate feedback for the current iteration.

        Args:
            result: SifakaResult containing generation history

        Returns:
            Formatted string describing refinement context including:
            - Number of refinement iterations completed
            - Text length changes over iterations
            - Evolution indicators for context-aware feedback

        Note:
            Provides important context to prevent over-refinement and focus
            on the most valuable improvements at each stage.
        """
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
