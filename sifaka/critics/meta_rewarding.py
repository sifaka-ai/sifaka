"""Meta-Rewarding critic implementation for self-improving critique quality.

Based on: Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge
Paper: https://arxiv.org/abs/2407.19594
Authors: Wu et al. (2024)

Meta-Rewarding uses a three-stage judgment process where the model generates
an initial critique, evaluates its own evaluation, then refines the critique
based on meta-rewards to improve overall critique quality.

## Similarity to Original Paper:

- PRESERVED: Two-stage evaluation process (initial + meta-evaluation)
- PRESERVED: Meta-judgment of evaluation quality using self-reflection
- PRESERVED: Self-improving judgment through iterative refinement
- ENHANCED: Three-stage process (initial → meta-eval → refined critique)
- ENHANCED: Explicit preference learning from meta-evaluation feedback
- ADAPTED: Applied to text critique rather than general alignment

## Implementation Strategy:

1. **Initial Critique Generation**: Standard evaluation across quality dimensions
   (content, structure, clarity, completeness, engagement)

2. **Meta-Evaluation**: Self-assessment of the initial critique for:
   - Accuracy: How well does the critique reflect actual text quality?
   - Helpfulness: Are suggestions actionable and constructive?
   - Coverage: Does the critique address all important aspects?
   - Fairness: Is the evaluation balanced and unbiased?
   - Specificity: Are observations concrete rather than generic?

3. **Refined Critique**: Enhanced evaluation incorporating meta-evaluation insights,
   resulting in higher-quality, more actionable feedback

## Why This Approach:

- **Quality Assurance**: Meta-evaluation catches weak or inaccurate critiques
- **Self-Improvement**: Each evaluation refines the critic's assessment ability
- **Thoroughness**: Three-stage process ensures comprehensive evaluation
- **Actionability**: Meta-evaluation emphasizes practical, implementable feedback
- **Transparency**: Process makes critique reasoning explicit and traceable

## Trade-offs:

**Benefits:**
- Higher-quality, more reliable critiques
- Self-correcting evaluation process
- Excellent for high-stakes content review

**Costs:**
- 3x token usage compared to single-stage critics
- Longer processing time due to multi-stage evaluation
- May be overkill for simple or low-stakes content

This critic is ideal when critique quality itself is critical to success.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.llm_client import Provider
from ..core.models import SifakaResult
from .core.base import BaseCritic

# Removed CritiqueEvaluation and SuggestionPreference classes since they're not used in generation.py


class MetaRewardingResponse(BaseModel):
    """Structured response model for Meta-Rewarding critique.

    Contains the final refined critique after the three-stage meta-rewarding
    process, providing high-quality feedback that has been self-evaluated
    and improved through meta-reflection.

    Attributes:
        feedback: Final refined critique incorporating meta-evaluation insights.
            Represents the highest-quality assessment after self-improvement.
        suggestions: Actionable improvement recommendations that have been
            refined through meta-evaluation for maximum helpfulness.
        needs_improvement: Boolean assessment refined through meta-evaluation
            to ensure accuracy and appropriateness.
        confidence: Final confidence level adjusted based on meta-evaluation
            of critique quality and thoroughness.
        metadata: Additional meta-rewarding process data including stages
            of evaluation and refinement insights.
    """

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
    """Implements Meta-Rewarding three-stage critique for superior quality.

    Provides the highest-quality critiques through a sophisticated three-stage
    process: initial evaluation, meta-assessment of that evaluation, and
    refined critique based on self-improvement insights.

    ## When to Use This Critic:

    ✅ **Ideal for:**
    - High-stakes content requiring exceptional critique quality
    - Documents where accuracy of evaluation is critical
    - Complex content needing deep, reflective analysis
    - Final quality assurance before important publications
    - Content where the cost of poor feedback is high
    - Situations requiring the most thorough possible evaluation

    ❌ **Avoid when:**
    - Quick iterative improvements during drafting
    - Simple text corrections or minor edits
    - Resource-constrained environments (3x token cost)
    - Time-sensitive evaluations needing rapid feedback
    - Content where good-enough critique suffices

    🎯 **Optimal applications:**
    - C-suite communications and board presentations
    - Legal documents and compliance materials
    - Published research papers and white papers
    - Critical business proposals and RFP responses
    - Public relations and crisis communications
    - Academic papers and peer review preparation
    - High-visibility marketing and brand content

    ## Process Overview:

    1. **Initial Critique**: Comprehensive evaluation across standard dimensions
    2. **Meta-Evaluation**: Self-assessment of critique quality and completeness
    3. **Refined Critique**: Enhanced feedback incorporating meta-insights

    ## Quality Guarantees:

    - **Accuracy**: Meta-evaluation catches assessment errors
    - **Completeness**: Ensures all important aspects are covered
    - **Actionability**: Emphasizes practical, implementable suggestions
    - **Balance**: Self-correction prevents overly harsh or lenient feedback
    - **Specificity**: Meta-evaluation promotes concrete over generic observations

    ## Resource Considerations:

    This critic uses approximately 3x the tokens of standard critics due to
    the three-stage process. Budget accordingly for high-volume usage.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """Initialize Meta-Rewarding critic for premium-quality evaluation.

        Creates a sophisticated critic that uses three-stage evaluation to
        deliver the highest possible critique quality through self-improvement
        and meta-reflection.

        Args:
            model: LLM model for the three-stage evaluation process.
                GPT-4o-mini recommended for cost-effectiveness, though
                GPT-4 may provide even higher quality for critical use cases.
            temperature: Generation temperature (0.0-1.0). Moderate values
                (0.6-0.8) balance consistency with creative insight.
            provider: LLM provider (OpenAI, Anthropic, etc.)
            api_key: API key override if not using environment variables
            config: Full Sifaka configuration object

        Example:
            >>> # Standard high-quality critic
            >>> critic = MetaRewardingCritic()
            >>>
            >>> # Premium critic for critical content
            >>> critic = MetaRewardingCritic(model="gpt-4")
            >>>
            >>> # Conservative, consistent evaluation
            >>> critic = MetaRewardingCritic(temperature=0.5)

        Resource planning:
            This critic uses ~3x the tokens of standard critics. Consider
            this when budgeting for high-volume or cost-sensitive applications.

        Note:
            The three-stage process (initial → meta-eval → refined) happens
            within a single critique() call for seamless integration.
        """
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        """Return the unique identifier for this critic.

        Returns:
            "meta_rewarding" - used in configuration, logging, and metadata
        """
        return "meta_rewarding"

    def _get_response_type(self) -> type[BaseModel]:
        """Specify the structured response format for meta-rewarding critique.

        Returns:
            MetaRewardingResponse class providing refined, high-quality
            critique results after three-stage meta-evaluation process.
        """
        return MetaRewardingResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create comprehensive messages for three-stage meta-rewarding critique.

        Builds detailed instructions for the sophisticated evaluation process,
        including initial critique, meta-evaluation, and refined assessment.

        Args:
            text: Text to evaluate using the meta-rewarding process
            result: SifakaResult containing context and history

        Returns:
            List of message dictionaries instructing the LLM to perform
            the complete three-stage meta-rewarding evaluation process

        Note:
            The process is designed to be completed in a single LLM call
            for efficiency, with the model performing all three stages
            internally before providing the final refined critique.
        """
        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""Apply the Meta-Rewarding framework: generate an initial critique, then meta-evaluate it, and finally provide a refined critique.

Text to evaluate:
{text}
{previous_context}

Three-Stage Process:

1. **Initial Critique**: Evaluate the text across these dimensions:
   - Content quality and accuracy
   - Structure and organization
   - Clarity and readability
   - Completeness and thoroughness
   - Engagement and effectiveness

2. **Meta-Evaluation**: Critically assess your initial critique for:
   - Accuracy: Does the critique accurately reflect the text's actual qualities?
   - Helpfulness: Are the suggestions actionable and constructive?
   - Coverage: Have all important aspects been addressed?
   - Fairness: Is the evaluation balanced, neither too harsh nor too lenient?
   - Specificity: Are observations concrete rather than generic?

3. **Refined Critique**: Provide an improved critique that:
   - Addresses any weaknesses identified in the meta-evaluation
   - Incorporates insights from the self-reflection process
   - Delivers the highest quality, most actionable feedback possible

Your final output should represent critique quality enhanced through self-improvement and meta-reflection."""

        return [
            {
                "role": "system",
                "content": "You are a Meta-Rewarding critic that uses three-stage evaluation: initial critique, meta-evaluation of that critique, then refined critique based on meta-rewards. You learn to improve your judgments through self-reflection. Your goal is to deliver the highest possible quality critique through self-improvement.",
            },
            {"role": "user", "content": user_prompt},
        ]
