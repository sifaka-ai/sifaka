"""Constitutional AI critic implementation.

Based on: Constitutional AI: Harmlessness from AI Feedback
Paper: https://arxiv.org/abs/2212.08073
Authors: Bai et al. (2022)

Constitutional AI uses a set of principles to guide AI behavior toward
being helpful, harmless, and honest. This implementation applies those
principles to text critique and improvement.

## Similarity to Original Paper:

- PRESERVED: Principle-based evaluation framework
- PRESERVED: Focus on helpfulness, harmlessness, and honesty
- PRESERVED: Explicit principle violation identification
- ADAPTED: Applied to text critique rather than training
- SIMPLIFIED: Uses structured output instead of chain-of-thought

## Implementation Approach:

1. **Principle Definition**: Uses configurable constitutional principles
2. **Violation Detection**: Identifies specific principle violations
3. **Actionable Feedback**: Provides concrete improvement suggestions
4. **Transparency**: Reports which principles were violated and how

## Why This Approach:

- Ensures content aligns with organizational values
- Provides consistent ethical evaluation framework
- Enables customization for different use cases
- Maintains transparency in principle application
- Balances safety with creative expression

## Default Principles:

The critic ships with sensible defaults covering:
- Helpfulness and constructiveness
- Safety and appropriateness
- Accuracy and truth
- Clarity and communication
- Diversity and bias awareness
- Professional tone

These can be customized for specific domains or organizational needs.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.llm_client import Provider
from ..core.models import CritiqueResult, SifakaResult
from .core.base import BaseCritic

if TYPE_CHECKING:
    from ..core.models import SifakaResult


# Default constitutional principles
DEFAULT_PRINCIPLES = [
    "Be helpful and constructive in feedback",
    "Ensure content is safe and appropriate for all audiences",
    "Promote accuracy and avoid misinformation",
    "Encourage clarity and good communication",
    "Respect diverse perspectives and avoid bias",
    "Maintain professional and respectful tone",
]


class ConstitutionalResponse(BaseModel):
    """Structured response model for Constitutional AI critique.

    Provides a clear format for constitutional principle evaluation,
    including specific violations, actionable suggestions, and confidence
    in the assessment.

    Attributes:
        feedback: Overall assessment of constitutional compliance
        suggestions: Specific improvements to address violations
        needs_improvement: Whether any principles were violated
        confidence: Confidence in the constitutional assessment
    """

    feedback: str = Field(..., description="Overall constitutional assessment")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific actionable improvements to align with principles",
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text violates any constitutional principles"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in constitutional assessment",
    )


class ConstitutionalCritic(BaseCritic):
    """Implements Constitutional AI principles for text evaluation.

    Evaluates text against a set of constitutional principles to ensure
    content is helpful, harmless, and honest. Provides specific feedback
    on principle violations with actionable improvement suggestions.

    ## When to Use This Critic:

    ✅ **Best for:**
    - Safety-critical content (public communications, educational materials)
    - Compliance-sensitive documents (legal, medical, financial)
    - Content with ethical implications (social media, marketing)
    - Organizational value alignment (internal communications)
    - Multi-stakeholder content requiring broad acceptance

    ❌ **Avoid when:**
    - Creative expression is the primary goal
    - Principles might unnecessarily constrain innovation
    - Speed is more important than thorough ethical review
    - Working with highly technical content where principles don't apply

    🎯 **Ideal applications:**
    - Public-facing blog posts and marketing content
    - Educational curricula and training materials
    - Policy documents and compliance guides
    - Customer support response templates
    - Content moderation and review processes

    ## Customization:

    The critic supports custom principles for domain-specific needs:
    - Legal compliance principles for law firms
    - Medical ethics principles for healthcare
    - Educational standards for academic content
    - Brand voice principles for marketing teams
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        principles: Optional[List[str]] = None,
    ):
        """Initialize Constitutional AI critic with principles.

        Creates a critic that evaluates text against constitutional principles,
        providing feedback on ethical compliance and value alignment.

        Args:
            model: LLM model to use for evaluation. GPT-4o-mini works well
                for constitutional reasoning tasks.
            temperature: Generation temperature (0.0-1.0). Lower values
                (0.5-0.7) recommended for consistent principle application.
            provider: LLM provider (OpenAI, Anthropic, etc.)
            api_key: API key override if not using environment variables
            config: Full Sifaka configuration object. If provided, principles
                will be read from config.constitutional_principles.
            principles: Custom constitutional principles to use. If None,
                uses principles from config or DEFAULT_PRINCIPLES.

        Example:
            >>> # Use default principles
            >>> critic = ConstitutionalCritic()
            >>>
            >>> # Use custom principles for legal content
            >>> legal_principles = [
            ...     "Ensure accuracy of legal information",
            ...     "Avoid giving specific legal advice",
            ...     "Include appropriate disclaimers"
            ... ]
            >>> critic = ConstitutionalCritic(principles=legal_principles)

        Principle priority:
            1. Explicit principles parameter (highest priority)
            2. Principles from config.constitutional_principles
            3. DEFAULT_PRINCIPLES (fallback)
        """
        super().__init__(model, temperature, config, provider, api_key)
        # Use provided principles or from config or defaults
        if principles is not None:
            self.principles = principles
        elif config and config.critic.constitutional_principles:
            self.principles = config.critic.constitutional_principles
        else:
            self.principles = DEFAULT_PRINCIPLES

    @property
    def name(self) -> str:
        """Return the unique identifier for this critic.

        Returns:
            "constitutional" - used in configuration, logging, and metadata
        """
        return "constitutional"

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create structured messages for constitutional evaluation.

        Builds a detailed prompt that includes the constitutional principles,
        the text to evaluate, and specific instructions for identifying
        violations and providing actionable feedback.

        Args:
            text: Text to evaluate against constitutional principles
            result: SifakaResult with context (currently unused but available)

        Returns:
            List of message dictionaries for the LLM conversation

        Note:
            The prompt emphasizes specific, actionable feedback with principle
            numbers and quotes to ensure transparency and accountability.
        """
        return [
            {
                "role": "user",
                "content": f"""Evaluate the following text against these constitutional principles:

{chr(10).join(f"{i + 1}. {p}" for i, p in enumerate(self.principles))}

Text to evaluate:
{text}

Analyze the text carefully:
1. Identify which specific principles are violated and explain HOW
2. Provide SPECIFIC suggestions to fix each violation
3. Your feedback must reference the principle numbers and quote problematic parts

For example:
- "Violates principle #2 (safety) because the text says 'don't worry about instructions' which could lead to harm"
- "Violates principle #3 (accuracy) because 'more is always better' is false and dangerous"

Be specific and actionable. Generic feedback like 'doesn't align with principles' is not acceptable.""",
            }
        ]

    def _get_system_prompt(self) -> str:
        """Generate system prompt for constitutional evaluation.

        Creates a system prompt that establishes the critic's role and
        includes the specific constitutional principles to apply.

        Returns:
            System prompt string with embedded principles

        Note:
            The prompt focuses on practical, actionable feedback rather
            than academic analysis to ensure useful improvement suggestions.
        """
        return f"""You are a Constitutional AI critic that evaluates text against ethical and quality principles.

Your task is to assess whether text follows these principles:
{chr(10).join(f"- {p}" for p in self.principles)}

Provide practical, actionable feedback focused on the most important improvements.
Don't generate excessive metadata - only what's needed for meaningful improvement."""

    def _get_response_type(self) -> type[BaseModel]:
        """Specify the structured response format.

        Returns:
            ConstitutionalResponse class for type-safe critique results
        """
        return ConstitutionalResponse

    async def critique(self, text: str, result: "SifakaResult") -> "CritiqueResult":
        """Perform constitutional critique with principle tracking.

        Evaluates the text against constitutional principles and adds
        the specific principles used to the result metadata for transparency
        and debugging.

        Args:
            text: Text to evaluate
            result: SifakaResult to store critique in

        Returns:
            CritiqueResult with constitutional feedback and principles used

        Note:
            The principles used are stored in metadata.principles_used for
            full traceability of the evaluation criteria.
        """
        critique_result = await super().critique(text, result)
        # Add principles used to metadata for traceability
        critique_result.metadata["principles_used"] = self.principles
        return critique_result
