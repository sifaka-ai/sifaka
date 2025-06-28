"""Constitutional AI critic implementation.

Based on: Constitutional AI: Harmlessness from AI Feedback
Paper: https://arxiv.org/abs/2212.08073
Authors: Bai et al. (2022)

Constitutional AI uses a set of principles to guide AI behavior toward
being helpful, harmless, and honest.
"""

from typing import Optional, Union, List, Dict, TYPE_CHECKING
from pydantic import BaseModel, Field

from ..core.models import SifakaResult, CritiqueResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config

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
    """Simplified response model for Constitutional AI critic."""

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

    When to Use This Critic:
    - ✅ Safety and accuracy are paramount
    - ✅ You have specific principles/guidelines to follow
    - ✅ Ethical considerations matter
    - ✅ Need consistent principle-based evaluation
    - ❌ Creative freedom is the priority
    - ❌ Principles might constrain innovation
    - 🎯 Best for: Public content, educational material, compliance docs
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
        """Initialize critic.

        Args:
            model: LLM model to use
            temperature: Generation temperature
            provider: LLM provider
            api_key: API key override
            config: Full Sifaka config
            principles: Custom constitutional principles (or use defaults)
        """
        super().__init__(model, temperature, config, provider, api_key)
        # Use provided principles or from config or defaults
        if principles is not None:
            self.principles = principles
        elif config and config.constitutional_principles:
            self.principles = config.constitutional_principles
        else:
            self.principles = DEFAULT_PRINCIPLES

    @property
    def name(self) -> str:
        """Return the name of this critic."""
        return "constitutional"

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for the LLM."""
        return [
            {
                "role": "user",
                "content": f"""Evaluate the following text against these constitutional principles:

{chr(10).join(f"{i+1}. {p}" for i, p in enumerate(self.principles))}

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
        """Get the system prompt for constitutional evaluation."""
        return f"""You are a Constitutional AI critic that evaluates text against ethical and quality principles.

Your task is to assess whether text follows these principles:
{chr(10).join(f"- {p}" for p in self.principles)}

Provide practical, actionable feedback focused on the most important improvements.
Don't generate excessive metadata - only what's needed for meaningful improvement."""

    def _get_response_type(self) -> type[BaseModel]:
        """Get the response type for this critic."""
        return ConstitutionalResponse

    async def critique(self, text: str, result: "SifakaResult") -> "CritiqueResult":
        """Critique with principles in metadata."""
        critique_result = await super().critique(text, result)
        # Add principles used to metadata for traceability
        critique_result.metadata["principles_used"] = self.principles
        return critique_result
