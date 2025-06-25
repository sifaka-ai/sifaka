"""Prompt-based critic implementation.

The PromptCritic allows users to define custom evaluation criteria through
natural language prompts, making it highly flexible and adaptable.

## Implementation Choices:
1. User-defined evaluation criteria via prompt
2. Structured JSON response format
3. Context-aware evaluation
4. Configurable confidence calculation

## Why This Approach:
- Maximum flexibility for custom use cases
- No need to create new critic classes
- Natural language interface
- Easy to experiment with different criteria
"""

from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class CustomCriteria(BaseModel):
    """A custom evaluation criterion."""

    criterion: str = Field(..., description="The evaluation criterion")
    assessment: str = Field(..., description="Assessment against this criterion")
    score: float = Field(..., ge=0.0, le=1.0, description="Score for this criterion")
    improvements: list[str] = Field(
        default_factory=list, description="Improvements for this criterion"
    )


class PromptResponse(BaseModel):
    """Response model for custom prompt-based critic."""

    feedback: str = Field(..., description="Overall feedback based on custom criteria")
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    custom_criteria_results: list[CustomCriteria] = Field(
        default_factory=list, description="Results for each custom criterion"
    )
    overall_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Overall score across all criteria"
    )
    key_findings: list[str] = Field(
        default_factory=list, description="Key findings from the custom evaluation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom evaluation data"
    )


class PromptCritic(BaseCritic):
    """Flexible critic that uses custom prompts for evaluation.

    ## When to Use This Critic:

    ✅ When to use:
    - Implementing custom evaluation criteria
    - Domain-specific requirements not covered by other critics
    - Experimenting with new evaluation approaches
    - Highly specialized content types

    ❌ When to avoid:
    - When standard critics already meet your needs
    - If you're unsure about evaluation criteria
    - When consistency across evaluations is critical

    🎯 Best for:
    - Custom business requirements
    - Experimental evaluation criteria
    - Niche content types
    - Rapid prototyping of new critic ideas
    """

    def __init__(
        self,
        custom_prompt: str = "Evaluate this text for quality and suggest improvements.",
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
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "prompt"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom PromptResponse for structured output."""
        return PromptResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages using the custom prompt."""
        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""{self.custom_prompt}

Text to evaluate:
{text}
{previous_context}

Please provide specific, actionable feedback based on the evaluation criteria."""

        return [
            {
                "role": "system",
                "content": "You are a customizable text critic that evaluates based on user-defined criteria.",
            },
            {"role": "user", "content": user_prompt},
        ]


def create_academic_critic(
    model: str = "gpt-4o-mini", temperature: float = 0.7, **kwargs: Any
) -> PromptCritic:
    """Create a critic for academic writing."""
    prompt = """Evaluate this text as an academic paper excerpt. Consider:
1. Clarity and precision of language
2. Logical flow and argumentation
3. Use of evidence and citations
4. Academic tone and style
5. Contribution to the field"""

    return PromptCritic(
        custom_prompt=prompt, model=model, temperature=temperature, **kwargs
    )


def create_business_critic(
    model: str = "gpt-4o-mini", temperature: float = 0.7, **kwargs: Any
) -> PromptCritic:
    """Create a critic for business documents."""
    prompt = """Evaluate this business document. Consider:
1. Clarity of message and call-to-action
2. Professional tone and language
3. Value proposition and benefits
4. Structure and organization
5. Persuasiveness and impact"""

    return PromptCritic(
        custom_prompt=prompt, model=model, temperature=temperature, **kwargs
    )


def create_creative_critic(
    model: str = "gpt-4o-mini", temperature: float = 0.7, **kwargs: Any
) -> PromptCritic:
    """Create a critic for creative writing."""
    prompt = """Evaluate this creative writing. Consider:
1. Narrative flow and pacing
2. Character development and voice
3. Descriptive language and imagery
4. Emotional impact and engagement
5. Originality and creativity"""

    return PromptCritic(
        custom_prompt=prompt, model=model, temperature=temperature, **kwargs
    )
