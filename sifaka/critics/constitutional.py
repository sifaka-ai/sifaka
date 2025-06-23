"""Constitutional AI critic implementation.

Based on: Constitutional AI: Harmlessness from AI Feedback  
Paper: https://arxiv.org/abs/2212.08073
Authors: Bai et al. (2022)

Constitutional AI uses a set of principles to guide AI behavior toward
being helpful, harmless, and honest.

## Similarity to Original Paper:
- PRESERVED: Use of constitutional principles for evaluation
- PRESERVED: Focus on harmlessness and helpfulness
- SIMPLIFIED: Single-stage critique instead of full RLHF pipeline
- ADAPTED: Principles focused on text quality rather than safety alone

## Implementation Choices:
1. Configurable principles via CriticConfig
2. Balance between safety and quality considerations
3. Structured evaluation against each principle
4. Clear feedback on constitutional violations

## Why This Approach:
- Principle-based evaluation provides clear criteria
- Balances multiple concerns (safety, quality, truthfulness)
- Easy to understand and debug
- Aligns well with text improvement goals
"""

from typing import Optional, Union, List, Dict

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from .core.config import CriticConfig


# Default constitutional principles
DEFAULT_PRINCIPLES = [
    "Be helpful and constructive in feedback",
    "Ensure content is safe and appropriate for all audiences",
    "Promote accuracy and avoid misinformation",
    "Encourage clarity and good communication",
    "Respect diverse perspectives and avoid bias",
    "Maintain professional and respectful tone",
]


class ConstitutionalCritic(BaseCritic):
    """Implements Constitutional AI principles for text evaluation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
        principles: Optional[List[str]] = None,
    ):
        # Initialize with custom config
        if config is None:
            config = CriticConfig(response_format="json")
        
        super().__init__(model, temperature, config, provider, api_key)
        
        # Use provided principles or defaults
        self.principles = principles or DEFAULT_PRINCIPLES

    @property
    def name(self) -> str:
        return "constitutional"

    async def _create_messages(self, text: str, result: SifakaResult) -> List[Dict[str, str]]:
        """Create messages for constitutional evaluation."""
        # Format principles
        principles_text = "\n".join(
            f"{i+1}. {principle}" for i, principle in enumerate(self.principles)
        )
        
        # Get previous context
        previous_context = self._get_previous_context(result)
        
        user_prompt = f"""Please evaluate the following text against these constitutional principles:

{principles_text}

Text to evaluate:
{text}
{previous_context}

For each principle:
1. Assess whether the text aligns with or violates the principle
2. Identify specific issues if any
3. Suggest improvements to better align with the principles

Focus on constructive feedback that helps improve the text while maintaining these standards."""

        return [
            {
                "role": "system",
                "content": "You are a Constitutional AI critic that evaluates text against a set of principles for safety, quality, and effectiveness."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]