"""N-Critics implementation for ensemble critique.

Based on: N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics
Paper: https://arxiv.org/abs/2310.18679
Authors: Chen et al. (2023)

N-Critics uses multiple diverse critical perspectives to provide comprehensive
feedback for text improvement.

## Similarity to Original Paper:
- PRESERVED: Multiple diverse critic perspectives
- PRESERVED: Ensemble approach to critique
- SIMPLIFIED: Fixed set of perspectives vs dynamic generation
- ADAPTED: Focus on text quality vs code generation

## Implementation Choices:
1. 5 fixed perspectives: Technical, Creative, Clarity, Audience, Strategic
2. Each perspective provides focused feedback
3. Aggregates insights across perspectives
4. Balances between depth and breadth of critique

## Why This Approach:
- Multiple perspectives catch different issues
- Reduces blind spots in evaluation
- Provides comprehensive feedback
- Easy to understand and implement
"""

from typing import Optional, Union, List, Dict

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


# Define critic perspectives
CRITIC_PERSPECTIVES = {
    "Technical Expert": "Focus on accuracy, logic, evidence, and technical correctness",
    "Creative Writer": "Evaluate engagement, flow, style, and narrative quality",
    "Clarity Editor": "Assess clarity, structure, organization, and readability",
    "Target Audience": "Consider audience needs, accessibility, and relevance",
    "Strategic Advisor": "Examine purpose, impact, persuasiveness, and goals",
}


class NCriticsCritic(BaseCritic):
    """Implements N-Critics ensemble approach."""

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
        return "n_critics"

    async def _create_messages(self, text: str, result: SifakaResult) -> List[Dict[str, str]]:
        """Create messages for ensemble critique."""
        # Format perspectives
        perspectives_text = "\n".join(
            f"- **{role}**: {focus}"
            for role, focus in CRITIC_PERSPECTIVES.items()
        )
        
        # Get previous context
        previous_context = self._get_previous_context(result)
        
        user_prompt = f"""You are an ensemble of critics providing feedback from multiple perspectives.

Consider these critic perspectives:
{perspectives_text}

Text to evaluate:
{text}
{previous_context}

Provide an integrated critique that:
1. Synthesizes insights from all perspectives
2. Identifies the most critical issues across perspectives
3. Suggests improvements that address multiple concerns
4. Balances different viewpoints constructively

Focus on actionable feedback that considers the text from all angles."""

        return [
            {
                "role": "system",
                "content": "You are an N-Critics ensemble that evaluates text from multiple expert perspectives to provide comprehensive feedback."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]