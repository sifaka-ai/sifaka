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


# Default critic perspectives
DEFAULT_PERSPECTIVES = {
    "Technical Expert": "Focus on accuracy, logic, evidence, and technical correctness",
    "Creative Writer": "Evaluate engagement, flow, style, and narrative quality",
    "Clarity Editor": "Assess clarity, structure, organization, and readability",
    "Target Audience": "Consider audience needs, accessibility, and relevance",
    "Strategic Advisor": "Examine purpose, impact, persuasiveness, and goals",
}


class NCriticsCritic(BaseCritic):
    """Implements N-Critics ensemble approach.

    ## When to Use This Critic:

    ✅ When to use:
    - Evaluating content from multiple stakeholder perspectives
    - Complex documents requiring diverse viewpoints
    - Identifying blind spots in evaluation
    - Balancing technical accuracy with readability

    ❌ When to avoid:
    - Simple, straightforward text improvements
    - When you need deep expertise in one specific area
    - Time-sensitive evaluations (due to multiple perspectives)

    🎯 Best for:
    - Multi-audience content
    - Cross-functional documentation
    - Marketing materials requiring diverse appeal
    - Comprehensive evaluation of complex topics
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        perspectives: Optional[Dict[str, str]] = None,
        auto_generate_perspectives: bool = False,
    ):
        """Initialize N-Critics with configurable perspectives.

        Args:
            model: Model to use for critique
            temperature: Temperature for generation
            provider: LLM provider
            api_key: API key for provider
            config: Configuration object
            perspectives: Custom perspectives dict {role: focus_description}
            auto_generate_perspectives: Whether to generate perspectives based on text
        """
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

        self.perspectives = perspectives or DEFAULT_PERSPECTIVES
        self.auto_generate = auto_generate_perspectives

    @property
    def name(self) -> str:
        return "n_critics"

    async def _generate_perspectives(self, text: str) -> Dict[str, str]:
        """Generate perspectives dynamically based on the text."""
        generation_prompt = f"""Based on this text, identify 3-5 distinct expert perspectives that would provide valuable critique:

Text: {text[:500]}...

For each perspective, provide:
1. A role name (e.g., "Domain Expert", "Style Editor")
2. Their focus area in 1-2 sentences

Format as:
Role Name: Focus description
Role Name: Focus description
..."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at identifying relevant perspectives for text critique.",
            },
            {"role": "user", "content": generation_prompt},
        ]

        response = await self.llm_client.complete(
            messages, temperature=0.7, max_tokens=300
        )

        # Parse the response into perspectives
        perspectives = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                role, focus = line.split(":", 1)
                perspectives[role.strip()] = focus.strip()

        # Fallback to defaults if parsing fails
        return perspectives if len(perspectives) >= 3 else DEFAULT_PERSPECTIVES

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for ensemble critique."""
        # Get perspectives (generate if needed)
        if self.auto_generate:
            perspectives_to_use = await self._generate_perspectives(text)
        else:
            perspectives_to_use = self.perspectives

        # Format perspectives
        perspectives_text = "\n".join(
            f"- **{role}**: {focus}" for role, focus in perspectives_to_use.items()
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
                "content": "You are an N-Critics ensemble that evaluates text from multiple expert perspectives to provide comprehensive feedback.",
            },
            {"role": "user", "content": user_prompt},
        ]
