"""Reflexion critic implementation.

Based on: Reflexion: Language Agents with Verbal Reinforcement Learning
Paper: https://arxiv.org/abs/2303.11366
Authors: Shinn et al. (2023)

The Reflexion technique enables agents to learn from mistakes through
self-reflection and verbal feedback.

## Similarity to Original Paper:
- SIMPLIFIED: Original uses full RL with rewards; we use confidence scores
- PRESERVED: Core concept of learning from reflection history
- PRESERVED: Episodic memory via critique history tracking
- ADAPTED: Task completion → text improvement focus

## Implementation Choices:
1. Context window of 3 previous critiques (balances memory vs context length)
2. No actual RL machinery - confidence serves as implicit reward signal
3. Builds context from previous iterations to simulate episodic memory
4. Focuses on constructive feedback rather than task success/failure

## Why This Approach:
- Text improvement doesn't need full RL framework
- Reflection on previous attempts is the key insight we preserve
- Simpler implementation while maintaining iterative learning concept
- Works well with other critics in ensemble scenarios
"""

from typing import Optional, Union

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, create_prompt_with_format


class ReflexionCritic(BaseCritic):
    """Implements Reflexion self-reflection for text improvement."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for reflexion
        if config is None:
            config = CriticConfig(
                response_format="json",
                # Use default values from CriticConfig
            )
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "reflexion"
    
    def get_system_prompt(self) -> str:
        """Get the system prompt used by this critic."""
        return "You are an expert text critic using the Reflexion technique for self-improvement."

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate critique using reflection approach."""
        # Build context from previous iterations
        context = self._build_context(result)

        # Get previous feedback to avoid repetition
        previous_context = self._get_previous_feedback_context(result)
        
        # Create reflection prompt
        base_prompt = f"""You are a thoughtful critic using self-reflection to improve text.

Context from previous iterations:
{context}
{previous_context}

Current text to evaluate:
{text}

Reflect on this text and identify:
1. What aspects work well
2. What specific areas need improvement
3. How the text has evolved from previous iterations (if applicable)
4. Concrete suggestions for the next iteration

Focus on being constructive and specific. Analyze the text's strengths, weaknesses, clarity, engagement, and completeness."""

        prompt = create_prompt_with_format(
            base_prompt, self.config.response_format, include_examples=True
        )

        # Get reflection from model
        # Use the LLM client to get completion
        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert text critic using the Reflexion technique for self-improvement.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return response.content

    def _build_context(self, result: SifakaResult) -> str:
        """Build context from previous iterations."""
        if not result.critiques:
            return "This is the first iteration. No previous feedback available."

        context_parts = ["Previous reflections and evolution:"]

        # Get last few critiques for context
        # Convert deque to list for slicing
        critiques_list = list(result.critiques)
        recent_critiques = critiques_list[-self.config.context_window_size:]  # Use config value
        for i, critique in enumerate(recent_critiques, 1):
            context_parts.append(f"\nIteration {i} ({critique.critic}):")
            context_parts.append(f"  Feedback: {critique.feedback[:self.config.feedback_truncation_length]}...")
            if critique.suggestions:
                context_parts.append(f"  Key suggestion: {critique.suggestions[0]}")

        # Note the evolution
        if len(result.generations) > 1:
            context_parts.append(
                f"\nText has gone through {len(result.generations)} iterations."
            )

        return "\n".join(context_parts)
