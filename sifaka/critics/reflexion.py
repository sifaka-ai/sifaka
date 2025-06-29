"""Reflexion critic - Learning through iterative self-reflection.

Based on: "Reflexion: Language Agents with Verbal Reinforcement Learning"
Paper: https://arxiv.org/abs/2303.11366
Authors: Shinn et al. (2023)

The Reflexion technique enables agents to learn from mistakes through
self-reflection and verbal feedback. This implementation adapts the core
insights for text improvement tasks.

## Key Concepts from the Paper:

The original Reflexion paper introduces verbal reinforcement learning where
agents reflect on task failures and use that reflection to improve future
attempts. We adapt this for text improvement by:

1. **Self-Reflection**: After each iteration, the critic reflects on what
   worked and what didn't, building a memory of the improvement process.

2. **Episodic Memory**: Previous critiques and suggestions are maintained
   as context, allowing the critic to learn from past attempts.

3. **Iterative Refinement**: Each iteration builds on insights from previous
   ones, leading to progressively better text.

## Implementation Choices:

- **Simplified Reward Signal**: Instead of full RL, we use confidence scores
  as an implicit reward signal
- **Context Window**: We keep the last 3 critiques to balance memory with
  context length constraints
- **Focus on Improvement**: Rather than binary success/failure, we focus on
  continuous improvement through constructive feedback

## When to Use Reflexion:

Reflexion excels when:
- Text benefits from iterative refinement over multiple passes
- Previous attempts provide valuable learning signals
- You have budget for 3-5 iterations
- The task involves creative or complex writing

Avoid when:
- You need single-shot improvements
- Historical context isn't relevant
- Quick, simple edits are sufficient
"""

from typing import Optional, Union, List, Dict
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from ..core.config import Config
from .core.base import BaseCritic


class ReflexionResponse(BaseModel):
    """Response model specific to Reflexion critic."""

    feedback: str = Field(
        ..., description="Reflective feedback on the text and its evolution"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Specific improvements based on reflection"
    )
    needs_improvement: bool = Field(
        ..., description="Whether further iterations would benefit the text"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence based on improvement trajectory",
    )


class ReflexionCritic(BaseCritic):
    """Self-reflective critic that learns from previous improvement attempts.
    
    ReflexionCritic implements the core insight from the Reflexion paper:
    agents can improve by reflecting on their previous attempts. In the
    context of text improvement, this means each iteration considers what
    worked and didn't work in previous iterations.
    
    The critic maintains an episodic memory of previous critiques and uses
    this history to provide increasingly refined feedback. This makes it
    particularly effective for complex writing tasks that benefit from
    multiple rounds of revision.
    
    Example:
        >>> # Use reflexion for iterative improvement
        >>> result = await improve(
        ...     "Initial draft of my essay",
        ...     critics=["reflexion"],
        ...     max_iterations=5  # Reflexion benefits from multiple iterations
        ... )
        >>> 
        >>> # The critic will progressively refine its feedback
        >>> for critique in result.critiques:
        ...     if critique.critic == "reflexion":
        ...         print(f"Iteration {critique.iteration}: {critique.feedback}")
    
    Key Features:
        - Maintains context from previous iterations
        - Learns from what has and hasn't worked
        - Provides increasingly targeted feedback
        - Best suited for 3-5 iteration workflows
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        # Initialize with custom config for reflexion
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "reflexion"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom ReflexionResponse for structured output."""
        return ReflexionResponse

    def _get_system_prompt(self) -> str:
        """Get system prompt for Reflexion critic.
        
        The prompt emphasizes the self-reflective nature of the Reflexion
        approach and the importance of learning from previous iterations.
        """
        return "You are an expert text critic using the Reflexion technique for self-improvement through iterative reflection."

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for the LLM using reflection approach."""
        # Build context from previous iterations
        context = self._build_context(result)

        instructions = f"""You are a thoughtful critic using self-reflection to improve text.

Context from previous iterations:
{context}

Reflect on this text and identify:
1. What aspects work well
2. What specific areas need improvement
3. How the text has evolved from previous iterations (if applicable)
4. Concrete suggestions for the next iteration

Focus on being constructive and specific. Analyze the text's strengths, weaknesses, clarity, engagement, and completeness."""

        return await self._simple_critique(text, result, instructions)

    def _build_context(self, result: SifakaResult) -> str:
        """Build episodic memory context from previous iterations.
        
        This method creates a summary of previous critiques and improvements,
        allowing the critic to learn from past attempts. The context helps
        the critic avoid repeating suggestions and recognize patterns in
        the text's evolution.
        
        Args:
            result: The SifakaResult containing critique history
            
        Returns:
            Formatted string summarizing previous iterations
        """
        if not result.critiques:
            return "This is the first iteration. No previous feedback available."

        context_parts = ["Previous reflections and evolution:"]

        # Get last few critiques for context
        # Convert deque to list for slicing
        critiques_list = list(result.critiques)
        recent_critiques = critiques_list[-3:]  # Last 3 critiques

        for i, critique in enumerate(recent_critiques, 1):
            context_parts.append(f"\nIteration {i} ({critique.critic}):")
            context_parts.append(f"  Feedback: {critique.feedback[:200]}...")
            if critique.suggestions:
                context_parts.append(f"  Key suggestion: {critique.suggestions[0]}")

        # Note the evolution
        if len(result.generations) > 1:
            context_parts.append(
                f"\nText has gone through {len(result.generations)} iterations."
            )

        return "\n".join(context_parts)
