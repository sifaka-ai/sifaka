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

from typing import List, Type

from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from .core.base import BaseCritic


class ReflexionResponse(BaseModel):
    """Response model specific to Reflexion critic."""

    feedback: str = Field(
        ..., description="Reflective feedback on the text and its evolution"
    )
    suggestions: List[str] = Field(
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
        ...         print(f"Iteration: {critique.feedback}")
    """

    @property
    def name(self) -> str:
        """Return the identifier for this critic."""
        return "reflexion"

    def _get_system_prompt(self) -> str:
        """System prompt that establishes the reflexion approach."""
        return """You are an expert text critic using the Reflexion technique for iterative improvement.

Your approach:
1. Analyze the current text quality
2. Reflect on what has improved from previous versions (if any)
3. Identify remaining areas that need work
4. Provide specific, actionable suggestions

Focus on:
- Learning from past attempts
- Avoiding repetition of previous feedback
- Building on successful improvements
- Identifying persistent issues

Be constructive and specific in your feedback."""

    def _get_response_type(self) -> Type[BaseModel]:
        """Use our custom response model."""
        return ReflexionResponse

    def get_instructions(self, text: str, result: SifakaResult) -> str:
        """Generate reflexion-specific instructions."""
        # Build context from previous iterations
        context = self._build_context(result)

        return f"""You are a thoughtful critic using self-reflection to improve text.

Context from previous iterations:
{context}

Reflect on this text and identify:
1. What aspects work well
2. What specific areas need improvement
3. How the text has evolved from previous iterations (if applicable)
4. Concrete suggestions for the next iteration

Focus on being constructive and specific. Analyze the text's strengths, weaknesses, clarity, engagement, and completeness."""

    def _build_context(self, result: SifakaResult) -> str:
        """Build episodic memory context from previous iterations.

        This method creates a summary of previous critiques and improvements,
        allowing the critic to learn from past attempts. The context helps
        the critic avoid repeating suggestions and recognize patterns in
        the text's evolution.

        Args:
            result: The result object containing all previous iterations

        Returns:
            Formatted context string summarizing the improvement history
        """
        if result.iteration == 0:
            return "This is the first iteration - no previous context."

        context_parts = []

        # Get recent critiques (not just from this critic)
        recent_critiques = list(result.critiques)[-self.config.critic.context_window :]

        for i, critique in enumerate(recent_critiques, 1):
            context_parts.append(f"Iteration {i} ({critique.critic}):")
            context_parts.append(f"- Feedback: {critique.feedback[:200]}...")
            if critique.suggestions:
                context_parts.append(
                    f"- Key suggestion: {critique.suggestions[0][:100]}..."
                )
            context_parts.append(f"- Confidence: {critique.confidence:.2f}")
            context_parts.append("")

        # Add evolution summary if we have multiple generations
        if len(result.generations) > 1:
            context_parts.append("Text evolution:")
            context_parts.append(
                f"- Started with {len(result.original_text.split())} words"
            )
            context_parts.append(
                f"- Now has {len(result.generations[-1].text.split())} words"
            )
            context_parts.append(f"- Has undergone {len(result.generations)} revisions")

        return (
            "\n".join(context_parts)
            if context_parts
            else "No previous context available."
        )
