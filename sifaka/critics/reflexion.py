"""Reflexion critic for Sifaka.

This module implements a Reflexion-inspired approach for iterative improvement
through trial-and-error learning with verbal reinforcement using PydanticAI agents
with structured output.

Based on "Reflexion: Language Agents with Verbal Reinforcement Learning":
https://arxiv.org/abs/2303.11366

@misc{shinn2023reflexionlanguageagentsverbal,
      title={Reflexion: Language Agents with Verbal Reinforcement Learning},
      author={Noah Shinn and Federico Cassano and Edward Berman and Ashwin Gopinath and Karthik Narasimhan and Shunyu Yao},
      year={2023},
      eprint={2303.11366},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2303.11366},
}

The ReflexionCritic implements key Reflexion concepts:
1. Trial-based learning with episodic memory
2. Task performance feedback integration
3. Self-reflection on failures and successes
4. Verbal reinforcement for future attempts

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation adapts the core Reflexion principle-based evaluation
for text critique using Sifaka's Thought infrastructure. The original
Reflexion paper focuses on multi-agent Actor/Evaluator/Self-Reflection architecture
for reinforcement learning from AI feedback.

Our implementation focuses on the critique and self-reflection aspects without the
full multi-agent training component, making it suitable for real-time text improvement
rather than model training. We leverage Sifaka's Thought system for episodic memory
instead of duplicating memory infrastructure.

CAVEATS AND LIMITATIONS:
1. This is a simplified implementation that captures core Reflexion principles
   without the full multi-agent Actor/Evaluator/Self-Reflection architecture.
2. The original paper uses environment-specific reward signals; we adapt this
   to text quality assessment which is more subjective.
3. Memory persistence is handled by Sifaka's thought system rather than
   the specialized memory buffer described in the original paper.
4. The reflection mechanism is simplified to focus on text improvement
   rather than the complex action-space exploration in the original work.
5. Performance may vary significantly depending on the underlying language model's
   capability for self-reflection and critique.

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance reflection quality
by providing external context, examples, or domain-specific knowledge during
the critique process.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic


class ReflexionCritic(BaseCritic):
    """Reflexion critic implementing Shinn et al. 2023 methodology.

    This critic implements the Reflexion approach for improving text through
    self-reflection and iterative learning from past attempts. It performs a critique,
    reflects on the critique to identify specific improvements, and then generates
    improvement suggestions.

    The process involves:
    1. Analyzing the current text for quality and accuracy
    2. Reflecting on previous attempts and their outcomes
    3. Identifying specific patterns of success and failure
    4. Generating targeted improvement suggestions based on reflection

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting reflection suggestions.
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        retrieval_tools: Optional[List[Any]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Reflexion critic.

        Args:
            model_name: The model name for the PydanticAI agent
            retrieval_tools: Optional list of retrieval tools for RAG support
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). "
            "Reflexion: Language Agents with Verbal Reinforcement Learning. "
            "arXiv preprint arXiv:2303.11366. https://arxiv.org/abs/2303.11366"
        )
        methodology = (
            "Reflexion methodology: Trial-and-error learning with verbal reinforcement. "
            "Uses episodic memory to reflect on past attempts and generate targeted improvements. "
            "Adapted for text critique with simplified reflection mechanism."
        )

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            **agent_kwargs,
        )

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Reflexion critic."""
        return """You are a Reflexion critic implementing the methodology from Shinn et al. 2023.

Your role is to analyze text through self-reflection and provide targeted improvement suggestions
based on patterns of success and failure from previous attempts.

REFLEXION METHODOLOGY:
1. Analyze the current text for quality, accuracy, and effectiveness
2. Reflect on the history of attempts and their outcomes
3. Identify patterns of what works and what doesn't work
4. Generate specific, actionable improvement suggestions
5. Provide confidence assessment based on reflection quality

RESPONSE FORMAT:
- needs_improvement: boolean indicating if text needs improvement
- message: detailed analysis with reflection on patterns and outcomes
- suggestions: 1-3 specific, actionable improvement suggestions
- confidence: float 0.0-1.0 based on reflection quality and pattern clarity
- reasoning: explanation of the reflection process and pattern identification

FOCUS AREAS:
- Content accuracy and factual correctness
- Logical flow and reasoning quality
- Clarity and comprehensibility
- Completeness and thoroughness
- Learning from previous iteration patterns

Be constructive and specific. If no improvements are needed, set needs_improvement to false
and explain why the text meets quality standards based on reflection."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for Reflexion methodology."""
        if not thought.current_text:
            return "No text available for critique."

        prompt_parts = [
            "REFLEXION CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            f"Max Iterations: {thought.max_iterations}",
            "",
            "TEXT TO CRITIQUE:",
            thought.current_text,
            "",
        ]

        # Add episodic memory from previous iterations
        if thought.iteration > 0:
            prompt_parts.extend(
                [
                    "EPISODIC MEMORY - PREVIOUS ATTEMPTS:",
                    "=" * 40,
                ]
            )

            # Add previous generations and their outcomes
            for i in range(thought.iteration):
                iteration_generations = [g for g in thought.generations if g.iteration == i]
                iteration_validations = [v for v in thought.validations if v.iteration == i]
                iteration_critiques = [c for c in thought.critiques if c.iteration == i]

                if iteration_generations:
                    gen = iteration_generations[-1]  # Get the last generation for this iteration
                    prompt_parts.extend(
                        [
                            f"Iteration {i}:",
                            f"Text: {gen.text[:200]}{'...' if len(gen.text) > 200 else ''}",
                        ]
                    )

                    # Add validation outcomes
                    if iteration_validations:
                        passed = all(v.passed for v in iteration_validations)
                        prompt_parts.append(f"Validation: {'PASSED' if passed else 'FAILED'}")
                        if not passed:
                            failures = [v for v in iteration_validations if not v.passed]
                            for failure in failures[:2]:  # Limit to 2 failures
                                prompt_parts.append(f"  - {failure.validator}: {failure.details}")

                    # Add critique outcomes
                    if iteration_critiques:
                        total_suggestions = sum(len(c.suggestions) for c in iteration_critiques)
                        prompt_parts.append(
                            f"Critique: {total_suggestions} suggestions from {len(iteration_critiques)} critics"
                        )

                    prompt_parts.append("")

        # Add current validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend(
                [
                    "CURRENT VALIDATION CONTEXT:",
                    "=" * 30,
                    validation_context,
                    "",
                ]
            )

        # Add reflection instructions
        prompt_parts.extend(
            [
                "REFLECTION INSTRUCTIONS:",
                "=" * 25,
                "1. Analyze patterns from previous attempts (if any)",
                "2. Identify what approaches worked vs. failed",
                "3. Assess current text quality against task requirements",
                "4. Generate targeted improvements based on reflection",
                "5. Provide confidence based on pattern clarity",
                "",
                "Focus on learning from the episodic memory to avoid repeating past mistakes",
                "and build on successful patterns from previous iterations.",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract Reflexion-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)

        # Add Reflexion-specific metadata
        reflexion_metadata = {
            "reflection_quality": (
                "high"
                if feedback.confidence > 0.7
                else "medium" if feedback.confidence > 0.4 else "low"
            ),
            "methodology": "reflexion_verbal_reinforcement",
            "episodic_memory_used": True,  # Always true for Reflexion
            "pattern_analysis": len(feedback.suggestions)
            > 0,  # Indicates pattern-based suggestions
        }

        base_metadata.update(reflexion_metadata)
        return base_metadata
