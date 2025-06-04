"""Self-Refine critic for Sifaka.

This module implements the Self-Refine approach for iterative text improvement
through self-feedback and refinement cycles.

Based on "Self-Refine: Iterative Refinement with Self-Feedback":
https://arxiv.org/abs/2303.17651

@misc{madaan2023selfrefineiterativerefinementselffeedback,
      title={Self-Refine: Iterative Refinement with Self-Feedback},
      author={Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Wen-tau Yih and Hannaneh Hajishirzi},
      year={2023},
      eprint={2303.17651},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2303.17651},
}

The SelfRefineCritic implements the core Self-Refine algorithm:
1. Iterative refinement through self-feedback
2. Multi-round critique and revision cycles
3. Self-generated improvement suggestions
4. Convergence detection for stopping criteria

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation follows the original Self-Refine paper closely,
using a simple FEEDBACK → REFINE → FEEDBACK loop without additional
learning mechanisms that were not part of the original research.

The original Self-Refine paper demonstrates that large language models can
improve their outputs through iterative self-feedback without additional
training or fine-tuning. The approach is model-agnostic and relies on
the model's inherent ability to critique and improve its own outputs.

CAVEATS AND LIMITATIONS:
1. Performance is heavily dependent on the underlying model's self-critique
   capabilities - smaller or less capable models may not benefit significantly.
2. The approach can sometimes lead to over-refinement where the model makes
   unnecessary changes that don't improve quality.
3. Self-feedback may be biased toward the model's own preferences rather
   than objective quality measures.
4. The method works best for tasks where quality can be assessed through
   self-evaluation (e.g., writing, reasoning) rather than tasks requiring
   external validation.
5. Convergence detection is challenging - the model may continue suggesting
   changes even when the text is already high quality.
6. The approach may amplify existing model biases through the iterative
   refinement process.

METHODOLOGY ADAPTATION:
Our implementation adapts Self-Refine for integration with Sifaka's validation
and multi-critic framework. We maintain the core self-feedback loop while
adding validation context awareness and structured output formatting.

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance self-refinement
by providing external examples, best practices, or domain-specific knowledge
during the self-critique process.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic


class SelfRefineCritic(BaseCritic):
    """Self-Refine critic implementing Madaan et al. 2023 methodology.

    This critic uses the Self-Refine approach to iteratively improve text through
    self-critique and revision. It uses the same language model to critique its
    own output and then suggest refinements based on that critique.

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting self-refinement suggestions.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        refinement_focus: Optional[List[str]] = None,
        retrieval_tools: Optional[List[Any]] = None,
        auto_discover_tools: bool = False,
        tool_categories: Optional[List[str]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Self-Refine critic.

        Args:
            model_name: The model name for the PydanticAI agent
            refinement_focus: Specific areas to focus refinement on (uses defaults if None)
            retrieval_tools: Optional list of retrieval tools for RAG support
            auto_discover_tools: If True, automatically discover and use all available tools
            tool_categories: Optional list of tool categories to include when auto-discovering
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.refinement_focus = refinement_focus or self._get_default_focus_areas()

        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., ... & Hajishirzi, H. (2023). "
            "Self-Refine: Iterative Refinement with Self-Feedback. "
            "arXiv preprint arXiv:2303.17651. https://arxiv.org/abs/2303.17651"
        )
        methodology = (
            "Self-Refine methodology: Iterative refinement through self-feedback loops. "
            "Model critiques its own output and generates targeted improvements. "
            "Focuses on self-generated feedback without external training."
        )

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            auto_discover_tools=auto_discover_tools,
            tool_categories=tool_categories,
            **agent_kwargs,
        )

    def _get_default_focus_areas(self) -> List[str]:
        """Get the default refinement focus areas."""
        return [
            "Content Quality: Accuracy, completeness, and relevance of information",
            "Structure and Flow: Logical organization and smooth transitions between ideas",
            "Clarity and Readability: Clear expression and appropriate language for the audience",
            "Coherence: Consistency of arguments and elimination of contradictions",
            "Conciseness: Removal of redundancy while maintaining necessary detail",
        ]

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Self-Refine critic."""
        focus_text = "\n".join([f"- {area}" for area in self.refinement_focus])

        return f"""You are a Self-Refine critic implementing the methodology from Madaan et al. 2023.

Your role is to provide self-feedback on text and suggest iterative refinements
to improve quality through multiple rounds of critique and revision.

SELF-REFINE METHODOLOGY:
1. Analyze the current text with a critical, self-reflective perspective
2. Identify specific areas where the text could be improved
3. Generate concrete, actionable refinement suggestions
4. Focus on iterative improvement rather than complete rewrites
5. Consider the refinement history to avoid repetitive suggestions

REFINEMENT FOCUS AREAS:
{focus_text}

RESPONSE FORMAT:
- needs_improvement: boolean indicating if refinements would improve the text
- message: detailed self-critique with specific areas for improvement
- suggestions: 1-3 concrete, actionable refinement suggestions
- confidence: float 0.0-1.0 based on self-assessment certainty
  * 0.9-1.0: Very confident in refinement needs or text quality
  * 0.7-0.8: Moderately confident in assessment
  * 0.5-0.6: Somewhat uncertain about refinement direction
  * 0.3-0.4: Low confidence in assessment
  * 0.0-0.2: Very uncertain or conflicting self-assessment
- reasoning: explanation of the self-refinement analysis process

SELF-REFINEMENT PRINCIPLES:
- Be constructively critical of the current text
- Focus on incremental improvements rather than major overhauls
- Provide specific, actionable suggestions that can be implemented
- Consider the cumulative effect of multiple refinement rounds
- Balance thoroughness with conciseness

If the text is already well-refined and further changes would not improve quality,
set needs_improvement to false and explain why no further refinement is beneficial."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for Self-Refine methodology."""
        if not thought.current_text:
            return "No text available for self-refinement critique."

        prompt_parts = [
            "SELF-REFINE CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            f"Refinement Round: {thought.iteration + 1}",
            "",
            "TEXT TO REFINE:",
            thought.current_text,
            "",
        ]

        # Add refinement history from previous iterations
        if thought.iteration > 0:
            prompt_parts.extend(
                [
                    "REFINEMENT HISTORY:",
                    "=" * 20,
                    "Previous refinement attempts and outcomes:",
                    "",
                ]
            )

            # Show progression through iterations
            for i in range(min(thought.iteration, 3)):  # Limit to last 3 iterations
                iteration_generations = [g for g in thought.generations if g.iteration == i]
                iteration_critiques = [
                    c
                    for c in thought.critiques
                    if c.iteration == i and c.critic == "SelfRefineCritic"
                ]

                if iteration_generations and iteration_critiques:
                    gen = iteration_generations[-1]
                    critique = iteration_critiques[-1]

                    prompt_parts.extend(
                        [
                            f"Iteration {i}:",
                            f"Text: {gen.text[:150]}{'...' if len(gen.text) > 150 else ''}",
                            f"Self-Critique: {critique.feedback[:100]}{'...' if len(critique.feedback) > 100 else ''}",
                            f"Suggestions Applied: {len(critique.suggestions)} suggestions",
                            "",
                        ]
                    )

        # Add validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend(
                [
                    "VALIDATION REQUIREMENTS:",
                    "=" * 25,
                    validation_context,
                    "",
                    "NOTE: Self-refinement should prioritize addressing validation failures",
                    "while also improving overall text quality.",
                    "",
                ]
            )

        # Add current refinement focus
        prompt_parts.extend(
            [
                "CURRENT REFINEMENT FOCUS:",
                "=" * 30,
            ]
        )
        for area in self.refinement_focus:
            prompt_parts.append(f"- {area}")

        prompt_parts.extend(
            [
                "",
                "SELF-REFINEMENT INSTRUCTIONS:",
                "=" * 35,
                "1. Critically analyze the current text from a self-improvement perspective",
                "2. Identify specific areas where refinement would add value",
                "3. Consider the refinement history to avoid repetitive suggestions",
                "4. Generate concrete, implementable improvement suggestions",
                "5. Assess whether further refinement is beneficial or if the text is sufficiently refined",
                "6. Set confidence based on:",
                "   - Clarity of refinement needs (clear needs = higher confidence)",
                "   - Quality of current text (poor quality = higher confidence in improvement)",
                "   - Certainty of suggestions (specific improvements = higher confidence)",
                "   - Assessment ambiguity (unclear cases = lower confidence)",
                "",
                "CONFIDENCE SCORING REQUIREMENTS:",
                "- Use the FULL range 0.0-1.0, not just 0.8!",
                "- High confidence (0.9-1.0): Very clear refinement needs or excellent text quality",
                "- Medium confidence (0.6-0.8): Moderate refinement needs or good text quality",
                "- Low confidence (0.3-0.5): Uncertain refinement direction or mixed quality",
                "- Very low confidence (0.0-0.2): Highly uncertain or conflicting assessment",
                "",
                "Focus on iterative improvement that builds on previous refinement rounds.",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract Self-Refine-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)

        # Add Self-Refine-specific metadata
        self_refine_metadata = {
            "methodology": "self_refine_iterative",
            "refinement_focus_areas": len(self.refinement_focus),
            "refinement_needed": feedback.needs_improvement,
            "refinement_confidence": feedback.confidence,
            "iterative_improvement": True,  # Always true for Self-Refine
            "self_critique_quality": (
                "high"
                if feedback.confidence > 0.7
                else "medium" if feedback.confidence > 0.4 else "low"
            ),
        }

        base_metadata.update(self_refine_metadata)
        return base_metadata
