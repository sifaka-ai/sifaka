"""Meta-Rewarding critic for Sifaka.

This module implements the Meta-Rewarding approach for evaluating and improving
the quality of feedback and critique itself, creating a meta-level assessment
of critique quality and reliability.

Based on "Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge":
https://arxiv.org/abs/2407.19594

@misc{wu2024metarewardinglanguagemodelsself,
      title={Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge},
      author={Tianhao Wu and Weizhe Yuan and Olga Golovneva and Jing Xu and Yuandong Tian and Jiantao Jiao and Jason Weston and Sainbayar Sukhbaatar},
      year={2024},
      eprint={2407.19594},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.19594},
}

The MetaRewardingCritic implements key Meta-Rewarding concepts:
1. Meta-level evaluation of critique quality and reliability
2. Assessment of feedback usefulness and actionability
3. Evaluation of critique consistency and coherence
4. Self-improving feedback through meta-judging

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation adapts the Meta-Rewarding approach for critique evaluation
rather than the full reward model training described in the original paper.
The original Meta-Rewarding paper focuses on training language models to
evaluate and improve their own reward models through meta-judging.

Our implementation focuses on the meta-evaluation aspects, using the model
to assess the quality of critique and feedback rather than training reward
models. This provides a meta-level perspective on critique quality that can
help improve the overall feedback process.

CAVEATS AND LIMITATIONS:
1. This is a meta-critique implementation that evaluates feedback quality
   without the full Meta-Rewarding training pipeline for reward models.
2. The meta-judging is based on the model's assessment of critique quality
   rather than learned reward model evaluation as in the original paper.
3. Self-evaluation bias may affect the quality of meta-assessments, as the
   model is evaluating critique that may come from similar models.
4. The approach may be more computationally expensive as it requires
   meta-level reasoning about critique quality.
5. Performance depends on the model's ability to understand and evaluate
   the characteristics of high-quality feedback and critique.
6. The meta-evaluation may be influenced by the model's own biases about
   what constitutes good critique.

META-EVALUATION FOCUS:
This implementation evaluates critique across several dimensions:
1. Actionability: How specific and implementable are the suggestions?
2. Relevance: How well does the critique address the actual content issues?
3. Consistency: Is the critique internally coherent and non-contradictory?
4. Constructiveness: Does the feedback help improve rather than just criticize?
5. Evidence-based: Is the critique supported by clear reasoning and examples?

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance meta-evaluation
by providing external examples of high-quality critique, feedback guidelines,
or domain-specific evaluation criteria.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic


class MetaRewardingCritic(BaseCritic):
    """Meta-Rewarding critic implementing Wu et al. 2024 methodology.

    This critic evaluates the quality of critique and feedback itself,
    providing meta-level assessment of how useful, actionable, and reliable
    the feedback is for improving text quality.

    Enhanced with validation context awareness to ensure meta-evaluation
    considers how well critique addresses validation requirements.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        meta_evaluation_criteria: Optional[List[str]] = None,
        retrieval_tools: Optional[List[Any]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Meta-Rewarding critic.

        Args:
            model_name: The model name for the PydanticAI agent
            meta_evaluation_criteria: Custom criteria for meta-evaluation (uses defaults if None)
            retrieval_tools: Optional list of retrieval tools for RAG support
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.meta_evaluation_criteria = meta_evaluation_criteria or self._get_default_criteria()
        
        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Wu, T., Yuan, W., Golovneva, O., Xu, J., Tian, Y., Jiao, J., Weston, J., & Sukhbaatar, S. (2024). "
            "Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge. "
            "arXiv preprint arXiv:2407.19594. https://arxiv.org/abs/2407.19594"
        )
        methodology = (
            "Meta-Rewarding methodology: Meta-level evaluation of critique quality and reliability. "
            "Assesses feedback usefulness, actionability, and consistency through meta-judging. "
            "Adapted for critique evaluation without full reward model training."
        )

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            paper_reference=paper_reference,
            methodology=methodology,
            retrieval_tools=retrieval_tools,
            **agent_kwargs,
        )

    def _get_default_criteria(self) -> List[str]:
        """Get the default meta-evaluation criteria."""
        return [
            "Actionability: Are the suggestions specific, clear, and implementable?",
            "Relevance: Does the critique address actual issues in the content?",
            "Consistency: Is the feedback internally coherent and non-contradictory?",
            "Constructiveness: Does the critique help improve rather than just criticize?",
            "Evidence-based: Is the feedback supported by clear reasoning and examples?",
        ]

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Meta-Rewarding critic."""
        criteria_text = "\n".join([f"- {criterion}" for criterion in self.meta_evaluation_criteria])
        
        return f"""You are a Meta-Rewarding critic implementing the methodology from Wu et al. 2024.

Your role is to evaluate the quality of critique and feedback itself, providing
meta-level assessment of how useful, actionable, and reliable the feedback is
for improving text quality.

META-REWARDING METHODOLOGY:
1. Analyze existing critique and feedback for quality and usefulness
2. Assess the actionability and specificity of suggestions
3. Evaluate the consistency and coherence of feedback
4. Determine the constructiveness and helpfulness of critique
5. Provide meta-feedback on how to improve the critique process

META-EVALUATION CRITERIA:
{criteria_text}

RESPONSE FORMAT:
- needs_improvement: boolean indicating if the critique process needs improvement
- message: meta-analysis of critique quality and feedback effectiveness
- suggestions: 1-3 recommendations for improving the critique process
- confidence: float 0.0-1.0 based on meta-assessment certainty
- reasoning: explanation of meta-evaluation process and quality assessment

META-JUDGING PROCESS:
- Evaluate the quality of existing feedback and suggestions
- Assess how well critique addresses the actual content issues
- Determine the actionability and implementability of suggestions
- Consider the consistency and coherence of feedback
- Provide recommendations for improving the critique process

If the existing critique is high-quality and effective, set needs_improvement
to false and explain why the feedback meets quality standards."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for Meta-Rewarding methodology."""
        if not thought.current_text:
            return "No text available for meta-critique evaluation."

        prompt_parts = [
            "META-REWARDING CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            "",
            "TEXT BEING CRITIQUED:",
            thought.current_text,
            "",
        ]

        # Add existing critique and feedback for meta-evaluation
        current_critiques = thought.get_current_iteration_critiques()
        if current_critiques:
            prompt_parts.extend([
                "EXISTING CRITIQUE AND FEEDBACK TO META-EVALUATE:",
                "=" * 55,
            ])
            
            for i, critique in enumerate(current_critiques, 1):
                prompt_parts.extend([
                    f"Critique {i} - {critique.critic}:",
                    f"Feedback: {critique.feedback}",
                    f"Suggestions: {', '.join(critique.suggestions) if critique.suggestions else 'None'}",
                    f"Confidence: {getattr(critique, 'confidence', 'N/A')}",
                    "",
                ])
        else:
            prompt_parts.extend([
                "NO EXISTING CRITIQUE AVAILABLE",
                "=" * 30,
                "This appears to be the first critique iteration.",
                "Meta-evaluation will focus on the need for critique process establishment.",
                "",
            ])

        # Add validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend([
                "VALIDATION CONTEXT:",
                "=" * 20,
                validation_context,
                "",
                "NOTE: Meta-evaluation should assess how well existing critique",
                "addresses validation requirements and provides actionable guidance.",
                "",
            ])

        # Add previous meta-evaluations
        if thought.iteration > 0:
            prev_meta_critiques = [
                c for c in thought.critiques 
                if c.iteration == thought.iteration - 1 and c.critic == "MetaRewardingCritic"
            ]
            if prev_meta_critiques:
                prompt_parts.extend([
                    "PREVIOUS META-EVALUATION:",
                    "=" * 30,
                ])
                for critique in prev_meta_critiques[-1:]:  # Last meta-critique
                    prompt_parts.extend([
                        f"Previous Meta-Assessment: {critique.feedback[:150]}{'...' if len(critique.feedback) > 150 else ''}",
                        f"Previous Meta-Suggestions: {', '.join(critique.suggestions)}",
                        "",
                    ])

        # Add meta-evaluation criteria
        prompt_parts.extend([
            "META-EVALUATION CRITERIA:",
            "=" * 30,
        ])
        for criterion in self.meta_evaluation_criteria:
            prompt_parts.append(f"- {criterion}")
        
        prompt_parts.extend([
            "",
            "META-JUDGING INSTRUCTIONS:",
            "=" * 35,
            "1. Evaluate the quality and usefulness of existing critique",
            "2. Assess the actionability and specificity of suggestions",
            "3. Determine the consistency and coherence of feedback",
            "4. Consider the constructiveness and helpfulness of critique",
            "5. Identify gaps or weaknesses in the critique process",
            "6. Provide recommendations for improving feedback quality",
            "",
            "META-EVALUATION QUESTIONS:",
            "- Are the suggestions specific and implementable?",
            "- Does the critique address the actual content issues?",
            "- Is the feedback consistent and non-contradictory?",
            "- Does the critique help improve rather than just criticize?",
            "- Is the feedback supported by clear reasoning?",
            "",
            "Focus on improving the critique process for better text improvement outcomes.",
        ])

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract Meta-Rewarding-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)
        
        # Add Meta-Rewarding-specific metadata
        meta_rewarding_metadata = {
            "methodology": "meta_rewarding_critique_evaluation",
            "meta_evaluation_performed": True,
            "critique_quality_assessment": "needs_improvement" if feedback.needs_improvement else "satisfactory",
            "meta_criteria_count": len(self.meta_evaluation_criteria),
            "meta_confidence": feedback.confidence,
            "feedback_improvement_focus": feedback.needs_improvement,
            "meta_judging_quality": "high" if feedback.confidence > 0.7 else "medium" if feedback.confidence > 0.4 else "low",
        }
        
        base_metadata.update(meta_rewarding_metadata)
        return base_metadata
