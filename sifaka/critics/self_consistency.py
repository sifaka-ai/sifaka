"""Self-Consistency critic for Sifaka.

This module implements the Self-Consistency approach for improving text evaluation
reliability through multiple critique attempts and consensus building.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models":
https://arxiv.org/abs/2203.11171

@misc{wang2022selfconsistencyimproveschainofthought,
      title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
      author={Xuezhi Wang and Jason Wei and Dale Schuurmans and Quoc V. Le and Ed H. Chi and Sharan Narang and Aakanksha Chowdhery and Denny Zhou},
      year={2022},
      eprint={2203.11171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.11171},
}

The SelfConsistencyCritic implements key Self-Consistency concepts:
1. Multiple independent critique attempts
2. Consensus building across different reasoning paths
3. Reliability assessment through consistency analysis
4. Aggregated feedback from diverse perspectives

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation adapts the Self-Consistency approach from chain-of-thought
reasoning to text critique. The original Self-Consistency paper focuses on
improving reasoning accuracy by generating multiple reasoning paths and
selecting the most consistent answer through majority voting.

Our implementation applies this concept to critique by generating multiple
independent assessments and building consensus around the most reliable
feedback. This helps reduce the variance and improve the reliability of
critique compared to single-shot evaluation.

CAVEATS AND LIMITATIONS:
1. This is an adaptation of Self-Consistency from reasoning tasks to critique
   tasks, which may not capture all the benefits of the original approach.
2. The consensus building is simplified compared to the sophisticated voting
   mechanisms described in the original paper for reasoning tasks.
3. Multiple critique attempts increase computational cost compared to single
   evaluations, which may not always be justified by quality improvements.
4. The approach assumes that consistency indicates correctness, which may not
   always hold for subjective critique tasks.
5. Self-consistency may reinforce model biases if the same biases appear
   consistently across multiple attempts.
6. The quality of consensus depends on the diversity of reasoning paths,
   which may be limited when using the same model and prompt.

CONSISTENCY METHODOLOGY:
This implementation uses multiple critique perspectives to build consensus:
1. Generate multiple independent critique assessments
2. Analyze consistency across different evaluation attempts
3. Identify areas of agreement and disagreement
4. Build consensus around the most reliable feedback
5. Provide confidence based on consistency strength

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance consistency
evaluation by providing external context or examples that can inform
multiple critique perspectives.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought
from sifaka.critics.base import BaseCritic


class SelfConsistencyCritic(BaseCritic):
    """Self-Consistency critic implementing Wang et al. 2022 methodology.

    This critic generates multiple independent critique assessments and builds
    consensus around the most consistent and reliable feedback to improve
    critique quality and reliability.

    Enhanced with validation context awareness to ensure consistency evaluation
    considers validation requirements across multiple assessment attempts.
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-3.5-turbo",
        num_consistency_attempts: int = 3,
        consistency_threshold: float = 0.6,
        retrieval_tools: Optional[List[Any]] = None,
        auto_discover_tools: bool = False,
        tool_categories: Optional[List[str]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Self-Consistency critic.

        Args:
            model_name: The model name for the PydanticAI agent
            num_consistency_attempts: Number of independent critique attempts for consistency
            consistency_threshold: Minimum consistency score for reliable feedback
            retrieval_tools: Optional list of retrieval tools for RAG support
            auto_discover_tools: If True, automatically discover and use all available tools
            tool_categories: Optional list of tool categories to include when auto-discovering
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.num_consistency_attempts = num_consistency_attempts
        self.consistency_threshold = consistency_threshold

        system_prompt = self._create_system_prompt()
        paper_reference = (
            "Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). "
            "Self-Consistency Improves Chain of Thought Reasoning in Language Models. "
            "arXiv preprint arXiv:2203.11171. https://arxiv.org/abs/2203.11171"
        )
        methodology = (
            "Self-Consistency methodology: Multiple independent critique attempts with consensus building. "
            "Improves reliability through consistency analysis across diverse evaluation perspectives. "
            "Adapted from reasoning tasks to text critique evaluation."
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

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Self-Consistency critic."""
        return f"""You are a Self-Consistency critic implementing the methodology from Wang et al. 2022.

Your role is to provide reliable critique through multiple independent assessments
and consensus building, improving the consistency and reliability of feedback
compared to single-shot evaluation.

SELF-CONSISTENCY METHODOLOGY:
1. Generate independent critique assessments from different perspectives
2. Analyze the text thoroughly with varied reasoning approaches
3. Identify consistent patterns across multiple evaluation attempts
4. Build consensus around the most reliable and consistent feedback
5. Provide confidence based on consistency strength across attempts

CONSISTENCY EVALUATION PROCESS:
- Approach the text evaluation from multiple angles
- Use different reasoning paths for each assessment attempt
- Look for consistent patterns in quality assessment
- Identify areas where multiple perspectives agree
- Build reliable feedback based on consensus

RESPONSE FORMAT:
- needs_improvement: boolean based on consensus across multiple assessments
- message: comprehensive analysis incorporating consistent findings
- suggestions: 1-3 suggestions that show consistency across evaluation attempts
- confidence: float 0.0-1.0 based on consistency strength and consensus reliability
- reasoning: explanation of consistency analysis and consensus building process

CONFIDENCE SCORING REQUIREMENTS:
- Use the FULL range 0.0-1.0, not just 0.8!
- High confidence (0.9-1.0): Strong consistency across all attempts
- Medium confidence (0.6-0.8): Moderate consistency with some variation
- Low confidence (0.3-0.5): Inconsistent results across attempts
- Very low confidence (0.0-0.2): Highly inconsistent or conflicting assessments

CONSISTENCY PRINCIPLES:
- Generate diverse reasoning paths for the same evaluation task
- Look for agreement patterns across different assessment approaches
- Prioritize feedback that appears consistently across multiple attempts
- Build confidence based on the strength of consensus
- Acknowledge uncertainty when consistency is low

Focus on providing reliable, consistent feedback that improves with multiple
independent assessment attempts and consensus building."""

    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for Self-Consistency methodology."""
        if not thought.current_text:
            return "No text available for self-consistency critique."

        prompt_parts = [
            "SELF-CONSISTENCY CRITIQUE REQUEST",
            "=" * 50,
            "",
            f"Original Task: {thought.prompt}",
            f"Current Iteration: {thought.iteration}",
            f"Consistency Attempts: {self.num_consistency_attempts}",
            f"Consistency Threshold: {self.consistency_threshold}",
            "",
            "TEXT TO EVALUATE:",
            thought.current_text,
            "",
        ]

        # Add validation context
        validation_context = self._get_validation_context(thought)
        if validation_context:
            prompt_parts.extend(
                [
                    "VALIDATION REQUIREMENTS:",
                    "=" * 25,
                    validation_context,
                    "",
                    "NOTE: Consistency evaluation should consider validation requirements",
                    "across all assessment attempts and build consensus around addressing them.",
                    "",
                ]
            )

        # Add previous consistency assessments
        if thought.iteration > 0:
            prev_consistency_critiques = [
                c
                for c in thought.critiques
                if c.iteration == thought.iteration - 1 and c.critic == "SelfConsistencyCritic"
            ]
            if prev_consistency_critiques:
                prompt_parts.extend(
                    [
                        "PREVIOUS CONSISTENCY ASSESSMENT:",
                        "=" * 40,
                    ]
                )
                for critique in prev_consistency_critiques[-1:]:  # Last consistency critique
                    prompt_parts.extend(
                        [
                            f"Previous Consensus: {critique.feedback[:150]}{'...' if len(critique.feedback) > 150 else ''}",
                            f"Previous Consistency: {getattr(critique, 'confidence', 'N/A')}",
                            f"Previous Suggestions: {', '.join(critique.suggestions)}",
                            "",
                        ]
                    )

        # Add consistency evaluation instructions
        prompt_parts.extend(
            [
                "SELF-CONSISTENCY EVALUATION INSTRUCTIONS:",
                "=" * 50,
                f"You will perform {self.num_consistency_attempts} independent assessment attempts:",
                "",
                "ATTEMPT 1 - STRUCTURAL ANALYSIS:",
                "- Focus on organization, flow, and logical structure",
                "- Evaluate clarity and coherence of presentation",
                "- Assess completeness and coverage of the topic",
                "",
                "ATTEMPT 2 - CONTENT QUALITY ANALYSIS:",
                "- Focus on accuracy, relevance, and depth of information",
                "- Evaluate evidence quality and factual correctness",
                "- Assess appropriateness for the intended purpose",
                "",
                "ATTEMPT 3 - AUDIENCE AND EFFECTIVENESS ANALYSIS:",
                "- Focus on audience appropriateness and engagement",
                "- Evaluate tone, style, and communication effectiveness",
                "- Assess overall impact and usefulness",
                "",
                "CONSENSUS BUILDING:",
                "- Identify consistent patterns across all attempts",
                "- Look for agreement in quality assessment and suggestions",
                "- Build consensus around the most reliable feedback",
                f"- Require consistency above {self.consistency_threshold} threshold for high confidence",
                "",
                "Provide final assessment based on consensus across multiple independent attempts.",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_critic_specific_metadata(self, feedback) -> Dict[str, Any]:
        """Extract Self-Consistency-specific metadata."""
        base_metadata = super()._get_critic_specific_metadata(feedback)

        # Add Self-Consistency-specific metadata
        self_consistency_metadata = {
            "methodology": "self_consistency_consensus",
            "num_attempts": self.num_consistency_attempts,
            "consistency_threshold": self.consistency_threshold,
            "consensus_achieved": feedback.confidence >= self.consistency_threshold,
            "consistency_strength": (
                "strong"
                if feedback.confidence > 0.8
                else "moderate" if feedback.confidence > 0.6 else "weak"
            ),
            "reliability_assessment": (
                "high" if feedback.confidence >= self.consistency_threshold else "low"
            ),
            "multiple_perspectives": True,
        }

        base_metadata.update(self_consistency_metadata)
        return base_metadata
