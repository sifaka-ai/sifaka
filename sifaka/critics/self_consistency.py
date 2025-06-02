"""Self-Consistency critic for Sifaka v0.3.0+

This module implements a Self-Consistency approach for text critique and improvement
using PydanticAI agents with structured output. Multiple critiques are generated for
the same text and majority voting is used to determine the most reliable feedback.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models":
https://arxiv.org/abs/2203.11171

@misc{wang2022selfconsistency,
      title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
      author={Xuezhi Wang and Jason Wei and Dale Schuurmans and Quoc Le and Ed Chi and Sharan Narang and Aakanksha Chowdhery and Denny Zhou},
      year={2022},
      eprint={2203.11171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.11171},
}

The SelfConsistencyCritic implements the core Self-Consistency algorithm:
1. Multiple critique generation for the same text using chain-of-thought reasoning
2. Majority voting to select the most consistent feedback
3. Confidence scoring based on agreement level

Note: This implementation follows the original Self-Consistency paper closely,
using simple majority voting over multiple reasoning paths without additional
learning mechanisms that were not part of the original research.
"""

import time
from collections import Counter
from typing import List

from sifaka.core.thought import Thought
from sifaka.critics.base_pydantic import PydanticAICritic
from sifaka.models.critic_results import (
    CriticResult,
    CritiqueFeedback,
    ConfidenceScore,
    ViolationReport,
    ImprovementSuggestion,
)

from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SelfConsistencyCritic(PydanticAICritic):
    """Modern Self-Consistency critic using PydanticAI agents with structured output.

    This critic implements the Self-Consistency approach which generates multiple critiques
    of the same text and uses consensus to determine the most reliable feedback.
    This improves critique reliability by reducing the impact of single inconsistent
    or low-quality critiques.

    Key features:
    - Structured output using CritiqueFeedback model
    - Multiple critique generation with consensus building
    - Majority voting for reliable feedback
    - Confidence scoring based on agreement level
    - Validation context awareness
    """

    def __init__(
        self,
        model_name: str,
        num_iterations: int = 5,
        consensus_threshold: float = 0.5,
        use_chain_of_thought: bool = True,
        **agent_kwargs,
    ):
        """Initialize the Self-Consistency critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            num_iterations: Number of critique iterations to generate (default: 5).
            consensus_threshold: Minimum agreement ratio for consensus (default: 0.5).
            use_chain_of_thought: Whether to use chain-of-thought prompting.
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        # Configuration parameters
        self.num_iterations = max(3, num_iterations)  # Minimum 3 for meaningful consensus
        self.consensus_threshold = max(
            0.1, min(1.0, consensus_threshold)
        )  # Clamp between 0.1 and 1.0
        self.use_chain_of_thought = use_chain_of_thought

        logger.info(f"Initialized SelfConsistencyCritic with {num_iterations} iterations")

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for Self-Consistency critique.

        Returns:
            The default system prompt string.
        """
        cot_instruction = ""
        if self.use_chain_of_thought:
            cot_instruction = " Use step-by-step reasoning and provide detailed analysis."

        return f"""You are an expert evaluator providing detailed, constructive feedback on text quality using Self-Consistency methodology.

Your task is to evaluate text through multiple perspectives and provide structured feedback.{cot_instruction}

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your evaluation (string)
- needs_improvement: Whether the text needs improvement (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "SelfConsistencyCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues
- suggestions: List of ImprovementSuggestion objects for addressing issues
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

Focus on providing consistent, reliable feedback that can be aggregated with other evaluations."""

    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for the given thought.

        Args:
            thought: The thought to critique.

        Returns:
            The formatted critique prompt.
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Get validation context if available
        validation_text = ""
        if hasattr(thought, "validation_results") and thought.validation_results:
            validation_text = f"\nValidation Context:\n{self._format_validation_context(thought.validation_results)}"

        cot_instruction = ""
        if self.use_chain_of_thought:
            cot_instruction = (
                " Think step-by-step and provide detailed reasoning for your evaluation."
            )

        return f"""Evaluate the following text for quality, accuracy, and areas for improvement.

Original Task: {thought.prompt}

Text to Evaluate:
{thought.text}

Context:
{context}
{validation_text}

Please provide a thorough critique.{cot_instruction}

Your evaluation should include:
1. Analysis of text quality and effectiveness
2. Identification of specific issues or problems
3. Concrete suggestions for improvement
4. Assessment of whether improvement is needed
5. Confidence level in your evaluation

Remember: This evaluation will be combined with others using Self-Consistency methodology, so focus on providing reliable, consistent feedback."""

    async def critique_async(self, thought: Thought) -> CriticResult:
        """Critique text using Self-Consistency approach with structured output.

        This method overrides the base implementation to perform multiple critiques
        and aggregate them using majority voting for improved reliability.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A CriticResult with aggregated feedback from multiple evaluations.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfConsistencyCritic",
            operation="critique",
            message_prefix="Failed to critique text with Self-Consistency approach",
        ):
            try:
                # Generate multiple critiques using PydanticAI agent
                individual_results = []
                for i in range(self.num_iterations):
                    try:
                        # Create critique prompt for this iteration
                        critique_prompt = await self._create_critique_prompt(thought)

                        # Run PydanticAI agent with structured output
                        result = await self.agent.run(critique_prompt)
                        individual_results.append(result.output)

                    except Exception as e:
                        logger.warning(f"Self-consistency iteration {i + 1} failed: {e}")
                        continue

                if not individual_results:
                    raise ImproverError(
                        message="All self-consistency iterations failed",
                        component="SelfConsistencyCritic",
                        operation="critique",
                        suggestions=["Check model availability", "Verify prompt format"],
                    )

                # Aggregate results using majority voting
                aggregated_feedback = self._aggregate_feedback(individual_results)

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000

                # Create CriticResult with aggregated feedback
                return CriticResult(
                    feedback=aggregated_feedback,
                    operation_type="critique",
                    success=True,
                    total_processing_time_ms=processing_time,
                    model_calls=len(individual_results),
                    input_text_length=len(thought.text),
                    validation_context=self._get_validation_context_dict(thought),
                    metadata={
                        "model_name": self.model_name,
                        "critic_name": self.__class__.__name__,
                        "num_iterations": self.num_iterations,
                        "successful_iterations": len(individual_results),
                        "consensus_threshold": self.consensus_threshold,
                        "use_chain_of_thought": self.use_chain_of_thought,
                        "individual_results": [
                            {
                                "message": result.message,
                                "needs_improvement": result.needs_improvement,
                                "confidence": (
                                    result.confidence.overall if result.confidence else 0.0
                                ),
                            }
                            for result in individual_results
                        ],
                    },
                )

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                logger.error(f"SelfConsistencyCritic critique failed: {e}")

                # Return failed result
                return CriticResult(
                    feedback=CritiqueFeedback(
                        message=f"Self-consistency critique failed: {str(e)}",
                        needs_improvement=True,
                        confidence=ConfidenceScore(overall=0.0),
                        critic_name="SelfConsistencyCritic",
                    ),
                    operation_type="critique",
                    success=False,
                    error_message=str(e),
                    total_processing_time_ms=processing_time,
                    model_calls=0,
                    input_text_length=len(thought.text),
                    validation_context=self._get_validation_context_dict(thought),
                    metadata={
                        "model_name": self.model_name,
                        "critic_name": self.__class__.__name__,
                        "error_type": type(e).__name__,
                    },
                )

    def _aggregate_feedback(self, individual_results: List[CritiqueFeedback]) -> CritiqueFeedback:
        """Aggregate multiple CritiqueFeedback objects using majority voting.

        Args:
            individual_results: List of individual critique feedback objects.

        Returns:
            Aggregated CritiqueFeedback object with consensus information.
        """
        if not individual_results:
            return CritiqueFeedback(
                message="No feedback available",
                needs_improvement=False,
                confidence=ConfidenceScore(overall=0.0),
                critic_name="SelfConsistencyCritic",
            )

        # Collect all violations and suggestions
        all_violations = []
        all_suggestions = []
        improvement_votes = []
        confidence_scores = []

        for result in individual_results:
            all_violations.extend(result.violations)
            all_suggestions.extend(result.suggestions)
            improvement_votes.append(result.needs_improvement)
            if result.confidence:
                confidence_scores.append(result.confidence.overall)

        # Find consensus violations and suggestions
        consensus_violations = self._find_consensus_violations(all_violations)
        consensus_suggestions = self._find_consensus_suggestions(all_suggestions)

        # Determine improvement need by majority vote
        improvement_ratio = sum(improvement_votes) / len(improvement_votes)
        needs_improvement = improvement_ratio > 0.5

        # Calculate aggregated confidence
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        )

        # Boost confidence based on consensus
        consensus_boost = len(consensus_violations + consensus_suggestions) / max(
            1, len(all_violations + all_suggestions)
        )
        final_confidence = min(1.0, avg_confidence * (1 + consensus_boost * 0.2))

        # Create consensus message
        message = self._create_consensus_message(
            individual_results,
            consensus_violations,
            consensus_suggestions,
            improvement_ratio,
            final_confidence,
        )

        return CritiqueFeedback(
            message=message,
            needs_improvement=needs_improvement,
            violations=consensus_violations,
            suggestions=consensus_suggestions,
            confidence=ConfidenceScore(overall=final_confidence),
            critic_name="SelfConsistencyCritic",
            metadata={
                "num_iterations": len(individual_results),
                "improvement_ratio": improvement_ratio,
                "consensus_threshold": self.consensus_threshold,
                "individual_confidence_scores": confidence_scores,
            },
        )

    def _find_consensus_violations(
        self, all_violations: List[ViolationReport]
    ) -> List[ViolationReport]:
        """Find consensus violations using frequency-based majority voting.

        Args:
            all_violations: List of all violation reports from individual critiques.

        Returns:
            List of consensus violation reports.
        """
        if not all_violations:
            return []

        # Group violations by description (simple text matching)
        violation_counts = Counter(v.description for v in all_violations)
        min_frequency = max(1, int(self.num_iterations * self.consensus_threshold))

        consensus_violations = []
        for description, count in violation_counts.items():
            if count >= min_frequency:
                # Find a representative violation with this description
                representative = next(v for v in all_violations if v.description == description)
                consensus_violations.append(
                    ViolationReport(
                        description=description,
                        severity=representative.severity,
                        location=representative.location,
                        suggestion=representative.suggestion,
                        metadata={
                            "frequency": count,
                            "consensus_ratio": count / self.num_iterations,
                        },
                    )
                )

        return consensus_violations

    def _find_consensus_suggestions(
        self, all_suggestions: List[ImprovementSuggestion]
    ) -> List[ImprovementSuggestion]:
        """Find consensus suggestions using frequency-based majority voting.

        Args:
            all_suggestions: List of all improvement suggestions from individual critiques.

        Returns:
            List of consensus improvement suggestions.
        """
        if not all_suggestions:
            return []

        # Group suggestions by description (simple text matching)
        suggestion_counts = Counter(s.description for s in all_suggestions)
        min_frequency = max(1, int(self.num_iterations * self.consensus_threshold))

        consensus_suggestions = []
        for description, count in suggestion_counts.items():
            if count >= min_frequency:
                # Find a representative suggestion with this description
                representative = next(s for s in all_suggestions if s.description == description)
                consensus_suggestions.append(
                    ImprovementSuggestion(
                        description=description,
                        priority=representative.priority,
                        category=representative.category,
                        expected_impact=representative.expected_impact,
                        metadata={
                            "frequency": count,
                            "consensus_ratio": count / self.num_iterations,
                        },
                    )
                )

        return consensus_suggestions

    def _create_consensus_message(
        self,
        individual_results: List[CritiqueFeedback],
        consensus_violations: List[ViolationReport],
        consensus_suggestions: List[ImprovementSuggestion],
        improvement_ratio: float,
        confidence: float,
    ) -> str:
        """Create a consensus message summarizing the aggregated feedback.

        Args:
            individual_results: List of individual critique results.
            consensus_violations: List of consensus violations.
            consensus_suggestions: List of consensus suggestions.
            improvement_ratio: Ratio of critics that suggested improvement.
            confidence: Final confidence score.

        Returns:
            Formatted consensus message.
        """
        num_iterations = len(individual_results)

        message = f"=== Self-Consistency Evaluation ({num_iterations} iterations) ===\n\n"
        message += f"Confidence Level: {confidence:.1%}\n"
        message += f"Consensus Threshold: {self.consensus_threshold:.1%}\n"
        message += f"Improvement Agreement: {improvement_ratio:.1%}\n\n"

        if consensus_violations:
            message += "CONSENSUS VIOLATIONS (found in multiple evaluations):\n"
            for violation in consensus_violations:
                freq_ratio = violation.metadata.get("consensus_ratio", 0.0)
                message += f"• {violation.description} (found in {freq_ratio:.0%} of evaluations)\n"
            message += "\n"

        if consensus_suggestions:
            message += "CONSENSUS SUGGESTIONS (found in multiple evaluations):\n"
            for suggestion in consensus_suggestions:
                freq_ratio = suggestion.metadata.get("consensus_ratio", 0.0)
                message += (
                    f"• {suggestion.description} (found in {freq_ratio:.0%} of evaluations)\n"
                )
            message += "\n"

        message += "EVALUATION SUMMARY:\n"
        message += f"• Total evaluations: {num_iterations}\n"
        message += f"• Consensus violations: {len(consensus_violations)}\n"
        message += f"• Consensus suggestions: {len(consensus_suggestions)}\n"
        message += f"• Agreement on improvement need: {improvement_ratio:.1%}\n"

        message += "\n=== End Self-Consistency Evaluation ==="

        return message

    # Note: The old improve_async and improve_with_validation_context_async methods
    # are not needed in the PydanticAI approach since improvement is handled
    # by the chain/agent architecture, not individual critics.
