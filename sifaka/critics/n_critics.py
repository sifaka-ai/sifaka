"""N-Critics critic for Sifaka v0.3.0+

This module implements the N-Critics approach for text improvement using PydanticAI
agents with structured output. It uses an ensemble of specialized critics to provide
comprehensive feedback and guide the refinement process.

Based on "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics":
https://arxiv.org/abs/2310.18679

@misc{mousavi2023ncriticsselfrefinementlargelanguage,
      title={N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics},
      author={Sajad Mousavi and Ricardo Luna Gutiérrez and Desik Rengarajan and Vineet Gundecha and Ashwin Ramesh Babu and Avisek Naug and Antonio Guillen and Soumyendu Sarkar},
      year={2023},
      eprint={2310.18679},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.18679},
}

The NCriticsCritic implements key N-Critics concepts:
1. Ensemble of specialized critics with different roles
2. Parallel critique generation for comprehensive feedback
3. Score-based aggregation and consensus building
4. Multi-perspective text evaluation and improvement

Note: This is a simplified implementation that captures core N-Critics principles
without the specialized training or the extensive role-specific fine-tuning
from the original paper.
"""

import time
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


class NCriticsCritic(PydanticAICritic):
    """Modern N-Critics critic using PydanticAI agents with structured output.

    This critic implements the N-Critics technique, which leverages an ensemble
    of specialized critics, each focusing on different aspects of the text,
    to provide comprehensive feedback and guide the refinement process.

    Key features:
    - Structured output using CritiqueFeedback model
    - Ensemble of specialized critics with different roles
    - Parallel critique generation for comprehensive feedback
    - Score-based aggregation and consensus building
    - Multi-perspective text evaluation
    """

    def __init__(
        self,
        model_name: str,
        num_critics: int = 3,
        critic_roles: List[str] = None,
        improvement_threshold: float = 7.0,
        **agent_kwargs,
    ):
        """Initialize the N-Critics critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            num_critics: Number of specialized critics to use.
            critic_roles: List of specialized critic roles/perspectives.
            improvement_threshold: Score threshold below which improvement is needed (default: 7.0).
            **agent_kwargs: Additional arguments passed to the PydanticAI agent.
        """
        # Initialize parent with system prompt
        super().__init__(model_name=model_name, **agent_kwargs)

        self.num_critics = num_critics
        self.improvement_threshold = improvement_threshold

        # Set up critic roles
        self.critic_roles = (
            critic_roles
            or [
                "Content Expert: Focus on factual accuracy, completeness, and relevance of information",
                "Style Editor: Focus on writing style, tone, clarity, and readability",
                "Structure Analyst: Focus on organization, flow, coherence, and logical structure",
                "Audience Specialist: Focus on appropriateness for target audience and effectiveness",
                "Quality Assurance: Focus on overall quality, consistency, and adherence to requirements",
            ][:num_critics]
        )

        logger.info(f"Initialized NCriticsCritic with {num_critics} specialized critics")

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for N-Critics ensemble critique.

        Returns:
            The default system prompt string.
        """
        return f"""You are an expert evaluator using the N-Critics ensemble approach with {self.num_critics} specialized perspectives.

Your task is to evaluate text from multiple specialized viewpoints and provide structured feedback.

You must return a CritiqueFeedback object with these REQUIRED fields:
- message: A clear summary of your ensemble evaluation (string)
- needs_improvement: Whether the text needs improvement (boolean)
- confidence: ConfidenceScore with overall confidence (object with 'overall' field as float 0.0-1.0)
- critic_name: Set this to "NCriticsCritic" (string)

And these OPTIONAL fields (can be empty lists or null):
- violations: List of ViolationReport objects for identified issues
- suggestions: List of ImprovementSuggestion objects for addressing issues
- processing_time_ms: Time taken in milliseconds (can be null)
- critic_version: Version string (can be null)
- metadata: Additional metadata dictionary (can be empty)

Focus on providing comprehensive feedback from multiple specialized perspectives."""

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

        # Create ensemble evaluation prompt
        roles_text = "\n".join([f"- {role}" for role in self.critic_roles])

        return f"""Evaluate the following text using the N-Critics ensemble approach with multiple specialized perspectives.

Original Task: {thought.prompt}

Text to Evaluate:
{thought.text}

Context:
{context}
{validation_text}

Specialized Critic Roles to Consider:
{roles_text}

Please provide a comprehensive critique that considers all {self.num_critics} specialized perspectives:

1. Evaluate the text from each specialized viewpoint
2. Identify issues and problems from multiple angles
3. Provide improvement suggestions from each perspective
4. Determine overall improvement need based on ensemble consensus
5. Assign confidence based on agreement across perspectives

Your evaluation should synthesize feedback from all specialized critics into a cohesive assessment."""

    async def critique_async(self, thought: Thought) -> CriticResult:
        """Critique text using N-Critics ensemble approach with structured output.

        This method overrides the base implementation to perform ensemble critique
        using multiple specialized perspectives for comprehensive evaluation.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A CriticResult with ensemble feedback from multiple specialized critics.
        """
        start_time = time.time()

        with critic_context(
            critic_name="NCriticsCritic",
            operation="critique",
            message_prefix="Failed to critique text with N-Critics ensemble approach",
        ):
            try:
                # Generate individual critiques from each specialized critic
                individual_results = []
                for role in self.critic_roles:
                    try:
                        # Create role-specific critique prompt
                        role_prompt = await self._create_role_specific_prompt(thought, role)

                        # Run PydanticAI agent with structured output for this role
                        result = await self.agent.run(role_prompt)
                        individual_results.append(
                            {
                                "role": role,
                                "feedback": result.output,
                                "score": self._extract_score_from_feedback(result.output),
                            }
                        )

                    except Exception as e:
                        logger.warning(f"N-Critics role '{role}' failed: {e}")
                        continue

                if not individual_results:
                    raise ImproverError(
                        message="All N-Critics ensemble roles failed",
                        component="NCriticsCritic",
                        operation="critique",
                        suggestions=["Check model availability", "Verify prompt format"],
                    )

                # Aggregate results using ensemble consensus
                aggregated_feedback = self._aggregate_ensemble_feedback(individual_results)

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
                        "num_critics": self.num_critics,
                        "successful_roles": len(individual_results),
                        "improvement_threshold": self.improvement_threshold,
                        "individual_results": [
                            {
                                "role": result["role"],
                                "message": result["feedback"].message,
                                "needs_improvement": result["feedback"].needs_improvement,
                                "confidence": (
                                    result["feedback"].confidence.overall
                                    if result["feedback"].confidence
                                    else 0.0
                                ),
                                "score": result["score"],
                            }
                            for result in individual_results
                        ],
                    },
                )

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                logger.error(f"NCriticsCritic critique failed: {e}")

                # Return failed result
                return CriticResult(
                    feedback=CritiqueFeedback(
                        message=f"N-Critics ensemble critique failed: {str(e)}",
                        needs_improvement=True,
                        confidence=ConfidenceScore(overall=0.0),
                        critic_name="NCriticsCritic",
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

    async def _create_role_specific_prompt(self, thought: Thought, role: str) -> str:
        """Create a role-specific critique prompt for a specialized critic.

        Args:
            thought: The thought to critique.
            role: The specialized role/perspective for this critic.

        Returns:
            The formatted role-specific critique prompt.
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Get validation context if available
        validation_text = ""
        if hasattr(thought, "validation_results") and thought.validation_results:
            validation_text = f"\nValidation Context:\n{self._format_validation_context(thought.validation_results)}"

        return f"""You are a specialized critic with the role: {role}

Evaluate the following text from your specialized perspective and provide structured feedback.

Original Task: {thought.prompt}

Text to Evaluate:
{thought.text}

Context:
{context}
{validation_text}

As a {role}, please provide a thorough critique focusing on your area of expertise:

1. Analyze the text quality from your specialized viewpoint
2. Identify specific issues or problems in your domain
3. Provide concrete suggestions for improvement
4. Assess whether improvement is needed from your perspective
5. Rate your confidence in this evaluation

Focus on your specialized area while providing constructive, actionable feedback."""

    def _extract_score_from_feedback(self, feedback: CritiqueFeedback) -> float:
        """Extract a numerical score from CritiqueFeedback for ensemble aggregation.

        Args:
            feedback: The CritiqueFeedback object.

        Returns:
            A numerical score (0.0-10.0) for ensemble aggregation.
        """
        # Use confidence as a proxy for score, scaled to 0-10
        if feedback.confidence:
            base_score = feedback.confidence.overall * 10.0
        else:
            base_score = 5.0  # Default neutral score

        # Adjust based on needs_improvement
        if feedback.needs_improvement:
            # If improvement is needed, cap the score at 6.0
            return min(base_score, 6.0)
        else:
            # If no improvement needed, ensure score is at least 7.0
            return max(base_score, 7.0)

    def _aggregate_ensemble_feedback(self, individual_results: List) -> CritiqueFeedback:
        """Aggregate feedback from multiple specialized critics using ensemble consensus.

        Args:
            individual_results: List of individual critic results with role, feedback, and score.

        Returns:
            Aggregated CritiqueFeedback object with ensemble consensus.
        """
        if not individual_results:
            return CritiqueFeedback(
                message="No feedback available from ensemble",
                needs_improvement=False,
                confidence=ConfidenceScore(overall=0.0),
                critic_name="NCriticsCritic",
            )

        # Collect all violations and suggestions
        all_violations = []
        all_suggestions = []
        improvement_votes = []
        confidence_scores = []
        scores = []

        for result in individual_results:
            feedback = result["feedback"]
            all_violations.extend(feedback.violations)
            all_suggestions.extend(feedback.suggestions)
            improvement_votes.append(feedback.needs_improvement)
            if feedback.confidence:
                confidence_scores.append(feedback.confidence.overall)
            scores.append(result["score"])

        # Calculate ensemble metrics
        average_score = sum(scores) / len(scores) if scores else 5.0
        improvement_ratio = sum(improvement_votes) / len(improvement_votes)
        needs_improvement = improvement_ratio > 0.5 or average_score < self.improvement_threshold

        # Calculate ensemble confidence based on agreement
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        )
        agreement_boost = 1.0 - abs(improvement_ratio - 0.5) * 2  # Higher when closer to consensus
        final_confidence = min(1.0, avg_confidence * (1 + agreement_boost * 0.2))

        # Create ensemble message
        message = self._create_ensemble_message(
            individual_results, average_score, improvement_ratio, final_confidence
        )

        # Find consensus violations and suggestions (simple frequency-based)
        consensus_violations = self._find_consensus_items(all_violations, "violations")
        consensus_suggestions = self._find_consensus_items(all_suggestions, "suggestions")

        return CritiqueFeedback(
            message=message,
            needs_improvement=needs_improvement,
            violations=consensus_violations,
            suggestions=consensus_suggestions,
            confidence=ConfidenceScore(overall=final_confidence),
            critic_name="NCriticsCritic",
            metadata={
                "num_critics": len(individual_results),
                "average_score": average_score,
                "improvement_ratio": improvement_ratio,
                "improvement_threshold": self.improvement_threshold,
                "individual_scores": scores,
                "individual_confidence_scores": confidence_scores,
            },
        )

    def _find_consensus_items(self, items: List, item_type: str) -> List:
        """Find consensus items (violations or suggestions) using frequency-based voting.

        Args:
            items: List of ViolationReport or ImprovementSuggestion objects.
            item_type: Type of items ("violations" or "suggestions").

        Returns:
            List of consensus items.
        """
        if not items:
            return []

        # Group items by description (simple text matching)
        from collections import Counter

        item_counts = Counter(item.description for item in items)
        min_frequency = max(1, len(self.critic_roles) // 2)  # At least half the critics

        consensus_items = []
        for description, count in item_counts.items():
            if count >= min_frequency:
                # Find a representative item with this description
                representative = next(item for item in items if item.description == description)

                # Create new item with consensus metadata
                if item_type == "violations":
                    consensus_items.append(
                        ViolationReport(
                            description=description,
                            severity=representative.severity,
                            location=representative.location,
                            suggestion=representative.suggestion,
                            metadata={
                                "frequency": count,
                                "consensus_ratio": count / len(self.critic_roles),
                            },
                        )
                    )
                else:  # suggestions
                    consensus_items.append(
                        ImprovementSuggestion(
                            description=description,
                            priority=representative.priority,
                            category=representative.category,
                            expected_impact=representative.expected_impact,
                            metadata={
                                "frequency": count,
                                "consensus_ratio": count / len(self.critic_roles),
                            },
                        )
                    )

        return consensus_items

    def _create_ensemble_message(
        self,
        individual_results: List,
        average_score: float,
        improvement_ratio: float,
        confidence: float,
    ) -> str:
        """Create an ensemble message summarizing the aggregated feedback.

        Args:
            individual_results: List of individual critic results.
            average_score: Average score across all critics.
            improvement_ratio: Ratio of critics that suggested improvement.
            confidence: Final confidence score.

        Returns:
            Formatted ensemble message.
        """
        num_critics = len(individual_results)

        message = f"=== N-Critics Ensemble Evaluation ({num_critics} specialized critics) ===\n\n"
        message += f"Average Score: {average_score:.1f}/10\n"
        message += f"Confidence Level: {confidence:.1%}\n"
        message += f"Improvement Agreement: {improvement_ratio:.1%}\n"
        message += f"Improvement Threshold: {self.improvement_threshold}/10\n\n"

        message += "INDIVIDUAL CRITIC ASSESSMENTS:\n"
        for result in individual_results:
            role = result["role"].split(":")[0]  # Get just the role name
            score = result["score"]
            needs_improvement = result["feedback"].needs_improvement
            status = "Needs Improvement" if needs_improvement else "Satisfactory"
            message += f"• {role}: {score:.1f}/10 ({status})\n"

        message += f"\nENSEMBLE CONSENSUS:\n"
        message += f"• Total critics: {num_critics}\n"
        message += f"• Average score: {average_score:.1f}/10\n"
        message += f"• Critics suggesting improvement: {int(improvement_ratio * num_critics)}/{num_critics}\n"
        message += f"• Overall assessment: {'Needs Improvement' if improvement_ratio > 0.5 else 'Satisfactory'}\n"

        message += "\n=== End N-Critics Ensemble Evaluation ==="

        return message

    # Note: The old improve_async and improve_with_validation_context_async methods
    # are not needed in the PydanticAI approach since improvement is handled
    # by the chain/agent architecture, not individual critics.
