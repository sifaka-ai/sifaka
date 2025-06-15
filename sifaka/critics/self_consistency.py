"""Self-Consistency critic for Sifaka with validator integration.

This module implements the Self-Consistency approach for improving text evaluation
reliability through multiple critique attempts and consensus building, enhanced
with comprehensive validator integration for validation-aware consistency analysis.

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

The SelfConsistencyCritic implements key Self-Consistency concepts with validator integration:
1. Multiple independent critique attempts with validation awareness
2. Consensus building across different reasoning paths considering validation results
3. Reliability assessment through consistency analysis of both critique and validation
4. Aggregated feedback from diverse perspectives informed by validation constraints
5. Validator-aware consistency scoring and confidence assessment

VALIDATOR INTEGRATION FEATURES:
- Validation-aware consistency analysis across multiple attempts
- Consensus building that considers validation constraints and failures
- Consistency scoring that incorporates validation result reliability
- Validator-informed confidence assessment and suggestion filtering
- Cross-validation consistency checking for improved reliability

IMPORTANT IMPLEMENTATION NOTES AND CAVEATS:

This implementation adapts the Self-Consistency approach from chain-of-thought
reasoning to text critique with comprehensive validator integration. The original
Self-Consistency paper focuses on improving reasoning accuracy by generating multiple
reasoning paths and selecting the most consistent answer through majority voting.

Our implementation applies this concept to critique by generating multiple
independent assessments and building consensus around the most reliable
feedback, enhanced with validation-aware consistency analysis. This helps reduce
the variance and improve the reliability of both critique and validation
compared to single-shot evaluation.

CAVEATS AND LIMITATIONS:
1. This is an adaptation of Self-Consistency from reasoning tasks to critique
   tasks with validator integration, which may not capture all benefits of the original approach.
2. The consensus building is simplified compared to the sophisticated voting
   mechanisms described in the original paper for reasoning tasks.
3. Multiple critique attempts increase computational cost compared to single
   evaluations, which may not always be justified by quality improvements.
4. The approach assumes that consistency indicates correctness, which may not
   always hold for subjective critique tasks or validation edge cases.
5. Self-consistency may reinforce model biases if the same biases appear
   consistently across multiple attempts.
6. The quality of consensus depends on the diversity of reasoning paths,
   which may be limited when using the same model and prompt.
7. Validator integration adds complexity and may introduce additional failure modes.

CONSISTENCY METHODOLOGY WITH VALIDATORS:
This implementation uses multiple critique perspectives with validator integration:
1. Generate multiple independent critique assessments with validation awareness
2. Analyze consistency across different evaluation attempts and validation results
3. Identify areas of agreement and disagreement in both critique and validation
4. Build consensus around the most reliable feedback considering validation constraints
5. Provide confidence based on consistency strength across critique and validation
6. Apply validator-aware filtering to ensure consistency with validation requirements

RETRIEVAL AUGMENTATION:
This critic supports optional retrieval augmentation to enhance consistency
evaluation by providing external context or examples that can inform
multiple critique perspectives and validation-aware analysis.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import time
from statistics import mean, stdev
from collections import Counter

from pydantic_ai import Agent
from pydantic import BaseModel, Field

from sifaka.core.thought import SifakaThought, ValidationContext
from sifaka.critics.base import BaseCritic
from sifaka.models.critic_results import CritiqueFeedback
from sifaka.validators.base import BaseValidator, ValidationResult
from sifaka.utils.errors import CritiqueError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ConsistencyAssessment(BaseModel):
    """Individual consistency assessment result."""

    attempt_number: int
    perspective: str
    needs_improvement: bool
    message: str
    suggestions: List[str]
    confidence: float
    reasoning: str
    validation_aware: bool = False
    validation_consistency_score: Optional[float] = None


class ValidationConsistencyResult(BaseModel):
    """Result of validation consistency analysis."""

    consistent_validators: List[str]
    inconsistent_validators: List[str]
    validation_consensus_score: float
    critical_validation_failures: List[str]
    validation_agreement_patterns: Dict[str, Any]


class SelfConsistencyCritic(BaseCritic):
    """Self-Consistency critic implementing Wang et al. 2022 methodology with validator integration.

    This critic generates multiple independent critique assessments and builds
    consensus around the most consistent and reliable feedback to improve
    critique quality and reliability.

    Enhanced with comprehensive validation context awareness to ensure consistency
    evaluation considers validation requirements across multiple assessment attempts
    and provides validator-aware consistency analysis.

    Key validator integration features:
    - Validation-aware consistency scoring across multiple attempts
    - Cross-validation consistency checking for improved reliability
    - Consensus building that considers validation constraints and failures
    - Validator-informed confidence assessment and suggestion filtering
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-3.5-turbo",
        num_consistency_attempts: int = 3,
        consistency_threshold: float = 0.6,
        validators: Optional[List[BaseValidator]] = None,
        enable_validation_consistency: bool = True,
        validation_weight: float = 0.4,
        retrieval_tools: Optional[List[Any]] = None,
        auto_discover_tools: bool = False,
        tool_categories: Optional[List[str]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the Self-Consistency critic with validator integration.

        Args:
            model_name: The model name for the PydanticAI agent
            num_consistency_attempts: Number of independent critique attempts for consistency
            consistency_threshold: Minimum consistency score for reliable feedback
            validators: Optional list of validators for validation-aware consistency analysis
            enable_validation_consistency: Whether to enable validation consistency checking
            validation_weight: Weight for validation consistency in overall confidence (0.0-1.0)
            retrieval_tools: Optional list of retrieval tools for RAG support
            auto_discover_tools: If True, automatically discover and use all available tools
            tool_categories: Optional list of tool categories to include when auto-discovering
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.num_consistency_attempts = num_consistency_attempts
        self.consistency_threshold = consistency_threshold
        self.validators = validators or []
        self.enable_validation_consistency = enable_validation_consistency
        self.validation_weight = validation_weight

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
        # Handle ConfidenceScore object vs float for confidence value
        confidence_value = feedback.confidence
        if hasattr(confidence_value, "overall"):
            # This is a ConfidenceScore object, extract the overall score
            confidence_float = confidence_value.overall
        else:
            # This is already a float
            confidence_float = confidence_value

        # Create base metadata directly without calling parent class to avoid type issues
        base_metadata = {
            "feedback_length": len(feedback.message),
            "num_suggestions": len(feedback.suggestions),
            "confidence_level": (
                "high" if confidence_float > 0.8 else "medium" if confidence_float > 0.5 else "low"
            ),
        }

        # Add Self-Consistency-specific metadata
        self_consistency_metadata = {
            "methodology": "self_consistency_consensus",
            "num_attempts": self.num_consistency_attempts,
            "consistency_threshold": self.consistency_threshold,
            "consensus_achieved": confidence_float >= self.consistency_threshold,
            "consistency_strength": (
                "strong"
                if confidence_float > 0.8
                else "moderate" if confidence_float > 0.6 else "weak"
            ),
            "reliability_assessment": (
                "high" if confidence_float >= self.consistency_threshold else "low"
            ),
            "multiple_perspectives": True,
        }

        base_metadata.update(self_consistency_metadata)
        return base_metadata

    async def _perform_validation_consistency_analysis(
        self, thought: SifakaThought
    ) -> Optional[ValidationConsistencyResult]:
        """Perform validation consistency analysis across multiple attempts.

        Args:
            thought: The SifakaThought containing validation results

        Returns:
            ValidationConsistencyResult with consistency analysis, or None if no validators
        """
        if not self.validators or not self.enable_validation_consistency:
            return None

        if not thought.current_text:
            return None

        logger.info(
            f"Performing validation consistency analysis with {len(self.validators)} validators",
            extra={
                "critic": "SelfConsistencyCritic",
                "thought_id": thought.id,
                "num_validators": len(self.validators),
                "num_attempts": self.num_consistency_attempts,
            },
        )

        # Perform multiple validation attempts for each validator
        validation_attempts = {}

        for validator in self.validators:
            validator_attempts = []

            for attempt in range(self.num_consistency_attempts):
                try:
                    start_time = time.time()
                    validation_result = await validator.validate_async(thought)
                    processing_time = (time.time() - start_time) * 1000

                    validator_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "result": validation_result,
                            "processing_time_ms": processing_time,
                        }
                    )

                except Exception as e:
                    logger.warning(
                        f"Validation attempt {attempt + 1} failed for {validator.name}",
                        extra={
                            "validator": validator.name,
                            "attempt": attempt + 1,
                            "error": str(e),
                        },
                    )
                    # Create a failed validation result
                    failed_result = validator.create_validation_result(
                        passed=False,
                        message=f"Validation failed: {str(e)}",
                        score=0.0,
                        issues=[f"Validation error: {str(e)}"],
                        suggestions=["Retry validation or check input"],
                    )
                    validator_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "result": failed_result,
                            "processing_time_ms": 0.0,
                            "error": str(e),
                        }
                    )

            validation_attempts[validator.name] = validator_attempts

        # Analyze consistency across attempts
        return self._analyze_validation_consistency(validation_attempts)

    def _analyze_validation_consistency(
        self, validation_attempts: Dict[str, List[Dict[str, Any]]]
    ) -> ValidationConsistencyResult:
        """Analyze consistency across validation attempts.

        Args:
            validation_attempts: Dictionary mapping validator names to attempt results

        Returns:
            ValidationConsistencyResult with consistency analysis
        """
        consistent_validators = []
        inconsistent_validators = []
        critical_validation_failures = []
        validation_agreement_patterns = {}

        total_consistency_scores = []

        for validator_name, attempts in validation_attempts.items():
            # Extract results from attempts
            results = [attempt["result"] for attempt in attempts if "result" in attempt]

            if not results:
                inconsistent_validators.append(validator_name)
                continue

            # Check consistency of pass/fail decisions
            pass_decisions = [result.passed for result in results]
            pass_consistency = len(set(pass_decisions)) == 1  # All same decision

            # Check consistency of scores
            scores = [result.score for result in results if result.score is not None]
            score_consistency = True
            if len(scores) > 1:
                score_std = stdev(scores) if len(scores) > 1 else 0.0
                score_consistency = score_std < 0.2  # Low standard deviation indicates consistency

            # Check consistency of issues and suggestions
            all_issues = [issue for result in results for issue in result.issues]
            all_suggestions = [
                suggestion for result in results for suggestion in result.suggestions
            ]

            issue_patterns = Counter(all_issues)
            suggestion_patterns = Counter(all_suggestions)

            # Calculate overall consistency score for this validator
            consistency_factors = [pass_consistency, score_consistency]
            validator_consistency = sum(consistency_factors) / len(consistency_factors)
            total_consistency_scores.append(validator_consistency)

            # Store agreement patterns
            validation_agreement_patterns[validator_name] = {
                "pass_consistency": pass_consistency,
                "score_consistency": score_consistency,
                "score_std": stdev(scores) if len(scores) > 1 else 0.0,
                "common_issues": dict(issue_patterns.most_common(3)),
                "common_suggestions": dict(suggestion_patterns.most_common(3)),
                "consistency_score": validator_consistency,
            }

            # Categorize validator
            if validator_consistency >= 0.8:
                consistent_validators.append(validator_name)
            else:
                inconsistent_validators.append(validator_name)

            # Check for critical failures
            if any(not result.passed for result in results):
                critical_issues = []
                for result in results:
                    if not result.passed:
                        critical_issues.extend(result.issues)

                if critical_issues:
                    critical_validation_failures.extend(critical_issues)

        # Calculate overall validation consensus score
        validation_consensus_score = (
            mean(total_consistency_scores) if total_consistency_scores else 0.0
        )

        return ValidationConsistencyResult(
            consistent_validators=consistent_validators,
            inconsistent_validators=inconsistent_validators,
            validation_consensus_score=validation_consensus_score,
            critical_validation_failures=list(set(critical_validation_failures)),
            validation_agreement_patterns=validation_agreement_patterns,
        )

    async def critique_async(self, thought: SifakaThought) -> None:
        """Enhanced critique with validation consistency analysis.

        This method performs multiple independent critique attempts and builds
        consensus while also analyzing validation consistency across attempts.

        Args:
            thought: The SifakaThought containing text to critique
        """
        start_time = time.time()
        tools_used = []
        retrieval_context = None

        try:
            # Perform validation consistency analysis if enabled
            validation_consistency = None
            if self.enable_validation_consistency and self.validators:
                validation_consistency = await self._perform_validation_consistency_analysis(
                    thought
                )

            # Perform multiple critique attempts
            critique_attempts = []

            for attempt in range(self.num_consistency_attempts):
                try:
                    # Build critique prompt for this attempt
                    prompt = await self._build_critique_prompt(thought)

                    # Add attempt-specific perspective
                    perspective_prompts = [
                        "Focus on structural analysis: organization, flow, and logical structure",
                        "Focus on content quality: accuracy, relevance, and depth of information",
                        "Focus on audience effectiveness: appropriateness, engagement, and communication impact",
                    ]

                    if attempt < len(perspective_prompts):
                        prompt += (
                            f"\n\nPERSPECTIVE FOR THIS ATTEMPT: {perspective_prompts[attempt]}"
                        )

                    # Run the PydanticAI agent for this attempt
                    result = await self.agent.run(prompt)
                    feedback = result.output

                    # Track tool usage
                    attempt_tools = []
                    if hasattr(result, "tool_calls") and result.tool_calls:
                        attempt_tools = [call.tool_name for call in result.tool_calls]
                        tools_used.extend(attempt_tools)

                        # Extract retrieval context from tool results
                        if not retrieval_context:
                            retrieval_context = self._extract_retrieval_context(result.tool_calls)

                    # Create consistency assessment
                    assessment = ConsistencyAssessment(
                        attempt_number=attempt + 1,
                        perspective=(
                            perspective_prompts[attempt]
                            if attempt < len(perspective_prompts)
                            else f"General perspective {attempt + 1}"
                        ),
                        needs_improvement=feedback.needs_improvement,
                        message=feedback.message,
                        suggestions=feedback.suggestions,
                        confidence=feedback.confidence,
                        reasoning=feedback.reasoning,
                        validation_aware=validation_consistency is not None,
                        validation_consistency_score=(
                            validation_consistency.validation_consensus_score
                            if validation_consistency
                            else None
                        ),
                    )

                    critique_attempts.append(assessment)

                except Exception as e:
                    logger.warning(
                        f"Critique attempt {attempt + 1} failed",
                        extra={
                            "critic": "SelfConsistencyCritic",
                            "attempt": attempt + 1,
                            "error": str(e),
                        },
                    )
                    # Create failed assessment
                    failed_assessment = ConsistencyAssessment(
                        attempt_number=attempt + 1,
                        perspective=f"Failed attempt {attempt + 1}",
                        needs_improvement=True,
                        message=f"Critique attempt failed: {str(e)}",
                        suggestions=["Retry critique or check input"],
                        confidence=0.0,
                        reasoning=f"Error in critique attempt: {str(e)}",
                        validation_aware=False,
                    )
                    critique_attempts.append(failed_assessment)

            # Build consensus from multiple attempts
            consensus_feedback = self._build_consensus_feedback(
                critique_attempts, validation_consistency
            )

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Apply validation-aware filtering to suggestions
            filtered_feedback = self._apply_validation_filtering(thought, consensus_feedback)

            # Extract reasoning from metadata
            reasoning = (
                filtered_feedback.metadata.get("reasoning", "No reasoning provided")
                if filtered_feedback.metadata
                else "No reasoning provided"
            )

            # Extract simple values for thought.add_critique()
            # Convert ImprovementSuggestion objects to strings
            simple_suggestions = []
            for suggestion in filtered_feedback.suggestions:
                if hasattr(suggestion, "suggestion"):
                    # This is an ImprovementSuggestion object
                    simple_suggestions.append(suggestion.suggestion)
                else:
                    # This is already a string
                    simple_suggestions.append(str(suggestion))

            # Extract confidence float from ConfidenceScore
            confidence_float = filtered_feedback.confidence
            if hasattr(confidence_float, "overall"):
                # This is a ConfidenceScore object
                confidence_float = confidence_float.overall

            # Add critique to thought with enhanced metadata
            thought.add_critique(
                critic=self.__class__.__name__,
                feedback=filtered_feedback.message,
                suggestions=simple_suggestions,
                confidence=confidence_float,
                reasoning=reasoning,
                needs_improvement=filtered_feedback.needs_improvement,
                critic_metadata=self._get_enhanced_critic_metadata(
                    filtered_feedback, critique_attempts, validation_consistency
                ),
                processing_time_ms=processing_time_ms,
                model_name=self.model_name,
                paper_reference=self.paper_reference,
                methodology=self.methodology,
                tools_used=list(set(tools_used)),
                retrieval_context=retrieval_context,
            )

        except Exception as e:
            # Handle overall critique failure
            processing_time_ms = (time.time() - start_time) * 1000

            logger.error(
                f"Self-consistency critique failed for thought {thought.id}",
                extra={
                    "critic": "SelfConsistencyCritic",
                    "thought_id": thought.id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            # Add error critique to thought
            error_reasoning = f"Error in self-consistency critique: {str(e)}"

            thought.add_critique(
                critic=self.__class__.__name__,
                feedback=f"Self-consistency critique failed: {str(e)}",
                suggestions=["Retry critique or check input"],
                confidence=0.0,
                reasoning=error_reasoning,
                needs_improvement=True,
                critic_metadata={"error": str(e), "error_type": type(e).__name__},
                processing_time_ms=processing_time_ms,
                model_name=self.model_name,
                paper_reference=self.paper_reference,
                methodology=self.methodology,
                tools_used=tools_used,
                retrieval_context=retrieval_context,
            )

    def _build_consensus_feedback(
        self,
        critique_attempts: List[ConsistencyAssessment],
        validation_consistency: Optional[ValidationConsistencyResult],
    ) -> CritiqueFeedback:
        """Build consensus feedback from multiple critique attempts.

        Args:
            critique_attempts: List of individual critique assessments
            validation_consistency: Optional validation consistency analysis results

        Returns:
            CritiqueFeedback representing the consensus across attempts
        """
        if not critique_attempts:
            return CritiqueFeedback(
                needs_improvement=True,
                message="No successful critique attempts available",
                suggestions=["Retry critique with valid input"],
                confidence=0.0,
                reasoning="No critique attempts completed successfully",
            )

        # Analyze consistency across attempts
        improvement_decisions = [attempt.needs_improvement for attempt in critique_attempts]
        confidence_scores = [attempt.confidence for attempt in critique_attempts]

        # Build consensus on improvement decision (majority vote)
        improvement_votes = Counter(improvement_decisions)
        consensus_needs_improvement = improvement_votes.most_common(1)[0][0]
        improvement_consensus_strength = improvement_votes.most_common(1)[0][1] / len(
            critique_attempts
        )

        # Calculate consensus confidence
        avg_confidence = mean(confidence_scores) if confidence_scores else 0.0
        confidence_consistency = 1.0 - (
            stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0
        )

        # Incorporate validation consistency if available
        validation_boost = 0.0
        if validation_consistency and self.enable_validation_consistency:
            validation_boost = (
                validation_consistency.validation_consensus_score * self.validation_weight
            )

        # Calculate final consensus confidence
        consensus_confidence = min(
            1.0,
            (
                avg_confidence * improvement_consensus_strength * confidence_consistency
                + validation_boost
            ),
        )

        # Aggregate suggestions from consistent attempts
        all_suggestions = []
        for attempt in critique_attempts:
            if attempt.needs_improvement == consensus_needs_improvement:
                all_suggestions.extend(attempt.suggestions)

        # Find most common suggestions
        suggestion_counts = Counter(all_suggestions)
        consensus_suggestions = [
            suggestion
            for suggestion, count in suggestion_counts.most_common(3)
            if count >= max(1, len(critique_attempts) // 2)  # Appear in at least half of attempts
        ]

        # Build consensus message
        consistent_messages = [
            attempt.message
            for attempt in critique_attempts
            if attempt.needs_improvement == consensus_needs_improvement
        ]

        consensus_message = self._synthesize_consensus_message(
            consistent_messages, validation_consistency
        )

        # Build consensus reasoning
        consensus_reasoning = self._build_consensus_reasoning(
            critique_attempts,
            validation_consistency,
            improvement_consensus_strength,
            consensus_confidence,
        )

        # Import required models
        from sifaka.models.critic_results import ConfidenceScore, ImprovementSuggestion

        # Convert string suggestions to ImprovementSuggestion objects
        structured_suggestions = []
        for i, suggestion in enumerate(consensus_suggestions):
            structured_suggestions.append(
                ImprovementSuggestion(
                    suggestion=suggestion,
                    category="consistency",
                    priority=min(i + 1, 10),  # Priority 1-3 based on order
                    confidence=consensus_confidence,
                )
            )

        # Create ConfidenceScore object
        confidence_score = ConfidenceScore(
            overall=consensus_confidence,
            calculation_method="self_consistency_consensus",
            factors_considered=["consistency_strength", "validation_analysis"],
            metadata={
                "num_attempts": len(critique_attempts),
                "consensus_threshold": self.consistency_threshold,
            },
        )

        return CritiqueFeedback(
            needs_improvement=consensus_needs_improvement,
            message=consensus_message,
            suggestions=structured_suggestions,
            confidence=confidence_score,
            critic_name="SelfConsistencyCritic",
            metadata={
                "reasoning": consensus_reasoning,
                "validation_consistency": (
                    validation_consistency.model_dump() if validation_consistency else None
                ),
            },
        )

    def _synthesize_consensus_message(
        self,
        consistent_messages: List[str],
        validation_consistency: Optional[ValidationConsistencyResult],
    ) -> str:
        """Synthesize a consensus message from consistent critique messages.

        Args:
            consistent_messages: Messages from attempts that agree on improvement decision
            validation_consistency: Optional validation consistency analysis results

        Returns:
            Synthesized consensus message
        """
        if not consistent_messages:
            return "No consistent feedback available across critique attempts"

        # Extract common themes from messages
        message_parts = []

        # Add validation consistency information if available
        if validation_consistency:
            if validation_consistency.critical_validation_failures:
                message_parts.append(
                    f"CRITICAL VALIDATION ISSUES: {len(validation_consistency.critical_validation_failures)} "
                    f"critical validation failures detected with {validation_consistency.validation_consensus_score:.2f} "
                    f"consensus score across {len(validation_consistency.consistent_validators)} consistent validators."
                )

            if validation_consistency.inconsistent_validators:
                message_parts.append(
                    f"VALIDATION INCONSISTENCIES: {len(validation_consistency.inconsistent_validators)} "
                    f"validators showed inconsistent results across attempts."
                )

        # Synthesize main consensus message
        if len(consistent_messages) == 1:
            main_message = consistent_messages[0]
        else:
            # Find common themes across messages
            main_message = (
                f"Consensus across {len(consistent_messages)} independent critique attempts: "
                f"Multiple perspectives agree on the assessment. "
                f"Key themes identified consistently across different evaluation approaches."
            )

        message_parts.append(main_message)

        return " ".join(message_parts)

    def _build_consensus_reasoning(
        self,
        critique_attempts: List[ConsistencyAssessment],
        validation_consistency: Optional[ValidationConsistencyResult],
        improvement_consensus_strength: float,
        consensus_confidence: float,
    ) -> str:
        """Build reasoning explanation for the consensus decision.

        Args:
            critique_attempts: All critique attempts
            validation_consistency: Optional validation consistency analysis results
            improvement_consensus_strength: Strength of improvement decision consensus
            consensus_confidence: Final consensus confidence score

        Returns:
            Detailed reasoning explanation
        """
        reasoning_parts = []

        # Self-consistency analysis
        reasoning_parts.append(
            f"Self-consistency analysis across {len(critique_attempts)} independent attempts: "
            f"{improvement_consensus_strength:.1%} agreement on improvement decision. "
        )

        # Confidence analysis
        confidence_scores = [attempt.confidence for attempt in critique_attempts]
        avg_confidence = mean(confidence_scores) if confidence_scores else 0.0
        confidence_std = stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0

        reasoning_parts.append(
            f"Confidence consistency: average {avg_confidence:.2f} with {confidence_std:.2f} standard deviation. "
        )

        # Validation consistency analysis
        if validation_consistency:
            reasoning_parts.append(
                f"Validation consistency: {validation_consistency.validation_consensus_score:.2f} consensus score "
                f"across {len(validation_consistency.consistent_validators)} consistent validators. "
            )

            if validation_consistency.critical_validation_failures:
                reasoning_parts.append(
                    f"Critical validation failures detected: {len(validation_consistency.critical_validation_failures)} issues. "
                )

        # Final consensus assessment
        reasoning_parts.append(
            f"Final consensus confidence: {consensus_confidence:.2f} based on consistency strength and validation analysis."
        )

        return "".join(reasoning_parts)

    def _get_enhanced_critic_metadata(
        self,
        feedback: CritiqueFeedback,
        critique_attempts: List[ConsistencyAssessment],
        validation_consistency: Optional[ValidationConsistencyResult],
    ) -> Dict[str, Any]:
        """Get enhanced metadata including validation consistency information.

        Args:
            feedback: The consensus feedback
            critique_attempts: All critique attempts
            validation_consistency: Optional validation consistency analysis results

        Returns:
            Enhanced metadata dictionary
        """
        base_metadata = self._get_critic_specific_metadata(feedback)

        # Add detailed consistency analysis
        consistency_metadata = {
            "critique_attempts": len(critique_attempts),
            "successful_attempts": len([a for a in critique_attempts if a.confidence > 0]),
            "failed_attempts": len([a for a in critique_attempts if a.confidence == 0]),
            "improvement_agreement": len(
                [a for a in critique_attempts if a.needs_improvement == feedback.needs_improvement]
            ),
            "confidence_range": {
                "min": min([a.confidence for a in critique_attempts]) if critique_attempts else 0.0,
                "max": max([a.confidence for a in critique_attempts]) if critique_attempts else 0.0,
                "avg": (
                    mean([a.confidence for a in critique_attempts]) if critique_attempts else 0.0
                ),
                "std": (
                    stdev([a.confidence for a in critique_attempts])
                    if len(critique_attempts) > 1
                    else 0.0
                ),
            },
            "perspectives_used": [a.perspective for a in critique_attempts],
        }

        # Add validation consistency metadata
        if validation_consistency:
            validation_metadata = {
                "validation_consistency_enabled": True,
                "validation_consensus_score": validation_consistency.validation_consensus_score,
                "consistent_validators": validation_consistency.consistent_validators,
                "inconsistent_validators": validation_consistency.inconsistent_validators,
                "critical_validation_failures": validation_consistency.critical_validation_failures,
                "validation_weight": self.validation_weight,
            }
            consistency_metadata.update(validation_metadata)
        else:
            consistency_metadata["validation_consistency_enabled"] = False

        base_metadata.update(consistency_metadata)
        return base_metadata
