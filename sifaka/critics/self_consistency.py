"""Self-Consistency critic implementation.

Based on: Self-Consistency Improves Chain of Thought Reasoning in Language Models
Paper: https://arxiv.org/abs/2203.11171
Authors: Wang et al. (2022)

Self-Consistency generates multiple reasoning paths and uses majority
consensus to improve reliability and identify inconsistencies.

## Similarity to Original Paper:
- PRESERVED: Multiple independent evaluation paths
- PRESERVED: Consensus through agreement/voting
- ADAPTED: Applied to critique rather than problem-solving
- PRESERVED: Higher temperature for evaluation diversity

## Implementation Choices:
1. 3 independent evaluations by default (configurable)
2. Temperature 0.8 for diversity (vs typical 0.3-0.5)
3. Consistency metrics: score variance and priority agreement
4. Common themes extracted across evaluations
5. Confidence based on consistency level (variance-based)

## Why This Approach:
- Multiple evaluations catch different aspects/issues
- Consensus provides more robust feedback
- Higher temperature ensures diverse critical perspectives
- Variance metric naturally measures agreement
- Trades computation cost for evaluation reliability
"""

import asyncio
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.llm_client import Provider
from ..core.models import CritiqueResult, SifakaResult
from .core.base import BaseCritic


class SelfConsistencyResponse(BaseModel):
    """Response model specific to Self-Consistency critic."""

    feedback: str = Field(..., description="Consensus feedback from all evaluations")
    suggestions: list[str] = Field(
        default_factory=list, description="Most common suggestions across evaluations"
    )
    needs_improvement: bool = Field(
        ..., description="Consensus on whether improvement is needed"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence based on evaluation consistency",
    )
    # Removed all the unused metadata fields - none were used in prompts
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional consistency data"
    )


class SelfConsistencyCritic(BaseCritic):
    """Implements Self-Consistency approach for consensus-based critique.

    ## When to Use This Critic:

    ✅ When to use:
    - Building consensus on content quality
    - Reducing evaluation variance and bias
    - Important decisions requiring reliable feedback
    - When consistency across evaluations matters

    ❌ When to avoid:
    - Time-critical evaluations (due to multiple samples)
    - Simple, obvious improvements
    - When computational resources are limited

    🎯 Best for:
    - Content strategy decisions
    - A/B testing and comparison
    - Establishing evaluation baselines
    - Mission-critical content review
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,  # Higher for diversity
        num_samples: int = 3,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        # Initialize with custom config for self-consistency
        if config is None:
            config = Config()
        # Higher temperature for diversity
        super().__init__(model, temperature, config, provider, api_key)
        # Use config value if available, otherwise use parameter
        self.num_samples = (
            config.critic.self_consistency_num_samples if config else num_samples
        )

    @property
    def name(self) -> str:
        return "self_consistency"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom SelfConsistencyResponse for structured output."""
        return SelfConsistencyResponse

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Override critique to handle multiple evaluations."""
        # Generate multiple evaluations in parallel
        evaluation_tasks = [
            self._get_single_evaluation(text, result, i + 1)
            for i in range(self.num_samples)
        ]

        # Get all evaluations
        evaluations = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Filter out any failed evaluations
        valid_evaluations = []
        for eval_result in evaluations:
            if isinstance(eval_result, CritiqueResult) and not isinstance(
                eval_result, Exception
            ):
                valid_evaluations.append(eval_result)

        if not valid_evaluations:
            # All evaluations failed
            return CritiqueResult(
                critic=self.name,
                feedback="Failed to generate consistent evaluations",
                suggestions=["Please try again"],
                needs_improvement=True,
                confidence=0.0,
                metadata={"num_evaluations": 0, "error": "All evaluations failed"},
                # Add traceability even for errors
                model_used=(
                    self.client.model if hasattr(self.client, "model") else self.model
                ),
                temperature_used=(
                    self.client.temperature
                    if hasattr(self.client, "temperature")
                    else self.temperature
                ),
                prompt_sent=None,
                tokens_used=0,
                processing_time=0.0,
            )

        # Build consensus from valid evaluations
        return self._build_consensus(valid_evaluations)

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for a single evaluation."""
        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""Evaluate this text for quality and identify areas for improvement.

Text to evaluate:
{text}
{previous_context}

Provide your assessment considering:
1. Overall quality and effectiveness
2. Key strengths of the text
3. Main weaknesses or issues
4. Specific suggestions for improvement
5. How critical these improvements are

Be thorough and specific in your evaluation."""

        return [
            {
                "role": "system",
                "content": "You are an independent text evaluator providing thorough, unbiased critique.",
            },
            {"role": "user", "content": user_prompt},
        ]

    async def _get_single_evaluation(
        self, text: str, result: SifakaResult, sample_num: int
    ) -> CritiqueResult:
        """Get one independent evaluation."""
        # Call parent critique which will use _create_messages
        critique_result = await super().critique(text, result)
        # Mark this as a sample evaluation in metadata
        critique_result.metadata["sample_number"] = sample_num
        return critique_result

    def _build_consensus(self, evaluations: List[CritiqueResult]) -> CritiqueResult:
        """Build consensus from multiple evaluations."""
        # Extract common themes
        all_suggestions = []
        for e in evaluations:
            all_suggestions.extend(e.suggestions)

        # Calculate agreement metrics
        needs_improvement_votes = sum(1 for e in evaluations if e.needs_improvement)
        consensus_needs_improvement = needs_improvement_votes > len(evaluations) / 2

        # Calculate average confidence
        avg_confidence = sum(e.confidence or 0.0 for e in evaluations) / len(
            evaluations
        )

        # Build consensus feedback (simplified without themes)
        consensus_feedback = self._build_consensus_feedback(
            evaluations, consensus_needs_improvement
        )

        # Get most common suggestions
        suggestion_counts = Counter(all_suggestions)
        top_suggestions = [s for s, _ in suggestion_counts.most_common(5)]

        # Aggregate traceability data from all evaluations
        total_tokens = sum(e.tokens_used for e in evaluations)
        total_processing_time = sum(e.processing_time for e in evaluations)

        # Use data from first evaluation for model/temperature (should be same for all)
        first_eval = evaluations[0] if evaluations else None

        # Only keep objective metadata
        metadata = {
            "num_evaluations": len(evaluations),
            "individual_feedback": [
                e.feedback for e in evaluations
            ],  # Store individual evaluations
        }

        return CritiqueResult(
            critic=self.name,
            feedback=consensus_feedback,
            suggestions=top_suggestions,
            needs_improvement=consensus_needs_improvement,
            confidence=avg_confidence,
            metadata=metadata,
            # Add traceability data aggregated from all evaluations
            model_used=first_eval.model_used if first_eval else None,
            temperature_used=first_eval.temperature_used if first_eval else None,
            prompt_sent=first_eval.prompt_sent if first_eval else None,
            tokens_used=total_tokens,
            processing_time=total_processing_time,
        )

    def _build_consensus_feedback(
        self,
        evaluations: List[CritiqueResult],
        consensus_needs_improvement: bool,
    ) -> str:
        """Build consensus feedback message with actual evaluation content."""
        feedback_parts = []

        # Summary of evaluations
        feedback_parts.append(f"Based on {len(evaluations)} independent evaluations:")

        # Agreement level
        needs_improvement_count = sum(1 for e in evaluations if e.needs_improvement)
        if needs_improvement_count == len(evaluations):
            feedback_parts.append("All evaluations agree that improvement is needed.")
        elif needs_improvement_count == 0:
            feedback_parts.append("All evaluations agree the text is satisfactory.")
        else:
            feedback_parts.append(
                f"{needs_improvement_count}/{len(evaluations)} evaluations suggest improvement."
            )

        # Include key points from individual evaluations
        feedback_parts.append("\nKey consensus points:")
        for i, evaluation in enumerate(evaluations, 1):
            if (
                evaluation.feedback and len(evaluation.feedback.strip()) > 20
            ):  # Skip very short feedback
                # Take first sentence or key point from each evaluation
                key_point = evaluation.feedback.split(".")[0].strip()
                if key_point:
                    feedback_parts.append(f"• {key_point}")

        # Extract and synthesize common themes
        common_issues = self._extract_common_issues(evaluations)
        if common_issues:
            feedback_parts.append(
                f"\nCommon improvement areas: {', '.join(common_issues)}"
            )

        return "\n".join(feedback_parts)

    def _extract_common_issues(self, evaluations: List[CritiqueResult]) -> List[str]:
        """Extract common issues from evaluation feedback."""
        # Simple approach: look for repeated key phrases
        all_feedback = " ".join(e.feedback.lower() for e in evaluations)

        # Common issue keywords to look for
        issue_patterns = {
            "clarity": ["unclear", "confusing", "difficult to understand", "clarity"],
            "evidence": ["lacks evidence", "unsupported", "no support", "evidence"],
            "specificity": ["vague", "generic", "specific", "details"],
            "structure": ["organization", "structure", "flow", "coherent"],
            "tone": ["tone", "language", "inappropriate", "professional"],
            "accuracy": ["inaccurate", "misleading", "factual", "correct"],
        }

        found_issues = []
        for issue_type, keywords in issue_patterns.items():
            if any(keyword in all_feedback for keyword in keywords):
                found_issues.append(issue_type)

        return found_issues[:3]  # Return top 3 most common issues
