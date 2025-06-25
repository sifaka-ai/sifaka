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

from typing import List, Optional, Union, Dict, Any
from collections import Counter
import asyncio
from pydantic import BaseModel, Field

from ..core.models import SifakaResult, CritiqueResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class ConsistencyEvaluation(BaseModel):
    """Individual evaluation in the consistency check."""

    feedback: str = Field(..., description="Feedback from this evaluation")
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions from this evaluation"
    )
    needs_improvement: bool = Field(
        ..., description="Whether improvement needed per this evaluation"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of this evaluation"
    )
    key_points: list[str] = Field(
        default_factory=list, description="Key points identified"
    )


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
    individual_evaluations: list[ConsistencyEvaluation] = Field(
        default_factory=list, description="Individual evaluation results"
    )
    consistency_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="How consistent the evaluations were"
    )
    common_themes: list[str] = Field(
        default_factory=list,
        description="Themes that appeared across multiple evaluations",
    )
    divergent_points: list[str] = Field(
        default_factory=list, description="Points where evaluations disagreed"
    )
    evaluation_variance: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Variance in evaluation scores"
    )
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
            config.self_consistency_num_samples if config else num_samples
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
        # Use the parent class critique method for a single evaluation
        # Temporarily modify system message for diversity
        self._temp_name = f"{self.name}_sample_{sample_num}"

        try:
            # Call parent critique which will use _create_messages
            critique_result = await super().critique(text, result)
            return critique_result
        finally:
            self._temp_name = self.name

    def _build_consensus(self, evaluations: List[CritiqueResult]) -> CritiqueResult:
        """Build consensus from multiple evaluations."""
        # Extract common themes
        all_feedback = [e.feedback for e in evaluations]
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

        # Extract common themes from feedback
        common_themes = self._extract_common_themes(all_feedback)

        # Build consensus feedback
        consensus_feedback = self._build_consensus_feedback(
            evaluations, common_themes, consensus_needs_improvement
        )

        # Get most common suggestions
        suggestion_counts = Counter(all_suggestions)
        top_suggestions = [s for s, _ in suggestion_counts.most_common(5)]

        # Create metadata
        metadata = {
            "num_evaluations": len(evaluations),
            "agreement_rate": needs_improvement_votes / len(evaluations),
            "common_themes": common_themes[:3],
            "confidence_variance": self._calculate_confidence_variance(evaluations),
        }

        return CritiqueResult(
            critic=self.name,
            feedback=consensus_feedback,
            suggestions=top_suggestions,
            needs_improvement=consensus_needs_improvement,
            confidence=avg_confidence,
            metadata=metadata,
        )

    def _extract_common_themes(self, feedback_list: List[str]) -> List[str]:
        """Extract common themes from feedback."""
        # Simple word frequency analysis
        all_words = []
        for feedback in feedback_list:
            # Basic tokenization
            words = feedback.lower().split()
            # Filter out common words and short words
            filtered_words = [
                w
                for w in words
                if len(w) > 4
                and w
                not in {
                    "about",
                    "text",
                    "this",
                    "that",
                    "with",
                    "from",
                    "have",
                    "been",
                    "will",
                    "would",
                    "could",
                    "should",
                }
            ]
            all_words.extend(filtered_words)

        # Get most common meaningful words
        word_counts = Counter(all_words)
        common_themes = [
            word for word, count in word_counts.most_common(10) if count >= 2
        ]

        return common_themes

    def _build_consensus_feedback(
        self,
        evaluations: List[CritiqueResult],
        common_themes: List[str],
        consensus_needs_improvement: bool,
    ) -> str:
        """Build consensus feedback message."""
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

        # Common themes
        if common_themes:
            feedback_parts.append(
                f"Common themes identified: {', '.join(common_themes[:5])}"
            )

        # Confidence assessment
        confidence_variance = self._calculate_confidence_variance(evaluations)

        if confidence_variance < 0.05:
            feedback_parts.append("Evaluations show high consistency.")
        elif confidence_variance < 0.15:
            feedback_parts.append("Evaluations show moderate consistency.")
        else:
            feedback_parts.append("Evaluations show significant variation.")

        return " ".join(feedback_parts)

    def _calculate_confidence_variance(
        self, evaluations: List[CritiqueResult]
    ) -> float:
        """Calculate variance in confidence scores."""
        if len(evaluations) < 2:
            return 0.0

        confidences = [e.confidence or 0.0 for e in evaluations]
        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(
            confidences
        )

        return variance
