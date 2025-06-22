"""Base confidence calculation utilities for critics."""

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import re


class ConfidenceCalculator(ABC):
    """Abstract base class for confidence calculation strategies."""

    @abstractmethod
    def calculate(self, **kwargs: Any) -> float:
        """Calculate confidence score based on evaluation factors."""
        pass


class BaseConfidenceCalculator(ConfidenceCalculator):
    """Base confidence calculator with common factors used across critics."""

    def __init__(self, base_confidence: float = 0.6):
        """Initialize with base confidence level.

        Args:
            base_confidence: Starting confidence level (0.0-1.0)
        """
        self.base_confidence = max(0.1, min(0.9, base_confidence))

    def calculate(
        self,
        feedback: str = "",
        suggestions: Optional[List[str]] = None,
        evaluation_text: str = "",
        structured_response: bool = False,
        specific_indicators: Optional[List[str]] = None,
        actionable_indicators: Optional[List[str]] = None,
        domain_indicators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> float:
        """Calculate confidence based on common evaluation quality factors.

        Args:
            feedback: The main feedback/assessment text
            suggestions: List of improvement suggestions
            evaluation_text: Full evaluation response text
            structured_response: Whether response followed expected structure
            specific_indicators: Words indicating specific/detailed analysis
            actionable_indicators: Words indicating actionable suggestions
            domain_indicators: Domain-specific terminology
            **kwargs: Additional factors for extensibility

        Returns:
            Confidence score between 0.1 and 0.95
        """
        suggestions = suggestions or []
        specific_indicators = specific_indicators or self._default_specific_indicators()
        actionable_indicators = (
            actionable_indicators or self._default_actionable_indicators()
        )
        domain_indicators = domain_indicators or []

        confidence = self.base_confidence

        # Factor 1: Response structure (0.0 to 0.1 bonus)
        confidence += self._calculate_structure_factor(
            evaluation_text, structured_response
        )

        # Factor 2: Feedback specificity (0.0 to 0.15 bonus)
        confidence += self._calculate_specificity_factor(feedback, specific_indicators)

        # Factor 3: Suggestion quality (0.0 to 0.15 bonus)
        confidence += self._calculate_suggestion_quality(
            suggestions, actionable_indicators
        )

        # Factor 4: Response completeness (0.0 to 0.1 bonus)
        confidence += self._calculate_completeness_factor(feedback, suggestions)

        # Factor 5: Domain expertise (0.0 to 0.1 bonus)
        confidence += self._calculate_domain_expertise(feedback, domain_indicators)

        # Factor 6: Length appropriateness (0.0 to 0.05 bonus)
        confidence += self._calculate_length_factor(feedback, suggestions)

        # Apply any custom factors from kwargs
        confidence += self._apply_custom_factors(**kwargs)

        # Ensure bounds
        return max(0.1, min(0.95, confidence))

    def _calculate_structure_factor(
        self, evaluation_text: str, structured_response: bool
    ) -> float:
        """Calculate bonus for structured response format."""
        if structured_response:
            return 0.1

        # Check for common structure patterns
        structure_patterns = [
            r"(EVALUATION|ASSESSMENT|ANALYSIS):\s*",
            r"(SUGGESTIONS|RECOMMENDATIONS):\s*",
            r"(VIOLATIONS|ISSUES|PROBLEMS):\s*",
            r"(CONFIDENCE|CERTAINTY):\s*[\d.]+",
        ]

        matches = sum(
            1
            for pattern in structure_patterns
            if re.search(pattern, evaluation_text, re.IGNORECASE)
        )

        return min(0.1, matches * 0.025)

    def _calculate_specificity_factor(
        self, feedback: str, specific_indicators: List[str]
    ) -> float:
        """Calculate bonus for specific, detailed feedback."""
        if not feedback:
            return 0.0

        feedback_lower = feedback.lower()

        # Count specific indicators
        specific_count = sum(
            1 for indicator in specific_indicators if indicator in feedback_lower
        )

        # Count vague indicators (negative factor)
        vague_indicators = [
            "somewhat",
            "might",
            "could",
            "perhaps",
            "possibly",
            "maybe",
        ]
        vague_count = sum(
            1 for indicator in vague_indicators if indicator in feedback_lower
        )

        # Calculate net specificity
        words_count = len(feedback.split())
        if words_count == 0:
            return 0.0

        specificity_ratio = (specific_count - vague_count * 0.5) / max(
            1, words_count / 20
        )
        return max(0.0, min(0.15, specificity_ratio * 0.3))

    def _calculate_suggestion_quality(
        self, suggestions: List[str], actionable_indicators: List[str]
    ) -> float:
        """Calculate bonus for actionable, specific suggestions."""
        if not suggestions:
            return 0.0

        actionable_count = 0
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if any(
                indicator in suggestion_lower for indicator in actionable_indicators
            ):
                actionable_count += 1

        # Bonus for high ratio of actionable suggestions
        actionable_ratio = actionable_count / len(suggestions)
        return min(0.15, actionable_ratio * 0.15)

    def _calculate_completeness_factor(
        self, feedback: str, suggestions: List[str]
    ) -> float:
        """Calculate bonus for complete responses."""
        completeness = 0.0

        # Bonus for substantial feedback
        if feedback and len(feedback.split()) >= 10:
            completeness += 0.05

        # Bonus for having suggestions
        if suggestions and len(suggestions) >= 1:
            completeness += 0.03

        # Small bonus for multiple suggestions (but not too many)
        if 2 <= len(suggestions) <= 5:
            completeness += 0.02

        return min(0.1, completeness)

    def _calculate_domain_expertise(
        self, feedback: str, domain_indicators: List[str]
    ) -> float:
        """Calculate bonus for domain-specific expertise indicators."""
        if not domain_indicators or not feedback:
            return 0.0

        feedback_lower = feedback.lower()
        domain_count = sum(
            1 for indicator in domain_indicators if indicator in feedback_lower
        )

        return min(0.1, domain_count * 0.02)

    def _calculate_length_factor(self, feedback: str, suggestions: List[str]) -> float:
        """Calculate small bonus for appropriate response length."""
        feedback_words = len(feedback.split()) if feedback else 0
        suggestion_words = sum(len(s.split()) for s in suggestions)
        total_words = feedback_words + suggestion_words

        # Sweet spot is 20-100 words
        if 20 <= total_words <= 100:
            return 0.05
        elif 10 <= total_words <= 150:
            return 0.02
        else:
            return 0.0

    def _apply_custom_factors(self, **kwargs: Any) -> float:
        """Apply any custom confidence factors passed in kwargs."""
        custom_bonus = 0.0

        # Extract numeric factors with 'factor' in name
        for key, value in kwargs.items():
            if "factor" in key.lower() and isinstance(value, (int, float)):
                custom_bonus += max(-0.1, min(0.1, value))

        return max(-0.2, min(0.2, custom_bonus))

    def _default_specific_indicators(self) -> List[str]:
        """Default indicators of specific, detailed analysis."""
        return [
            "specifically",
            "precisely",
            "exactly",
            "clearly",
            "explicitly",
            "particularly",
            "notably",
            "distinctly",
            "definitively",
            "obviously",
            "evidently",
            "demonstrably",
            "undoubtedly",
            "certainly",
            "definitely",
        ]

    def _default_actionable_indicators(self) -> List[str]:
        """Default indicators of actionable suggestions."""
        return [
            "add",
            "remove",
            "change",
            "replace",
            "revise",
            "rewrite",
            "improve",
            "enhance",
            "clarify",
            "specify",
            "expand",
            "reduce",
            "strengthen",
            "modify",
            "adjust",
            "restructure",
            "reorganize",
        ]


class StructuredConfidenceCalculator(BaseConfidenceCalculator):
    """Confidence calculator for critics using structured/Pydantic outputs."""

    def calculate(
        self,
        feedback: str = "",
        suggestions: Optional[List[str]] = None,
        evaluation_text: str = "",
        structured_response: bool = False,
        specific_indicators: Optional[List[str]] = None,
        actionable_indicators: Optional[List[str]] = None,
        domain_indicators: Optional[List[str]] = None,
        overall_confidence: Optional[float] = None,
        evaluation_quality: Optional[int] = None,
        violation_confidences: Optional[List[float]] = None,
        principle_scores: Optional[Dict[int, int]] = None,
        **kwargs: Any,
    ) -> float:
        """Calculate confidence for structured evaluations.

        Args:
            overall_confidence: LLM's self-assessed confidence
            evaluation_quality: LLM's self-assessed quality (1-5)
            violation_confidences: Individual violation confidence scores
            principle_scores: Scores for each principle/criterion
            **kwargs: Additional factors passed to base calculator
        """
        violation_confidences = violation_confidences or []
        principle_scores = principle_scores or {}

        # Start with LLM's self-assessment if available
        if overall_confidence is not None:
            base_confidence = overall_confidence
        else:
            base_confidence = self.base_confidence

        confidence = base_confidence

        # Factor 1: Evaluation quality self-assessment (-0.1 to +0.1)
        if evaluation_quality is not None:
            quality_factor = (evaluation_quality - 3) * 0.05
            confidence += quality_factor

        # Factor 2: Violation confidence consistency (0.0 to 0.1 bonus)
        if violation_confidences:
            avg_violation_confidence = sum(violation_confidences) / len(
                violation_confidences
            )
            # High average suggests careful analysis
            confidence += min(0.1, avg_violation_confidence * 0.1)

        # Factor 3: Score distribution variety (0.0 to 0.1 bonus)
        if principle_scores:
            scores = list(principle_scores.values())
            # Prefer varied scores over all middle scores
            score_variance = sum((score - 3) ** 2 for score in scores) / len(scores)
            confidence += min(0.1, score_variance * 0.02)

        # Apply base factors
        base_bonus = super().calculate(**kwargs) - self.base_confidence
        confidence += base_bonus

        return max(0.1, min(0.95, confidence))


# Factory function for critics
def create_confidence_calculator(
    calculator_type: str = "base", **kwargs: Any
) -> ConfidenceCalculator:
    """Create a confidence calculator instance.

    Args:
        calculator_type: Type of calculator ('base' or 'structured')
        **kwargs: Arguments passed to calculator constructor

    Returns:
        ConfidenceCalculator instance
    """
    if calculator_type == "structured":
        return StructuredConfidenceCalculator(**kwargs)
    else:
        return BaseConfidenceCalculator(**kwargs)
