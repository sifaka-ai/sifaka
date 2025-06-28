"""Confidence calculation for critic assessments."""

from typing import List, Optional, Dict, Any


class ConfidenceCalculator:
    """Calculates confidence scores for critic assessments.

    Uses simple, objective metrics rather than trying to be clever.
    """

    def __init__(self, base_confidence: float = 0.7):
        """Initialize calculator with base confidence."""
        self.base_confidence = base_confidence

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate confidence based on simple objective factors.

        Just uses basic metrics like length and suggestion count.
        No fancy heuristics that don't actually work.

        Args:
            feedback: The feedback text
            suggestions: List of suggestions made
            response_length: Length of full response
            metadata: Additional metadata

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start with base confidence
        confidence = self.base_confidence

        # Simple adjustments based on objective metrics

        # 1. Feedback length (longer = slightly more confident)
        feedback_words = len(feedback.split())
        if feedback_words > 100:
            confidence += 0.05
        elif feedback_words < 20:
            confidence -= 0.05

        # 2. Number of suggestions (some but not too many = good)
        suggestion_count = len(suggestions)
        if 1 <= suggestion_count <= 3:
            confidence += 0.05
        elif suggestion_count > 5:
            confidence -= 0.05
        elif suggestion_count == 0:
            confidence -= 0.1

        # 3. Average suggestion length (specific suggestions tend to be longer)
        if suggestions:
            avg_suggestion_length = sum(len(s.split()) for s in suggestions) / len(
                suggestions
            )
            if avg_suggestion_length > 10:
                confidence += 0.05
            elif avg_suggestion_length < 5:
                confidence -= 0.05

        # Ensure valid range
        return max(0.0, min(1.0, confidence))

    def _score_specificity(self, text: str) -> float:
        """Deprecated - kept for compatibility."""
        return 0.5

    def _score_uncertainty(self, text: str) -> float:
        """Deprecated - kept for compatibility."""
        return 0.5
