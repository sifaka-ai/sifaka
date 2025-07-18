"""Confidence scoring system for critic evaluations.

This module provides a pragmatic approach to calculating confidence scores
for critic assessments. Rather than using complex heuristics that may not
correlate with actual quality, it uses simple objective metrics.

## Design Philosophy:

The confidence calculator intentionally avoids:
- Complex NLP analysis that doesn't improve accuracy
- Subjective language pattern matching
- Over-engineered scoring algorithms

Instead, it focuses on objective, measurable factors:
- Length of feedback (proxy for thoroughness)
- Number of suggestions (proxy for actionability)
- Specificity of suggestions (proxy for quality)

## Usage:

    >>> calc = ConfidenceCalculator(base_confidence=0.7)
    >>> confidence = calc.calculate(
    ...     feedback="The text needs improvement in clarity...",
    ...     suggestions=["Break up long sentences", "Add examples"],
    ...     response_length=500
    ... )
    >>> print(f"Confidence: {confidence:.2f}")

## Score Interpretation:

- 0.8-1.0: High confidence - detailed, actionable feedback
- 0.6-0.8: Moderate confidence - reasonable feedback
- 0.4-0.6: Low confidence - vague or limited feedback
- 0.0-0.4: Very low confidence - minimal useful feedback
"""

from typing import Any, Dict, List, Optional


class ConfidenceCalculator:
    """Simple, objective confidence scoring for critic assessments.

    This calculator provides a pragmatic approach to scoring confidence
    based on easily measurable factors. It avoids complex heuristics
    in favor of simple metrics that correlate with feedback quality.

    The scoring system is designed to be:
    - Predictable: Same inputs always produce same score
    - Transparent: Clear which factors affect the score
    - Objective: Based on measurable quantities, not subjective analysis
    - Bounded: Always returns values between 0.0 and 1.0

    Example:
        >>> # Standard usage
        >>> calc = ConfidenceCalculator()
        >>>
        >>> # High confidence: detailed feedback with specific suggestions
        >>> high_conf = calc.calculate(
        ...     feedback="" * 150,  # 150 words
        ...     suggestions=[
        ...         "Replace passive voice in paragraph 2",
        ...         "Add transition between sections 3 and 4"
        ...     ],
        ...     response_length=1000
        ... )
        >>>
        >>> # Low confidence: minimal feedback
        >>> low_conf = calc.calculate(
        ...     feedback="Needs work",
        ...     suggestions=[],
        ...     response_length=50
        ... )
    """

    def __init__(self, base_confidence: float = 0.7):
        """Initialize calculator with base confidence level.

        Args:
            base_confidence: Starting confidence level before adjustments.
                Should be between 0.0 and 1.0. Default of 0.7 represents
                moderate confidence as a reasonable starting point.
        """
        self.base_confidence = base_confidence

    def calculate(
        self,
        feedback: str,
        suggestions: List[str],
        response_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate confidence score based on objective metrics.

        Uses simple, measurable factors to estimate confidence in the
        critic's assessment. The scoring adjusts the base confidence
        up or down based on feedback characteristics.

        Args:
            feedback: The main feedback text provided by the critic.
                Longer, detailed feedback increases confidence.
            suggestions: List of specific improvement suggestions.
                2-3 specific suggestions is ideal; too many or none
                reduces confidence.
            response_length: Total length of the critic's response
                in characters. Currently unused but kept for compatibility.
            metadata: Additional metadata from the critic. Currently
                unused but available for future enhancements.

        Returns:
            Confidence score between 0.0 and 1.0, where:
            - 1.0 = Maximum confidence in the assessment
            - 0.7 = Default moderate confidence
            - 0.0 = No confidence in the assessment

        Scoring Factors:
            - Feedback length: +0.05 for >100 words, -0.05 for <20 words
            - Suggestion count: +0.05 for 1-3, -0.05 for >5, -0.1 for 0
            - Suggestion specificity: +0.05 for detailed (>10 words avg)
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
