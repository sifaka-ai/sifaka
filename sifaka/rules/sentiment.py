from typing import Optional, Dict, Any
from textblob import TextBlob
from sifaka.rules.base import Rule, RuleResult


class SentimentRule(Rule):
    """
    Rule that checks if the output meets a minimum sentiment threshold.

    This rule is part of the Sifaka validation framework and implements sentiment
    analysis for text content. It uses a sentiment analysis model to determine
    the emotional tone of the output and ensures it meets a minimum threshold.

    Architecture Notes:
    - Inherits from the base Rule class to implement the validation contract
    - Uses a sentiment analysis model to score text
    - Returns RuleResult objects containing validation status, messages, and metadata
    - Follows the single responsibility principle by focusing only on sentiment validation
    - Includes error handling for sentiment analysis failures

    Data Flow:
    1. User creates SentimentRule with desired sentiment threshold
    2. validate() method receives output text
    3. Sentiment analysis is performed on the text
    4. Score is compared against threshold
    5. Result is wrapped in RuleResult with relevant metadata
    6. RuleResult is returned to the caller

    Usage Example:
        rule = SentimentRule(
            name="sentiment_rule",
            description="Ensures positive sentiment",
            config={"min_sentiment": 0.0}
        )
    """

    min_sentiment: float = 0.0

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the rule with a minimum sentiment threshold.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - min_sentiment: Minimum allowed sentiment score (inclusive).
                                  Must be between -1.0 (very negative) and 1.0 (very positive).
                                  Defaults to 0.0 (neutral).
            **kwargs: Additional arguments

        Raises:
            ValueError: If min_sentiment is not between -1.0 and 1.0
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract sentiment threshold from config
        config = config or {}
        min_sentiment = config.get("min_sentiment", 0.0)

        if not -1.0 <= min_sentiment <= 1.0:
            raise ValueError("min_sentiment must be between -1.0 and 1.0")

        # Set the value using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "min_sentiment", min_sentiment)

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output meets the minimum sentiment threshold.

        This method implements the core validation logic by:
        1. Analyzing the sentiment of the output text
        2. Comparing the sentiment score against the threshold
        3. Constructing a detailed result message
        4. Packaging the result with relevant metadata

        Args:
            output: The text to validate

        Returns:
            RuleResult: Contains:
                       - passed: Boolean indicating if sentiment meets threshold
                       - message: Human-readable validation result
                       - metadata: Additional validation details including scores
        """
        try:
            # TODO: Replace with actual sentiment analysis implementation
            sentiment_score = 0.0  # Placeholder implementation
            passed = sentiment_score >= self.min_sentiment

            return RuleResult(
                passed=passed,
                message=(
                    f"Output sentiment {sentiment_score:.2f} "
                    f"{'meets' if passed else 'does not meet'} "
                    f"minimum threshold of {self.min_sentiment}"
                ),
                metadata={"sentiment_score": sentiment_score, "min_sentiment": self.min_sentiment},
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during sentiment analysis: {str(e)}",
                metadata={"error": str(e), "min_sentiment": self.min_sentiment},
            )

    def _generate_description(
        self, min_sentiment: float, subjectivity_threshold: Optional[float]
    ) -> str:
        """Generate a human-readable description of the sentiment criteria."""
        criteria = [f"sentiment above {min_sentiment:.2f}"]
        if subjectivity_threshold is not None:
            criteria.append(f"subjectivity below {subjectivity_threshold:.2f}")
        return f"Checks if output has {' and '.join(criteria)}"
