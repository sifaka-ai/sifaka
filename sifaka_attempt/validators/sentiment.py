"""
Sentiment validator for evaluating sentiment of text.

This module provides a validator that checks if text has the desired sentiment
using the SentimentClassifier from the classifiers module.
"""

from typing import Dict, Any, List, Optional, Union
from ..types import ValidationResult
from ..classifiers.sentiment import SentimentClassifier


class SentimentValidator:
    """
    Validator that checks if text has the desired sentiment.

    This validator uses the SentimentClassifier to evaluate the sentiment of text
    and determines if it passes based on the desired sentiment and threshold.

    By default, it requires the 'nltk' and 'vader_lexicon' packages to be installed.
    Install them with: pip install nltk && python -c "import nltk; nltk.download('vader_lexicon')"
    """

    def __init__(
        self,
        desired_sentiment: str = "positive",
        threshold: float = 0.1,
        neutral_passes: bool = True,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
    ):
        """
        Initialize the sentiment validator.

        Args:
            desired_sentiment: The sentiment to check for ("positive", "negative", or "neutral")
            threshold: Minimum confidence threshold for validation to pass
            neutral_passes: Whether neutral sentiment should pass validation
            positive_threshold: Threshold for positive sentiment detection
            negative_threshold: Threshold for negative sentiment detection
        """
        if desired_sentiment not in ["positive", "negative", "neutral", "any"]:
            raise ValueError("desired_sentiment must be one of: positive, negative, neutral, any")

        self.desired_sentiment = desired_sentiment
        self.threshold = threshold
        self.neutral_passes = neutral_passes

        # Create the classifier
        self.classifier = SentimentClassifier(
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text for the desired sentiment.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with sentiment analysis results
        """
        if not text.strip():
            if self.desired_sentiment in ["neutral", "any"] or self.neutral_passes:
                return ValidationResult(
                    passed=True,
                    message="Empty text passes sentiment validation",
                    score=1.0,
                    issues=[],
                    suggestions=[],
                    metadata={"sentiment": "neutral", "scores": {}},
                )
            else:
                return ValidationResult(
                    passed=False,
                    message=f"Empty text does not express {self.desired_sentiment} sentiment",
                    score=0.0,
                    issues=["Text is empty"],
                    suggestions=[f"Add content with {self.desired_sentiment} sentiment"],
                    metadata={"sentiment": "neutral", "scores": {}},
                )

        # Use the classifier to get sentiment information
        classification = self.classifier.classify(text)

        # Get sentiment and confidence
        sentiment = classification.label
        confidence = classification.confidence

        # Get compound score from metadata
        compound_score = classification.metadata.get("compound_score", 0.0)

        # Determine if validation passed
        passed = False

        if self.desired_sentiment == "any":
            # If we accept any sentiment, always pass if we're confident
            passed = confidence >= self.threshold
        elif sentiment == self.desired_sentiment:
            # Direct match to desired sentiment
            passed = confidence >= self.threshold
        elif sentiment == "neutral" and self.neutral_passes:
            # Neutral sentiment passes if allowed
            passed = True

        # Create issues and suggestions
        issues = []
        suggestions = []

        if not passed:
            issues.append(f"Text does not express {self.desired_sentiment} sentiment")

            if self.desired_sentiment == "positive":
                suggestions.append("Use more positive language and emotions")
            elif self.desired_sentiment == "negative":
                suggestions.append("Express criticism or issues more clearly")
            elif self.desired_sentiment == "neutral":
                suggestions.append("Use more objective and neutral language")

        # Create message
        if passed:
            if sentiment == self.desired_sentiment:
                message = f"Text expresses {self.desired_sentiment} sentiment"
            elif sentiment == "neutral" and self.neutral_passes:
                message = "Text expresses neutral sentiment (passing)"
            else:
                message = f"Text passed sentiment validation"
        else:
            message = f"Text does not express {self.desired_sentiment} sentiment"

        # Calculate score based on sentiment and desired sentiment
        if self.desired_sentiment == "positive":
            # For positive desired sentiment, score is higher for more positive compound
            score = (compound_score + 1) / 2  # Scale from [-1,1] to [0,1]
        elif self.desired_sentiment == "negative":
            # For negative desired sentiment, score is higher for more negative compound
            score = (-compound_score + 1) / 2  # Scale from [-1,1] to [0,1]
        elif self.desired_sentiment == "neutral":
            # For neutral desired sentiment, score is higher when closer to zero
            score = 1.0 - abs(compound_score)
        else:  # "any"
            # For "any" desired sentiment, score is the confidence (higher is better)
            score = confidence

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "sentiment": sentiment,
                "confidence": confidence,
                "compound_score": compound_score,
                "pos_score": classification.metadata.get("pos_score", 0.0),
                "neg_score": classification.metadata.get("neg_score", 0.0),
                "neu_score": classification.metadata.get("neu_score", 0.0),
            },
        )
