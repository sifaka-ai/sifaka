"""Sentiment classifier for analyzing emotional tone in text.

This module provides a classifier for analyzing sentiment (positive, negative, neutral)
using TextBlob with fallback to simple lexicon-based analysis.
"""

import importlib
from typing import List, Optional

from sifaka.classifiers.base import (
    CachedTextClassifier,
    ClassificationResult,
    ClassifierError,
    TextClassifier,
)
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Configure logger
logger = get_logger(__name__)

# Simple sentiment lexicons for fallback
POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "fantastic",
    "awesome",
    "love",
    "like",
    "enjoy",
    "happy",
    "pleased",
    "satisfied",
    "delighted",
    "perfect",
    "brilliant",
    "outstanding",
    "superb",
    "magnificent",
    "beautiful",
    "best",
    "better",
    "positive",
    "nice",
    "fine",
    "glad",
    "thrilled",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "horrible",
    "disgusting",
    "hate",
    "dislike",
    "angry",
    "sad",
    "disappointed",
    "frustrated",
    "annoyed",
    "upset",
    "mad",
    "worst",
    "worse",
    "negative",
    "poor",
    "ugly",
    "stupid",
    "boring",
    "wrong",
    "fail",
    "failed",
    "broken",
    "useless",
    "worthless",
    "pathetic",
}


class SentimentClassifier(TextClassifier):
    """Classifier for analyzing sentiment in text.

    This classifier uses TextBlob when available for sentiment analysis,
    with fallback to simple lexicon-based scoring. It classifies text
    as positive, negative, or neutral.

    Attributes:
        positive_threshold: Threshold above which sentiment is positive
        negative_threshold: Threshold below which sentiment is negative
        textblob: The TextBlob library instance
    """

    def __init__(
        self,
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
        name: str = "SentimentClassifier",
        description: str = "Analyzes sentiment (positive/negative/neutral) in text",
    ):
        """Initialize the sentiment classifier.

        Args:
            positive_threshold: Threshold above which sentiment is positive
            negative_threshold: Threshold below which sentiment is negative
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.textblob = None
        self._initialize_textblob()

    def _initialize_textblob(self) -> None:
        """Initialize TextBlob for sentiment analysis."""
        try:
            # Try to use TextBlob
            self.textblob = importlib.import_module("textblob")  # type: ignore
            logger.debug("Initialized sentiment classifier with TextBlob")

        except ImportError:
            logger.warning(
                "TextBlob not available. Using lexicon-based sentiment analysis. "
                "Install TextBlob for better accuracy: pip install textblob"
            )
            self.textblob = None

    def classify(self, text: str) -> ClassificationResult:
        """Classify text for sentiment.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment prediction

        Raises:
            ClassifierError: If classification fails
        """
        if not text or not text.strip():
            return ClassificationResult(
                label="neutral",
                confidence=0.5,
                metadata={"reason": "empty_text", "input_length": 0},
            )

        try:
            if self.textblob is not None:
                return self._classify_with_textblob(text)  # type: ignore[unreachable]
            else:
                return self._classify_with_lexicon(text)

        except Exception as e:
            logger.error(f"Sentiment classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for sentiment: {str(e)}",
                component="SentimentClassifier",
                operation="classification",
            )

    def _classify_with_textblob(self, text: str) -> ClassificationResult:
        """Classify using TextBlob sentiment analysis."""
        if self.textblob is None:
            raise ClassifierError(
                message="TextBlob is not available",
                component="SentimentClassifier",
                operation="textblob_classification",
            )

        try:
            blob = self.textblob.TextBlob(text)
            polarity = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
            subjectivity = blob.sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)

            # Determine label based on thresholds
            if polarity > self.positive_threshold:
                label = "positive"
                confidence = min(0.5 + (polarity * 0.5), 0.95)
            elif polarity < self.negative_threshold:
                label = "negative"
                confidence = min(0.5 + (abs(polarity) * 0.5), 0.95)
            else:
                label = "neutral"
                confidence = 0.7 - abs(polarity)  # More neutral = higher confidence

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "method": "textblob",
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "input_length": len(text),
                },
            )

        except Exception as e:
            # Fallback to lexicon-based analysis
            logger.warning(f"TextBlob failed, using lexicon analysis: {e}")
            return self._classify_with_lexicon(text)

    def _classify_with_lexicon(self, text: str) -> ClassificationResult:
        """Classify using simple lexicon-based approach."""
        text_lower = text.lower()
        words = text_lower.split()

        # Count positive and negative words
        positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / len(words)

        # Determine label and confidence
        if sentiment_score > 0.05:  # More lenient threshold for lexicon
            label = "positive"
            confidence = min(0.6 + (sentiment_score * 2), 0.85)
        elif sentiment_score < -0.05:
            label = "negative"
            confidence = min(0.6 + (abs(sentiment_score) * 2), 0.85)
        else:
            label = "neutral"
            confidence = 0.7

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "lexicon_based",
                "sentiment_score": sentiment_score,
                "positive_words": positive_count,
                "negative_words": negative_count,
                "total_words": len(words),
                "input_length": len(text),
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["negative", "neutral", "positive"]


class CachedSentimentClassifier(CachedTextClassifier):
    """Cached version of SentimentClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
        cache_size: int = 128,
        name: str = "CachedSentimentClassifier",
        description: str = "Analyzes sentiment with LRU caching",
    ):
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.textblob = None
        self._initialize_textblob()

    def _initialize_textblob(self) -> None:
        """Initialize TextBlob for sentiment analysis."""
        try:
            self.textblob = importlib.import_module("textblob")  # type: ignore
            logger.debug("Initialized cached sentiment classifier with TextBlob")
        except ImportError:
            logger.warning(
                "TextBlob not available. CachedSentimentClassifier will use lexicon-based analysis."
            )
            self.textblob = None

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform sentiment classification without caching."""
        if not text or not text.strip():
            return ClassificationResult(
                label="neutral",
                confidence=0.5,
                metadata={"reason": "empty_text", "input_length": 0},
            )

        try:
            if self.textblob is not None:
                return self._classify_with_textblob(text)
            else:
                return self._classify_with_lexicon(text)
        except Exception as e:
            logger.error(f"Cached sentiment classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for sentiment: {str(e)}",
                component="CachedSentimentClassifier",
                operation="classification",
            )

    def _classify_with_textblob(self, text: str) -> ClassificationResult:
        """Classify using TextBlob sentiment analysis."""
        if self.textblob is None:
            raise ClassifierError(
                message="TextBlob is not available",
                component="CachedSentimentClassifier",
                operation="textblob_classification",
            )

        try:
            blob = self.textblob.TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > self.positive_threshold:
                label = "positive"
                confidence = min(0.5 + (polarity * 0.5), 0.95)
            elif polarity < self.negative_threshold:
                label = "negative"
                confidence = min(0.5 + (abs(polarity) * 0.5), 0.95)
            else:
                label = "neutral"
                confidence = 0.7 - abs(polarity)

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "method": "textblob",
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "input_length": len(text),
                    "cached": True,
                },
            )
        except Exception as e:
            logger.warning(f"TextBlob failed, using lexicon analysis: {e}")
            return self._classify_with_lexicon(text)

    def _classify_with_lexicon(self, text: str) -> ClassificationResult:
        """Classify using simple lexicon-based approach."""
        text_lower = text.lower()
        words = text_lower.split()

        positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / len(words)

        if sentiment_score > 0.05:
            label = "positive"
            confidence = min(0.6 + (sentiment_score * 2), 0.85)
        elif sentiment_score < -0.05:
            label = "negative"
            confidence = min(0.6 + (abs(sentiment_score) * 2), 0.85)
        else:
            label = "neutral"
            confidence = 0.7

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "lexicon_based",
                "sentiment_score": sentiment_score,
                "positive_words": positive_count,
                "negative_words": negative_count,
                "total_words": len(words),
                "input_length": len(text),
                "cached": True,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["negative", "neutral", "positive"]


def create_sentiment_validator(
    required_sentiment: Optional[str] = None,
    forbidden_sentiments: Optional[List[str]] = None,
    min_confidence: float = 0.6,
    name: str = "SentimentValidator",
) -> ClassifierValidator:
    """Create a validator that checks text sentiment.

    Args:
        required_sentiment: Required sentiment ('positive', 'negative', 'neutral')
        forbidden_sentiments: List of forbidden sentiments
        min_confidence: Minimum confidence for sentiment detection
        name: Name of the validator

    Returns:
        A ClassifierValidator configured for sentiment validation
    """
    classifier = SentimentClassifier()

    # Set up valid/invalid labels
    valid_labels = None
    invalid_labels = None

    if required_sentiment:
        valid_labels = [required_sentiment]
    elif forbidden_sentiments:
        invalid_labels = forbidden_sentiments

    return ClassifierValidator(
        classifier=classifier,
        threshold=min_confidence,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        name=name,
    )


def create_cached_sentiment_validator(
    required_sentiment: Optional[str] = None,
    forbidden_sentiments: Optional[List[str]] = None,
    min_confidence: float = 0.6,
    cache_size: int = 128,
    name: str = "CachedSentimentValidator",
) -> ClassifierValidator:
    """Create a cached validator that checks text sentiment with LRU caching."""
    classifier = CachedSentimentClassifier(cache_size=cache_size)

    valid_labels = None
    invalid_labels = None

    if required_sentiment:
        valid_labels = [required_sentiment]
    elif forbidden_sentiments:
        invalid_labels = forbidden_sentiments

    return ClassifierValidator(
        classifier=classifier,
        threshold=min_confidence,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        name=name,
    )
