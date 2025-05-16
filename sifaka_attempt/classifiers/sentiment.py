"""
Sentiment classifier for categorizing text sentiment.

This module provides a classifier that categorizes text sentiment as positive, negative,
or neutral using the VADER sentiment analysis tool.
"""

from typing import Dict, Any, List, Optional, Union
from ..di import inject
from . import ClassificationResult


class SentimentClassifier:
    """
    Classifier that categorizes text based on sentiment.

    This classifier uses the VADER (Valence Aware Dictionary and sEntiment Reasoner)
    library to detect sentiment in text, categorizing it as positive, neutral, or negative.

    By default, it requires the 'nltk' and 'vader_lexicon' packages to be installed.
    Install them with: pip install nltk && python -c "import nltk; nltk.download('vader_lexicon')"
    """

    def __init__(
        self,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
    ):
        """
        Initialize the sentiment classifier.

        Args:
            positive_threshold: Threshold for positive sentiment detection (0.0 to 1.0)
            negative_threshold: Threshold for negative sentiment detection (-1.0 to 0.0)
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self._analyzer = None  # Lazy-loaded

    @property
    def analyzer(self):
        """Lazy-load the VADER sentiment analyzer."""
        if self._analyzer is None:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer

                self._analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                raise ImportError(
                    "NLTK and vader_lexicon are required for SentimentClassifier. "
                    "Install them with: pip install nltk && python -c \"import nltk; nltk.download('vader_lexicon')\""
                )
        return self._analyzer

    def _get_sentiment_label(self, compound_score: float) -> tuple[str, float]:
        """
        Get the sentiment label and confidence based on compound score.

        Args:
            compound_score: VADER compound sentiment score (-1.0 to 1.0)

        Returns:
            Tuple of (label, confidence)
        """
        if compound_score >= self.positive_threshold:
            # Positive sentiment
            confidence = (compound_score + 1) / 2  # Scale from [-1,1] to [0,1]
            return "positive", confidence
        elif compound_score <= self.negative_threshold:
            # Negative sentiment
            confidence = (-compound_score + 1) / 2  # Scale from [-1,1] to [0,1]
            return "negative", confidence
        else:
            # Neutral sentiment
            # For neutral, confidence is higher when compound is close to 0
            distance_from_zero = abs(compound_score)
            confidence = 1.0 - (
                distance_from_zero / max(abs(self.positive_threshold), abs(self.negative_threshold))
            )
            return "neutral", confidence

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text based on sentiment.

        Args:
            text: The text to classify

        Returns:
            A ClassificationResult with the sentiment label and confidence score
        """
        if not text.strip():
            return ClassificationResult(
                label="neutral",
                confidence=1.0,
                message="Empty text is considered neutral",
                metadata={"scores": {}},
            )

        # Get sentiment scores from VADER
        try:
            scores = self.analyzer.polarity_scores(text)
        except Exception as e:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                passed=False,
                message=f"Error analyzing sentiment: {str(e)}",
                metadata={"error": str(e)},
            )

        # Get compound score and determine sentiment
        compound_score = scores["compound"]
        label, confidence = self._get_sentiment_label(compound_score)

        # Create message based on label
        if label == "positive":
            message = "Text expresses positive sentiment"
        elif label == "negative":
            message = "Text expresses negative sentiment"
        else:
            message = "Text expresses neutral sentiment"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            message=message,
            metadata={
                "compound_score": compound_score,
                "pos_score": scores["pos"],
                "neg_score": scores["neg"],
                "neu_score": scores["neu"],
            },
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: The list of texts to classify

        Returns:
            A list of ClassificationResults
        """
        return [self.classify(text) for text in texts]
