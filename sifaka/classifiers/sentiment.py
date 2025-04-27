"""
Sentiment classifier using VADER.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import importlib
import logging

from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Only import type hints during type checking
if TYPE_CHECKING:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentClassifier(Classifier):
    """
    A lightweight sentiment classifier using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based
    sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

    Requires the 'sentiment' extra to be installed:
    pip install sifaka[sentiment]

    Attributes:
        threshold_pos: Score threshold for positive sentiment
        threshold_neg: Score threshold for negative sentiment
        labels: Possible sentiment labels (positive, neutral, negative)
    """

    threshold_pos: float = 0.05
    threshold_neg: float = -0.05

    def __init__(
        self,
        name: str = "sentiment_classifier",
        description: str = "Analyzes text sentiment using VADER",
        threshold_pos: float = 0.05,
        threshold_neg: float = -0.05,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the sentiment classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            threshold_pos: Score threshold for positive sentiment
            threshold_neg: Score threshold for negative sentiment
            config: Additional configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            labels=["positive", "neutral", "negative"],
            cost=1,  # Low cost for lexicon-based analysis
            **kwargs,
        )
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self._analyzer = None

    def _load_vader(self) -> None:
        """Load the VADER sentiment analyzer."""
        try:
            vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
            self._analyzer = vader_module.SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError(
                "VADER package is required for SentimentClassifier. "
                "Install it with: pip install sifaka[sentiment]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load VADER: {e}")

    def warm_up(self) -> None:
        """Initialize the sentiment analyzer if needed."""
        if self._analyzer is None:
            self._load_vader()

    def _get_sentiment_label(self, compound_score: float) -> str:
        """Get sentiment label based on compound score."""
        if compound_score >= self.threshold_pos:
            return "positive"
        elif compound_score <= self.threshold_neg:
            return "negative"
        return "neutral"

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text sentiment.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment scores
        """
        self.warm_up()
        try:
            scores = self._analyzer.polarity_scores(text)
            compound_score = scores["compound"]
            label = self._get_sentiment_label(compound_score)

            # Convert compound score from [-1, 1] to confidence [0, 1]
            confidence = abs(compound_score)

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "compound_score": compound_score,
                    "pos_score": scores["pos"],
                    "neg_score": scores["neg"],
                    "neu_score": scores["neu"],
                },
            )
        except Exception as e:
            logger.error("Failed to classify text sentiment: %s", e)
            return ClassificationResult(
                label="neutral",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        return [self.classify(text) for text in texts]
