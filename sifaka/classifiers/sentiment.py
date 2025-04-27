"""
Sentiment classifier using VADER.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable, Final
from typing_extensions import TypeGuard
import importlib
import logging
from dataclasses import dataclass
from abc import abstractmethod

from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Protocol for sentiment analysis engines."""

    @abstractmethod
    def polarity_scores(self, text: str) -> Dict[str, float]: ...


@dataclass(frozen=True)
class SentimentThresholds:
    """Immutable thresholds for sentiment classification."""

    positive: float = 0.05
    negative: float = -0.05

    def __post_init__(self) -> None:
        if self.positive < self.negative:
            raise ValueError("Positive threshold must be greater than negative threshold")
        if not (-1.0 <= self.negative <= self.positive <= 1.0):
            raise ValueError("Thresholds must be between -1.0 and 1.0")


class SentimentClassifier(BaseClassifier):
    """
    A lightweight sentiment classifier using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based
    sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

    Requires the 'sentiment' extra to be installed:
    pip install sifaka[sentiment]
    """

    # Class-level constants
    DEFAULT_LABELS: Final[List[str]] = ["positive", "neutral", "negative"]
    DEFAULT_COST: Final[int] = 1  # Low cost for lexicon-based analysis

    def __init__(
        self,
        name: str = "sentiment_classifier",
        description: str = "Analyzes text sentiment using VADER",
        thresholds: Optional[SentimentThresholds] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the sentiment classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            thresholds: Sentiment threshold configuration
            analyzer: Custom sentiment analyzer implementation
            **kwargs: Additional configuration parameters
        """
        config = ClassifierConfig(labels=self.DEFAULT_LABELS, cost=self.DEFAULT_COST, **kwargs)
        super().__init__(name=name, description=description, config=config)

        self._thresholds = thresholds or SentimentThresholds()
        self._analyzer = analyzer
        self._initialized = False

    def _validate_analyzer(self, analyzer: Any) -> TypeGuard[SentimentAnalyzer]:
        """Validate that an analyzer implements the required protocol."""
        if not isinstance(analyzer, SentimentAnalyzer):
            raise ValueError(
                f"Analyzer must implement SentimentAnalyzer protocol, got {type(analyzer)}"
            )
        return True

    def _load_vader(self) -> SentimentAnalyzer:
        """Load the VADER sentiment analyzer."""
        try:
            vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
            analyzer = vader_module.SentimentIntensityAnalyzer()
            self._validate_analyzer(analyzer)
            return analyzer
        except ImportError:
            raise ImportError(
                "VADER package is required for SentimentClassifier. "
                "Install it with: pip install sifaka[sentiment]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load VADER: {e}")

    def warm_up(self) -> None:
        """Initialize the sentiment analyzer if needed."""
        if not self._initialized:
            self._analyzer = self._analyzer or self._load_vader()
            self._initialized = True

    def _get_sentiment_label(self, compound_score: float) -> str:
        """Get sentiment label based on compound score."""
        if compound_score >= self._thresholds.positive:
            return "positive"
        elif compound_score <= self._thresholds.negative:
            return "negative"
        return "neutral"

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement sentiment classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment scores
        """
        self.warm_up()

        # Handle empty or whitespace-only text
        if not text.strip():
            return ClassificationResult(
                label="neutral",
                confidence=0.0,
                metadata={
                    "compound_score": 0.0,
                    "pos_score": 0.0,
                    "neg_score": 0.0,
                    "neu_score": 1.0,
                    "reason": "empty_input",
                },
            )

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
                metadata={
                    "error": str(e),
                    "compound_score": 0.0,
                    "pos_score": 0.0,
                    "neg_score": 0.0,
                    "neu_score": 1.0,
                },
            )

    @classmethod
    def create_with_custom_analyzer(
        cls,
        analyzer: SentimentAnalyzer,
        name: str = "custom_sentiment_classifier",
        description: str = "Custom sentiment analyzer",
        thresholds: Optional[SentimentThresholds] = None,
        **kwargs,
    ) -> "SentimentClassifier":
        """
        Factory method to create a classifier with a custom analyzer.

        Args:
            analyzer: Custom sentiment analyzer implementation
            name: Name of the classifier
            description: Description of the classifier
            thresholds: Custom sentiment thresholds
            **kwargs: Additional configuration parameters

        Returns:
            Configured SentimentClassifier instance
        """
        instance = cls(
            name=name, description=description, thresholds=thresholds, analyzer=analyzer, **kwargs
        )
        instance._validate_analyzer(analyzer)
        return instance
