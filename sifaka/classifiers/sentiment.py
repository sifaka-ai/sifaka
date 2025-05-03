"""
Sentiment classifier using VADER.
"""

import importlib
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, ClassVar, runtime_checkable

from typing_extensions import TypeGuard
from pydantic import PrivateAttr

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Protocol for sentiment analysis engines."""

    @abstractmethod
    def polarity_scores(self, text: str) -> Dict[str, float]: ...


class SentimentClassifier(BaseClassifier[str, str]):
    """
    A lightweight sentiment classifier using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based
    sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

    Requires the 'sentiment' extra to be installed:
    pip install sifaka[sentiment]
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["positive", "neutral", "negative", "unknown"]
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for lexicon-based analysis

    # Default thresholds
    DEFAULT_POSITIVE_THRESHOLD: ClassVar[float] = 0.05
    DEFAULT_NEGATIVE_THRESHOLD: ClassVar[float] = -0.05

    # Private attributes using PrivateAttr
    _analyzer: Optional[SentimentAnalyzer] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        name: str = "sentiment_classifier",
        description: str = "Analyzes text sentiment using VADER",
        analyzer: Optional[SentimentAnalyzer] = None,
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the sentiment classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            analyzer: Custom sentiment analyzer implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Store analyzer for later use if provided
        if analyzer is not None:
            self._analyzer = analyzer
        self._initialized = False

        # Create config if not provided
        if config is None:
            # Extract thresholds from kwargs
            thresholds = {
                "positive_threshold": kwargs.pop("positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD),
                "negative_threshold": kwargs.pop("negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD),
            }

            # Create config with remaining kwargs
            config = ClassifierConfig[str](
                labels=self.DEFAULT_LABELS,
                cost=self.DEFAULT_COST,
                params=thresholds,
                **kwargs
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

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

    def _get_thresholds(self) -> Dict[str, float]:
        """Get thresholds from config."""
        params = self.config.params
        return {
            "positive_threshold": params.get("positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD),
            "negative_threshold": params.get("negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD),
        }

    def _get_sentiment_label(self, compound_score: float) -> tuple[str, float]:
        """Get sentiment label and confidence based on compound score."""
        # Get thresholds
        thresholds = self._get_thresholds()
        positive_threshold = thresholds["positive_threshold"]
        negative_threshold = thresholds["negative_threshold"]

        # Determine sentiment label
        if compound_score >= positive_threshold:
            label = "positive"
        elif compound_score <= negative_threshold:
            label = "negative"
        else:
            label = "neutral"

        # Convert compound score from [-1, 1] to confidence [0, 1]
        confidence = abs(compound_score)

        return label, confidence

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Implement sentiment classification logic without caching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment scores
        """
        if not self._initialized:
            self.warm_up()

        # Handle empty or whitespace-only text
        if not text.strip():
            return ClassificationResult(
                label="unknown",
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

            label, confidence = self._get_sentiment_label(compound_score)

            # Special case for unknown
            if label == "unknown":
                confidence = 0.0

            return ClassificationResult[str](
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
            return ClassificationResult[str](
                label="unknown",
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
        **kwargs,
    ) -> "SentimentClassifier":
        """
        Factory method to create a classifier with a custom analyzer.

        Args:
            analyzer: Custom sentiment analyzer implementation
            name: Name of the classifier
            description: Description of the classifier
            **kwargs: Additional configuration parameters

        Returns:
            Configured SentimentClassifier instance
        """
        # Validate analyzer first
        if not isinstance(analyzer, SentimentAnalyzer):
            raise ValueError(
                f"Analyzer must implement SentimentAnalyzer protocol, got {type(analyzer)}"
            )

        # Create instance with validated analyzer and kwargs
        instance = cls(
            name=name,
            description=description,
            analyzer=analyzer,
            **kwargs,
        )

        # Mark as initialized
        instance._initialized = True

        return instance


def create_sentiment_classifier(
    name: str = "sentiment_classifier",
    description: str = "Analyzes text sentiment using VADER",
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
    cache_size: int = 0,
    cost: int = 1,
    **kwargs,
) -> SentimentClassifier:
    """
    Factory function to create a sentiment classifier.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        positive_threshold: Positive sentiment threshold (default: 0.05)
        negative_threshold: Negative sentiment threshold (default: -0.05)
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured SentimentClassifier instance
    """
    # Prepare params
    params = kwargs.pop("params", {})
    params.update({
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
    })

    # Create config
    config = ClassifierConfig[str](
        labels=SentimentClassifier.DEFAULT_LABELS,
        cache_size=cache_size,
        cost=cost,
        params=params,
    )

    # Create and return classifier
    return SentimentClassifier(
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
