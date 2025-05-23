"""Bias classifier for detecting biased language in text.

This module provides a classifier for detecting various forms of bias in text,
including gender bias, racial bias, political bias, and other forms of
discriminatory language using machine learning.
"""

import logging
from typing import Any, List, Optional

from sifaka.classifiers.base import (
    ClassificationResult,
    ClassifierError,
    TextClassifier,
    CachedTextClassifier,
)
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Configure logger
logger = get_logger(__name__)

# Sample biased text for training (in a real implementation, use a proper dataset)
BIASED_SAMPLES = [
    "Men are naturally better at math than women",
    "Women are too emotional to be leaders",
    "All politicians are corrupt",
    "Young people are lazy and entitled",
    "Old people can't understand technology",
    "Rich people don't care about others",
    "Poor people are just not trying hard enough",
    "That religion is full of extremists",
    "People from that country are all criminals",
    "Those people don't belong here",
]

UNBIASED_SAMPLES = [
    "People have different strengths in mathematics",
    "Leadership styles vary among individuals",
    "Political systems have various challenges",
    "Different generations have unique perspectives",
    "Technology adoption varies by individual",
    "Wealth distribution affects social dynamics",
    "Economic opportunities differ across communities",
    "Religious beliefs are diverse and personal",
    "Immigration policies require careful consideration",
    "Communities benefit from diversity and inclusion",
]


class BiasClassifier(TextClassifier):
    """Classifier for detecting bias in text using machine learning.

    This classifier uses a simple TF-IDF vectorizer and logistic regression
    to detect potentially biased language. It's trained on sample data and
    can identify various forms of bias including gender, racial, political,
    and socioeconomic bias.

    Attributes:
        threshold: Confidence threshold for bias detection
        model: The trained classification model
    """

    def __init__(
        self,
        threshold: float = 0.7,
        name: str = "BiasClassifier",
        description: str = "Detects bias in text using machine learning",
    ):
        """Initialize the bias classifier.

        Args:
            threshold: Confidence threshold for bias detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.threshold = threshold
        self.model: Optional[Any] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize and train the bias detection model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            # Prepare training data
            X = BIASED_SAMPLES + UNBIASED_SAMPLES
            y = [1] * len(BIASED_SAMPLES) + [0] * len(UNBIASED_SAMPLES)

            # Create and train the model
            self.model = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=5000, ngram_range=(1, 2), stop_words="english"
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            )

            # Train the model
            self.model.fit(X, y)

            logger.debug(f"Initialized bias classifier with {len(X)} training samples")

        except ImportError:
            logger.warning(
                "scikit-learn not available. BiasClassifier will use rule-based detection. "
                "Install scikit-learn for better accuracy: pip install scikit-learn"
            )
            self.model = None

    def classify(self, text: str) -> ClassificationResult:
        """Classify text for bias.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with bias prediction

        Raises:
            ClassifierError: If classification fails
        """
        if not text or not text.strip():
            return ClassificationResult(
                label="unbiased",
                confidence=0.5,
                metadata={"reason": "empty_text", "input_length": 0},
            )

        try:
            if self.model is not None:
                return self._classify_with_ml(text)
            else:
                return self._classify_with_rules(text)

        except Exception as e:
            logger.error(f"Bias classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for bias: {str(e)}",
                component="BiasClassifier",
                operation="classification",
            )

    def _classify_with_ml(self, text: str) -> ClassificationResult:
        """Classify using machine learning model."""
        # Get prediction probabilities
        probabilities = self.model.predict_proba([text])[0]

        # Get the predicted class (0 = unbiased, 1 = biased)
        predicted_class = self.model.predict([text])[0]
        confidence = float(probabilities[predicted_class])

        # Map class to label
        label = "biased" if predicted_class == 1 else "unbiased"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "machine_learning",
                "biased_probability": float(probabilities[1]),
                "unbiased_probability": float(probabilities[0]),
                "input_length": len(text),
            },
        )

    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using simple rule-based approach."""
        text_lower = text.lower()

        # Simple bias indicators
        bias_indicators = [
            "men are",
            "women are",
            "all women",
            "all men",
            "people from",
            "those people",
            "they all",
            "naturally better",
            "can't understand",
            "too emotional",
            "all politicians",
            "rich people",
            "poor people",
        ]

        bias_count = sum(1 for indicator in bias_indicators if indicator in text_lower)

        if bias_count > 0:
            confidence = min(0.6 + (bias_count * 0.1), 0.9)
            label = "biased"
        else:
            confidence = 0.7
            label = "unbiased"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "bias_indicators_found": bias_count,
                "input_length": len(text),
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["unbiased", "biased"]


class CachedBiasClassifier(CachedTextClassifier):
    """Cached version of BiasClassifier with LRU caching for improved performance.

    This classifier extends CachedTextClassifier to provide efficient caching
    of bias classification results. It's particularly useful when the same text
    might be classified multiple times during chain execution.

    Attributes:
        threshold: Confidence threshold for bias detection
        model: The trained classification model
        cache_size: Maximum number of cached results
    """

    def __init__(
        self,
        threshold: float = 0.7,
        cache_size: int = 128,
        name: str = "CachedBiasClassifier",
        description: str = "Detects bias in text using machine learning with LRU caching",
    ):
        """Initialize the cached bias classifier.

        Args:
            threshold: Confidence threshold for bias detection
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.threshold = threshold
        self.model: Optional[Any] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize and train the bias detection model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            # Prepare training data
            X = BIASED_SAMPLES + UNBIASED_SAMPLES
            y = [1] * len(BIASED_SAMPLES) + [0] * len(UNBIASED_SAMPLES)

            # Create and train the model
            self.model = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=5000, ngram_range=(1, 2), stop_words="english"
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            )

            # Train the model
            self.model.fit(X, y)

            logger.debug(f"Initialized cached bias classifier with {len(X)} training samples")

        except ImportError:
            logger.warning(
                "scikit-learn not available. CachedBiasClassifier will use rule-based detection. "
                "Install scikit-learn for better accuracy: pip install scikit-learn"
            )
            self.model = None

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform bias classification without caching.

        This method implements the core classification logic that will be
        automatically wrapped with LRU caching by the parent class.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with bias prediction

        Raises:
            ClassifierError: If classification fails
        """
        if not text or not text.strip():
            return ClassificationResult(
                label="unbiased",
                confidence=0.5,
                metadata={"reason": "empty_text", "input_length": 0},
            )

        try:
            if self.model is not None:
                return self._classify_with_ml(text)
            else:
                return self._classify_with_rules(text)

        except Exception as e:
            logger.error(f"Cached bias classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for bias: {str(e)}",
                component="CachedBiasClassifier",
                operation="classification",
            )

    def _classify_with_ml(self, text: str) -> ClassificationResult:
        """Classify using machine learning model."""
        # Get prediction probabilities
        probabilities = self.model.predict_proba([text])[0]

        # Get the predicted class (0 = unbiased, 1 = biased)
        predicted_class = self.model.predict([text])[0]
        confidence = float(probabilities[predicted_class])

        # Map class to label
        label = "biased" if predicted_class == 1 else "unbiased"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "machine_learning",
                "biased_probability": float(probabilities[1]),
                "unbiased_probability": float(probabilities[0]),
                "input_length": len(text),
                "cached": True,  # Indicate this result can be cached
            },
        )

    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using simple rule-based approach."""
        text_lower = text.lower()

        # Simple bias indicators
        bias_indicators = [
            "men are",
            "women are",
            "all women",
            "all men",
            "people from",
            "those people",
            "they all",
            "naturally better",
            "can't understand",
            "too emotional",
            "all politicians",
            "rich people",
            "poor people",
        ]

        bias_count = sum(1 for indicator in bias_indicators if indicator in text_lower)

        if bias_count > 0:
            confidence = min(0.6 + (bias_count * 0.1), 0.9)
            label = "biased"
        else:
            confidence = 0.7
            label = "unbiased"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "bias_indicators_found": bias_count,
                "input_length": len(text),
                "cached": True,  # Indicate this result can be cached
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["unbiased", "biased"]


def create_bias_validator(
    threshold: float = 0.7, name: str = "BiasValidator"
) -> ClassifierValidator:
    """Create a validator that detects bias in text.

    Args:
        threshold: Confidence threshold for bias detection
        name: Name of the validator

    Returns:
        A ClassifierValidator configured for bias detection
    """
    classifier = BiasClassifier(threshold=threshold)

    return ClassifierValidator(
        classifier=classifier,
        threshold=threshold,
        invalid_labels=["biased"],  # Biased text is invalid
        name=name,
    )


def create_cached_bias_validator(
    threshold: float = 0.7, cache_size: int = 128, name: str = "CachedBiasValidator"
) -> ClassifierValidator:
    """Create a cached validator that detects bias in text with LRU caching.

    This validator uses LRU caching to improve performance when the same text
    might be validated multiple times during chain execution.

    Args:
        threshold: Confidence threshold for bias detection
        cache_size: Maximum number of results to cache
        name: Name of the validator

    Returns:
        A ClassifierValidator configured for cached bias detection
    """
    classifier = CachedBiasClassifier(threshold=threshold, cache_size=cache_size)

    return ClassifierValidator(
        classifier=classifier,
        threshold=threshold,
        invalid_labels=["biased"],  # Biased text is invalid
        name=name,
    )
