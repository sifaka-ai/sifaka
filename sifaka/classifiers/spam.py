"""Spam classifier for detecting spam content in text.

This module provides a classifier for detecting spam content using
machine learning with fallback to rule-based detection. Designed for
the new PydanticAI-based Sifaka architecture.
"""

import importlib
from typing import List, Set
import asyncio
import time

from sifaka.classifiers.base import (
    BaseClassifier,
    CachedClassifier,
    ClassificationResult,
    TimingMixin,
)
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Sample spam text for training
SPAM_SAMPLES = [
    "URGENT! You have won $1,000,000! Click here now!",
    "FREE MONEY! No strings attached! Act now!",
    "Congratulations! You are our lucky winner!",
    "CLICK HERE FOR AMAZING DEALS! LIMITED TIME ONLY!",
    "Make money fast! Work from home! No experience needed!",
    "LOSE WEIGHT FAST! Miracle pill! Doctors hate this trick!",
    "Hot singles in your area! Meet them tonight!",
    "Your account will be suspended! Verify now!",
    "FINAL NOTICE: Your warranty is about to expire!",
    "Get rich quick! This one simple trick!",
    "CONGRATULATIONS! You've been selected for a special offer!",
    "CLAIM YOUR PRIZE NOW! Limited time offer expires soon!",
    "AMAZING OPPORTUNITY! Make thousands from home!",
    "URGENT ACTION REQUIRED! Your account needs verification!",
    "FREE TRIAL! No credit card required! Cancel anytime!",
]

HAM_SAMPLES = [
    "Hi, how are you doing today?",
    "The meeting is scheduled for 3 PM tomorrow.",
    "Thanks for your help with the project.",
    "Can you please review this document?",
    "I'll be working from home tomorrow.",
    "The weather is nice today, isn't it?",
    "Let's grab lunch sometime this week.",
    "The report is due by Friday.",
    "Happy birthday! Hope you have a great day.",
    "The presentation went well yesterday.",
    "Could you send me the latest version?",
    "I appreciate your feedback on this matter.",
    "The conference call is at 2 PM.",
    "Please let me know if you have any questions.",
    "Looking forward to hearing from you soon.",
]

# Spam indicators for rule-based detection
SPAM_INDICATORS: Set[str] = {
    "urgent",
    "free",
    "money",
    "winner",
    "congratulations",
    "click here",
    "act now",
    "limited time",
    "amazing deals",
    "work from home",
    "lose weight",
    "miracle",
    "hot singles",
    "verify now",
    "final notice",
    "get rich",
    "no experience",
    "doctors hate",
    "one simple trick",
    "suspended",
    "warranty",
    "expires",
    "claim",
    "prize",
    "lottery",
    "selected",
    "special offer",
    "make thousands",
    "no credit card",
    "cancel anytime",
    "risk free",
    "guaranteed",
    "100% free",
    "no cost",
    "trial",
}

SPAM_PATTERNS: Set[str] = {
    "!!!",
    "URGENT",
    "FREE",
    "CLICK HERE",
    "ACT NOW",
    "LIMITED TIME",
    "$$$",
    "100% FREE",
    "NO COST",
    "RISK FREE",
    "GUARANTEED",
    "CONGRATULATIONS",
    "WINNER",
    "SELECTED",
    "CLAIM NOW",
    "EXPIRES",
}


class SpamClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting spam content in text.

    This classifier uses machine learning when scikit-learn is available,
    with fallback to rule-based spam detection. It identifies common
    spam patterns and indicators.

    Attributes:
        threshold: Confidence threshold for spam detection
        model: The trained classification model (if available)
    """

    def __init__(
        self,
        threshold: float = 0.7,
        name: str = "spam",
        description: str = "Detects spam content in text",
    ):
        """Initialize the spam classifier.

        Args:
            threshold: Confidence threshold for spam detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.threshold = threshold
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize and train the spam detection model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline

            # Prepare training data
            X = SPAM_SAMPLES + HAM_SAMPLES
            y = [1] * len(SPAM_SAMPLES) + [0] * len(HAM_SAMPLES)

            # Create and train the model
            model = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=3000,
                            ngram_range=(1, 2),
                            stop_words="english",
                            lowercase=True,
                        ),
                    ),
                    ("classifier", MultinomialNB(alpha=0.1)),
                ]
            )

            # Train the model
            model.fit(X, y)
            self.model = model

            logger.debug(
                f"Initialized spam classifier with ML model",
                extra={
                    "classifier": self.name,
                    "training_samples": len(X),
                    "method": "scikit-learn",
                },
            )

        except ImportError:
            logger.warning(
                "scikit-learn not available. Using rule-based spam detection. "
                "Install scikit-learn for better accuracy: pip install scikit-learn",
                extra={"classifier": self.name},
            )
            self.model = None

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for spam asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with spam prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("ham")

        with self.time_operation("spam_classification") as timer:
            try:
                if self.model is not None:
                    result = await self._classify_with_ml(text)
                else:
                    result = await self._classify_with_rules(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Spam classification completed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "label": result.label,
                        "confidence": result.confidence,
                        "method": result.metadata.get("method", "unknown"),
                    },
                )

                return result

            except Exception as e:
                logger.error(
                    f"Spam classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for spam: {str(e)}",
                    error_code="classification_error",
                    context={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check if scikit-learn is properly installed",
                        "Verify input text is valid",
                        "Try with shorter text",
                    ],
                ) from e

    async def _classify_with_ml(self, text: str) -> ClassificationResult:
        """Classify using machine learning model."""
        if self.model is None:
            raise ValidationError(
                "ML model is not available",
                error_code="dependency_missing",
                suggestions=["Install scikit-learn: pip install scikit-learn"],
            )

        try:
            # Run ML analysis in a thread to avoid blocking
            def analyze():
                probabilities = self.model.predict_proba([text])[0]
                predicted_class = self.model.predict([text])[0]
                return probabilities, predicted_class

            # Use asyncio to run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            probabilities, predicted_class = await loop.run_in_executor(None, analyze)

            confidence = float(probabilities[predicted_class])

            # Map class to label (0 = ham, 1 = spam)
            label = "spam" if predicted_class == 1 else "ham"

            return self.create_classification_result(
                label=label,
                confidence=confidence,
                metadata={
                    "method": "machine_learning",
                    "spam_probability": float(probabilities[1]),
                    "ham_probability": float(probabilities[0]),
                    "threshold": self.threshold,
                    "input_length": len(text),
                },
            )

        except Exception as e:
            # Fallback to rule-based analysis
            logger.warning(
                f"ML spam classification failed, using rule-based analysis: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_rules(text)

    async def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using rule-based approach."""

        def analyze():
            text_lower = text.lower()
            text_upper = text.upper()

            # Count spam indicators
            indicator_count = sum(1 for indicator in SPAM_INDICATORS if indicator in text_lower)

            # Count spam patterns
            pattern_count = sum(1 for pattern in SPAM_PATTERNS if pattern in text_upper)

            # Check for excessive capitalization
            if len(text) > 10:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            else:
                caps_ratio = 0

            # Check for excessive punctuation
            exclamation_count = text.count("!")
            dollar_count = text.count("$")

            # Calculate spam score
            spam_score = 0.0
            spam_score += indicator_count * 0.2
            spam_score += pattern_count * 0.3
            spam_score += caps_ratio * 0.5 if caps_ratio > 0.3 else 0
            spam_score += min(exclamation_count * 0.1, 0.3)
            spam_score += min(dollar_count * 0.15, 0.3)

            return (
                indicator_count,
                pattern_count,
                caps_ratio,
                exclamation_count,
                dollar_count,
                spam_score,
            )

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        indicator_count, pattern_count, caps_ratio, exclamation_count, dollar_count, spam_score = (
            await loop.run_in_executor(None, analyze)
        )

        # Determine label and confidence
        if spam_score > 0.5:
            label = "spam"
            confidence = min(0.6 + spam_score * 0.3, 0.9)
        else:
            label = "ham"
            confidence = max(0.6, 0.9 - spam_score)

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "spam_score": spam_score,
                "indicator_count": indicator_count,
                "pattern_count": pattern_count,
                "caps_ratio": caps_ratio,
                "exclamation_count": exclamation_count,
                "dollar_count": dollar_count,
                "threshold": self.threshold,
                "input_length": len(text),
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["ham", "spam"]


class CachedSpamClassifier(CachedClassifier, TimingMixin):
    """Cached version of SpamClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        threshold: float = 0.7,
        cache_size: int = 128,
        name: str = "cached_spam",
        description: str = "Detects spam content with LRU caching",
    ):
        """Initialize the cached spam classifier.

        Args:
            threshold: Confidence threshold for spam detection
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.threshold = threshold
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize and train the spam detection model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline

            # Prepare training data
            X = SPAM_SAMPLES + HAM_SAMPLES
            y = [1] * len(SPAM_SAMPLES) + [0] * len(HAM_SAMPLES)

            # Create and train the model
            model = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=3000,
                            ngram_range=(1, 2),
                            stop_words="english",
                            lowercase=True,
                        ),
                    ),
                    ("classifier", MultinomialNB(alpha=0.1)),
                ]
            )

            # Train the model
            model.fit(X, y)
            self.model = model

            logger.debug(
                f"Initialized cached spam classifier with ML model", extra={"classifier": self.name}
            )

        except ImportError:
            logger.warning(
                "scikit-learn not available. CachedSpamClassifier will use rule-based detection.",
                extra={"classifier": self.name},
            )
            self.model = None

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform spam classification without caching."""
        try:
            if self.model is not None:
                return self._classify_with_ml_sync(text)
            else:
                return self._classify_with_rules_sync(text)
        except Exception as e:
            logger.error(
                f"Cached spam classification failed",
                extra={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise ValidationError(
                f"Failed to classify text for spam: {str(e)}",
                error_code="classification_error",
                context={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                },
                suggestions=[
                    "Check if scikit-learn is properly installed",
                    "Verify input text is valid",
                    "Try with shorter text",
                ],
            ) from e

    def _classify_with_ml_sync(self, text: str) -> ClassificationResult:
        """Classify using machine learning model (synchronous)."""
        if self.model is None:
            raise ValidationError(
                "ML model is not available",
                error_code="dependency_missing",
                suggestions=["Install scikit-learn: pip install scikit-learn"],
            )

        try:
            probabilities = self.model.predict_proba([text])[0]
            predicted_class = self.model.predict([text])[0]
            confidence = float(probabilities[predicted_class])

            # Map class to label (0 = ham, 1 = spam)
            label = "spam" if predicted_class == 1 else "ham"

            return self.create_classification_result(
                label=label,
                confidence=confidence,
                metadata={
                    "method": "machine_learning",
                    "spam_probability": float(probabilities[1]),
                    "ham_probability": float(probabilities[0]),
                    "threshold": self.threshold,
                    "input_length": len(text),
                    "cached": True,
                },
            )
        except Exception as e:
            logger.warning(
                f"ML spam classification failed, using rule-based analysis: {e}",
                extra={"classifier": self.name},
            )
            return self._classify_with_rules_sync(text)

    def _classify_with_rules_sync(self, text: str) -> ClassificationResult:
        """Classify using rule-based approach (synchronous)."""
        text_lower = text.lower()
        text_upper = text.upper()

        # Count spam indicators
        indicator_count = sum(1 for indicator in SPAM_INDICATORS if indicator in text_lower)

        # Count spam patterns
        pattern_count = sum(1 for pattern in SPAM_PATTERNS if pattern in text_upper)

        # Check for excessive capitalization
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        else:
            caps_ratio = 0

        # Check for excessive punctuation
        exclamation_count = text.count("!")
        dollar_count = text.count("$")

        # Calculate spam score
        spam_score = 0.0
        spam_score += indicator_count * 0.2
        spam_score += pattern_count * 0.3
        spam_score += caps_ratio * 0.5 if caps_ratio > 0.3 else 0
        spam_score += min(exclamation_count * 0.1, 0.3)
        spam_score += min(dollar_count * 0.15, 0.3)

        # Determine label and confidence
        if spam_score > 0.5:
            label = "spam"
            confidence = min(0.6 + spam_score * 0.3, 0.9)
        else:
            label = "ham"
            confidence = max(0.6, 0.9 - spam_score)

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "spam_score": spam_score,
                "indicator_count": indicator_count,
                "pattern_count": pattern_count,
                "caps_ratio": caps_ratio,
                "exclamation_count": exclamation_count,
                "dollar_count": dollar_count,
                "threshold": self.threshold,
                "input_length": len(text),
                "cached": True,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["ham", "spam"]


# Factory functions for easy creation
def create_spam_classifier(
    threshold: float = 0.7,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a spam classifier with the specified parameters.

    Args:
        threshold: Confidence threshold for spam detection
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured spam classifier
    """
    if cached:
        return CachedSpamClassifier(
            threshold=threshold,
            cache_size=cache_size,
        )
    else:
        return SpamClassifier(threshold=threshold)
