"""Toxicity classifier for detecting toxic language in text.

This module provides a classifier for detecting toxic, harmful, or abusive
language using machine learning with fallback to rule-based detection.
"""

from typing import List

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

# Sample toxic text for training
TOXIC_SAMPLES = [
    "You are such an idiot and should kill yourself",
    "I hate you and wish you would die",
    "You're worthless and nobody likes you",
    "Go kill yourself, you piece of trash",
    "You're so stupid, I want to punch you",
    "I hope you get cancer and suffer",
    "You're a waste of space and oxygen",
    "Shut up, you moron, nobody cares",
    "You're disgusting and should be ashamed",
    "I wish violence upon you and your family",
]

NON_TOXIC_SAMPLES = [
    "I disagree with your opinion on this matter",
    "That's an interesting perspective to consider",
    "I think there might be a better approach",
    "Could you please explain your reasoning?",
    "I have a different view on this topic",
    "Let's discuss this in a constructive way",
    "I appreciate your input on this issue",
    "Thank you for sharing your thoughts",
    "I understand your point of view",
    "This is a complex issue worth exploring",
]

# Toxicity indicators for rule-based detection
TOXIC_WORDS = {
    "hate",
    "kill",
    "die",
    "death",
    "murder",
    "violence",
    "hurt",
    "pain",
    "stupid",
    "idiot",
    "moron",
    "dumb",
    "worthless",
    "useless",
    "trash",
    "disgusting",
    "ugly",
    "fat",
    "loser",
    "failure",
    "pathetic",
    "weak",
    "shut up",
    "go away",
    "nobody likes",
    "nobody cares",
    "waste of space",
}

SEVERE_TOXIC_WORDS = {
    "kill yourself",
    "kys",
    "suicide",
    "die in",
    "hope you die",
    "cancer",
    "suffer",
    "torture",
    "abuse",
    "violence upon",
}

THREAT_WORDS = {
    "i will kill",
    "i'll kill",
    "gonna kill",
    "going to hurt",
    "i will hurt",
    "i'll hurt",
    "gonna hurt",
    "beat you up",
    "kick your ass",
    "destroy you",
    "ruin your life",
}


class ToxicityClassifier(TextClassifier):
    """Classifier for detecting toxic language in text.

    This classifier uses machine learning when scikit-learn is available,
    with fallback to rule-based toxicity detection. It identifies various
    forms of toxic language including hate speech, threats, and abuse.

    Attributes:
        general_threshold: Threshold for general toxicity
        severe_threshold: Threshold for severe toxicity
        threat_threshold: Threshold for threats
        model: The trained classification model
    """

    def __init__(
        self,
        general_threshold: float = 0.7,
        severe_threshold: float = 0.8,
        threat_threshold: float = 0.9,
        name: str = "ToxicityClassifier",
        description: str = "Detects toxic language and harmful content",
    ):
        """Initialize the toxicity classifier.

        Args:
            general_threshold: Threshold for general toxicity
            severe_threshold: Threshold for severe toxicity
            threat_threshold: Threshold for threats
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.general_threshold = general_threshold
        self.severe_threshold = severe_threshold
        self.threat_threshold = threat_threshold
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize and train the toxicity detection model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            # Prepare training data
            X = TOXIC_SAMPLES + NON_TOXIC_SAMPLES
            y = [1] * len(TOXIC_SAMPLES) + [0] * len(NON_TOXIC_SAMPLES)

            # Create and train the model
            model = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=5000,
                            ngram_range=(1, 3),
                            stop_words="english",
                            lowercase=True,
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
            model.fit(X, y)
            self.model = model

            logger.debug(f"Initialized toxicity classifier with {len(X)} training samples")

        except ImportError:
            logger.warning(
                "scikit-learn not available. ToxicityClassifier will use rule-based detection. "
                "Install scikit-learn for better accuracy: pip install scikit-learn"
            )
            self.model = None

    def classify(self, text: str) -> ClassificationResult:
        """Classify text for toxicity.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with toxicity prediction

        Raises:
            ClassifierError: If classification fails
        """
        if not text or not text.strip():
            return ClassificationResult(
                label="non_toxic",
                confidence=0.9,
                metadata={"reason": "empty_text", "input_length": 0},
            )

        try:
            if self.model is not None:
                return self._classify_with_ml(text)  # type: ignore[unreachable]
            else:
                return self._classify_with_rules(text)

        except Exception as e:
            logger.error(f"Toxicity classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for toxicity: {str(e)}",
                component="ToxicityClassifier",
                operation="classification",
            )

    def _classify_with_ml(self, text: str) -> ClassificationResult:
        """Classify using machine learning model."""
        if self.model is None:
            raise ClassifierError(
                message="ML model is not available",
                component="ToxicityClassifier",
                operation="ml_classification",
            )

        # Get prediction probabilities
        probabilities = self.model.predict_proba([text])[0]  # type: ignore[unreachable]

        # Get the predicted class (0 = non_toxic, 1 = toxic)
        predicted_class = self.model.predict([text])[0]
        confidence = float(probabilities[predicted_class])

        # Map class to label with severity
        if predicted_class == 1:
            toxicity_score = float(probabilities[1])
            if toxicity_score >= self.threat_threshold:
                label = "threat"
            elif toxicity_score >= self.severe_threshold:
                label = "severe_toxic"
            else:
                label = "toxic"
        else:
            label = "non_toxic"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "machine_learning",
                "toxicity_probability": float(probabilities[1]),
                "non_toxic_probability": float(probabilities[0]),
                "toxicity_score": float(probabilities[1]),
                "input_length": len(text),
            },
        )

    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using rule-based approach."""
        text_lower = text.lower()

        # Count different types of toxic content
        toxic_count = sum(1 for word in TOXIC_WORDS if word in text_lower)
        severe_count = sum(1 for phrase in SEVERE_TOXIC_WORDS if phrase in text_lower)
        threat_count = sum(1 for phrase in THREAT_WORDS if phrase in text_lower)

        # Calculate toxicity scores
        words = text_lower.split()
        word_count = len(words)

        toxic_score = toxic_count / max(1, word_count) * 2
        severe_score = severe_count * 0.5
        threat_score = threat_count * 0.7

        total_score = toxic_score + severe_score + threat_score

        # Determine label and confidence
        if threat_count > 0 or total_score > 0.8:
            label = "threat"
            confidence = min(0.7 + total_score * 0.2, 0.95)
        elif severe_count > 0 or total_score > 0.5:
            label = "severe_toxic"
            confidence = min(0.6 + total_score * 0.3, 0.9)
        elif toxic_count > 0 or total_score > 0.2:
            label = "toxic"
            confidence = min(0.5 + total_score * 0.4, 0.85)
        else:
            label = "non_toxic"
            confidence = max(0.7, 0.95 - total_score)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "toxic_words": toxic_count,
                "severe_words": severe_count,
                "threat_words": threat_count,
                "toxicity_score": total_score,
                "input_length": len(text),
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["non_toxic", "toxic", "severe_toxic", "threat"]


class CachedToxicityClassifier(CachedTextClassifier):
    """Cached version of ToxicityClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        general_threshold: float = 0.7,
        severe_threshold: float = 0.8,
        threat_threshold: float = 0.9,
        cache_size: int = 128,
        name: str = "CachedToxicityClassifier",
        description: str = "Detects toxic language with LRU caching",
    ):
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.general_threshold = general_threshold
        self.severe_threshold = severe_threshold
        self.threat_threshold = threat_threshold
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize and train the toxicity detection model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            # Prepare training data
            X = TOXIC_SAMPLES + NON_TOXIC_SAMPLES
            y = [1] * len(TOXIC_SAMPLES) + [0] * len(NON_TOXIC_SAMPLES)

            # Create and train the model
            model = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=5000,
                            ngram_range=(1, 3),
                            stop_words="english",
                            lowercase=True,
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
            model.fit(X, y)
            self.model = model

            logger.debug(f"Initialized cached toxicity classifier with {len(X)} training samples")

        except ImportError:
            logger.warning(
                "scikit-learn not available. CachedToxicityClassifier will use rule-based detection."
            )
            self.model = None

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform toxicity classification without caching."""
        if not text or not text.strip():
            return ClassificationResult(
                label="non_toxic",
                confidence=0.9,
                metadata={"reason": "empty_text", "input_length": 0},
            )

        try:
            if self.model is not None:
                return self._classify_with_ml(text)  # type: ignore[unreachable]
            else:
                return self._classify_with_rules(text)

        except Exception as e:
            logger.error(f"Cached toxicity classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for toxicity: {str(e)}",
                component="CachedToxicityClassifier",
                operation="classification",
            )

    def _classify_with_ml(self, text: str) -> ClassificationResult:
        """Classify using machine learning model."""
        if self.model is None:
            raise ClassifierError(
                message="ML model is not available",
                component="CachedToxicityClassifier",
                operation="ml_classification",
            )

        probabilities = self.model.predict_proba([text])[0]  # type: ignore[unreachable]
        predicted_class = self.model.predict([text])[0]
        confidence = float(probabilities[predicted_class])

        if predicted_class == 1:
            toxicity_score = float(probabilities[1])
            if toxicity_score >= self.threat_threshold:
                label = "threat"
            elif toxicity_score >= self.severe_threshold:
                label = "severe_toxic"
            else:
                label = "toxic"
        else:
            label = "non_toxic"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "machine_learning",
                "toxicity_probability": float(probabilities[1]),
                "non_toxic_probability": float(probabilities[0]),
                "toxicity_score": float(probabilities[1]),
                "input_length": len(text),
                "cached": True,
            },
        )

    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using rule-based approach."""
        text_lower = text.lower()

        toxic_count = sum(1 for word in TOXIC_WORDS if word in text_lower)
        severe_count = sum(1 for phrase in SEVERE_TOXIC_WORDS if phrase in text_lower)
        threat_count = sum(1 for phrase in THREAT_WORDS if phrase in text_lower)

        words = text_lower.split()
        word_count = len(words)

        toxic_score = toxic_count / max(1, word_count) * 2
        severe_score = severe_count * 0.5
        threat_score = threat_count * 0.7
        total_score = toxic_score + severe_score + threat_score

        if threat_count > 0 or total_score > 0.8:
            label = "threat"
            confidence = min(0.7 + total_score * 0.2, 0.95)
        elif severe_count > 0 or total_score > 0.5:
            label = "severe_toxic"
            confidence = min(0.6 + total_score * 0.3, 0.9)
        elif toxic_count > 0 or total_score > 0.2:
            label = "toxic"
            confidence = min(0.5 + total_score * 0.4, 0.85)
        else:
            label = "non_toxic"
            confidence = max(0.7, 0.95 - total_score)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "toxic_words": toxic_count,
                "severe_words": severe_count,
                "threat_words": threat_count,
                "toxicity_score": total_score,
                "input_length": len(text),
                "cached": True,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["non_toxic", "toxic", "severe_toxic", "threat"]


def create_toxicity_validator(
    threshold: float = 0.7, allow_mild_toxicity: bool = False, name: str = "ToxicityValidator"
) -> ClassifierValidator:
    """Create a validator that detects toxicity in text.

    Args:
        threshold: Confidence threshold for toxicity detection
        allow_mild_toxicity: Whether to allow mild toxicity (only block severe/threats)
        name: Name of the validator

    Returns:
        A ClassifierValidator configured for toxicity detection
    """
    classifier = ToxicityClassifier()

    # Set up invalid labels based on tolerance
    if allow_mild_toxicity:
        invalid_labels = ["severe_toxic", "threat"]
    else:
        invalid_labels = ["toxic", "severe_toxic", "threat"]

    return ClassifierValidator(
        classifier=classifier, threshold=threshold, invalid_labels=invalid_labels, name=name
    )


def create_cached_toxicity_validator(
    threshold: float = 0.7,
    allow_mild_toxicity: bool = False,
    cache_size: int = 128,
    name: str = "CachedToxicityValidator",
) -> ClassifierValidator:
    """Create a cached validator that detects toxicity in text with LRU caching."""
    classifier = CachedToxicityClassifier(cache_size=cache_size)

    if allow_mild_toxicity:
        invalid_labels = ["severe_toxic", "threat"]
    else:
        invalid_labels = ["toxic", "severe_toxic", "threat"]

    return ClassifierValidator(
        classifier=classifier, threshold=threshold, invalid_labels=invalid_labels, name=name
    )
