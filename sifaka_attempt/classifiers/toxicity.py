"""
Toxicity classifier for categorizing text as toxic or non-toxic.

This module provides a classifier that categorizes text as toxic or non-toxic
using the Detoxify library.
"""

from typing import Dict, Any, List, Optional, Union, cast
from ..di import inject
from . import ClassificationResult


class ToxicityClassifier:
    """
    Classifier that categorizes text as toxic or non-toxic.

    This classifier uses the Detoxify library to detect various forms of toxic
    content in text, including general toxicity, severe toxicity, obscenity,
    threats, insults, and identity-based attacks.

    By default, it requires the 'detoxify' package to be installed.
    Install it with: pip install detoxify
    """

    # Toxicity categories and their descriptions
    TOXICITY_CATEGORIES = {
        "toxic": "toxic",
        "severe_toxic": "severely toxic",
        "obscene": "obscene",
        "threat": "threatening",
        "insult": "insulting",
        "identity_attack": "identity-attacking",
    }

    # Priority order for labels
    LABEL_PRIORITY = ["severe_toxic", "threat", "identity_attack", "toxic", "insult", "obscene"]

    def __init__(
        self,
        general_threshold: float = 0.5,
        severe_toxic_threshold: float = 0.7,
        threat_threshold: float = 0.7,
        model_name: str = "original",
    ):
        """
        Initialize the toxicity classifier.

        Args:
            general_threshold: General threshold for toxicity detection (0.0 to 1.0)
            severe_toxic_threshold: Threshold for severe toxicity (0.0 to 1.0)
            threat_threshold: Threshold for threats (0.0 to 1.0)
            model_name: Detoxify model to use ('original', 'unbiased', or 'multilingual')
        """
        self.general_threshold = general_threshold
        self.severe_toxic_threshold = severe_toxic_threshold
        self.threat_threshold = threat_threshold
        self.model_name = model_name
        self._model = None  # Lazy-loaded

    @property
    def model(self):
        """Lazy-load the Detoxify model to avoid unnecessary imports."""
        if self._model is None:
            try:
                from detoxify import Detoxify

                self._model = Detoxify(self.model_name)
            except ImportError:
                raise ImportError(
                    "Detoxify is required for ToxicityClassifier. "
                    "Install it with: pip install detoxify"
                )
        return self._model

    def get_threshold_for_category(self, category: str) -> float:
        """Get the threshold for a specific category."""
        if category == "severe_toxic":
            return self.severe_toxic_threshold
        elif category == "threat":
            return self.threat_threshold
        return self.general_threshold

    def _get_toxicity_label(self, scores: Dict[str, float]) -> tuple[str, float]:
        """
        Get the most appropriate toxicity label and confidence.

        Args:
            scores: Dictionary of toxicity scores

        Returns:
            Tuple of (label, confidence)
        """
        # Check for priority categories first
        for category in self.LABEL_PRIORITY:
            if category in scores:
                threshold = self.get_threshold_for_category(category)
                if scores[category] >= threshold:
                    return category, scores[category]

        # If no category exceeds its threshold, find max score
        max_category = max(
            scores.items(), key=lambda x: x[1] if x[1] >= self.general_threshold else 0
        )
        label, confidence = max_category

        # If max score doesn't exceed threshold, return non_toxic
        if confidence < self.general_threshold:
            # High confidence non_toxic for very low scores
            if max(scores.values()) < 0.01:
                return "non_toxic", 0.95
            # Lower confidence non_toxic
            return "non_toxic", 1.0 - confidence

        return label, confidence

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as toxic or non-toxic.

        Args:
            text: The text to classify

        Returns:
            A ClassificationResult with the toxicity label and confidence score
        """
        if not text.strip():
            return ClassificationResult(
                label="non_toxic",
                confidence=1.0,
                message="Empty text is non-toxic",
                metadata={"scores": {}},
            )

        # Get toxicity scores from the model
        try:
            results = self.model.predict(text)
        except Exception as e:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                passed=False,
                message=f"Error analyzing text: {str(e)}",
                metadata={"error": str(e)},
            )

        # Convert results to Python types if needed (some models return numpy arrays)
        scores = {k: float(v) for k, v in results.items()}

        # Determine toxicity label and confidence
        label, confidence = self._get_toxicity_label(scores)

        # Create message based on label
        if label == "non_toxic":
            message = "Text appears non-toxic"
        else:
            category_desc = self.TOXICITY_CATEGORIES.get(label, label)
            message = f"Text contains {category_desc} content"

        return ClassificationResult(
            label=label,
            confidence=confidence,
            message=message,
            metadata={"scores": scores},
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
