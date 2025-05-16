"""
Spam validator for detecting spam content in text.

This module provides a validator that checks if text contains spam content
using the SpamClassifier from the classifiers module.
"""

from typing import Dict, Any, List, Optional, Union
from ..types import ValidationResult
from ..classifiers.spam import SpamClassifier


class SpamValidator:
    """
    Validator that checks text for spam content.

    This validator uses the SpamClassifier to detect spam content in text and
    determines if the text passes validation based on the configured threshold.

    By default, it requires the 'scikit-learn' package to be installed.
    Install it with: pip install scikit-learn
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: Optional[str] = None,
        max_features: int = 1000,
        use_bigrams: bool = True,
    ):
        """
        Initialize the spam validator.

        Args:
            threshold: Threshold for spam detection (0.0 to 1.0)
            model_path: Path to a pre-trained model file
            max_features: Maximum number of features for the vectorizer
            use_bigrams: Whether to use bigrams in addition to unigrams
        """
        self.threshold = threshold

        # Create the classifier
        self.classifier = SpamClassifier(
            model_path=model_path,
            max_features=max_features,
            use_bigrams=use_bigrams,
        )

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text for spam content.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with spam detection results
        """
        if not text.strip():
            return ValidationResult(
                passed=True,
                message="Empty text passes spam validation",
                score=1.0,
                issues=[],
                suggestions=[],
                metadata={"probabilities": {"ham": 1.0, "spam": 0.0}},
            )

        # Use the classifier to get spam information
        classification = self.classifier.classify(text)

        # Get label and confidence
        label = classification.label
        confidence = classification.confidence

        # Get probabilities from metadata
        probabilities = classification.metadata.get("probabilities", {})
        spam_probability = probabilities.get("spam", 0.0)

        # Determine if validation passed (passed if it's ham or spam probability < threshold)
        passed = label == "ham" or spam_probability < self.threshold

        # Create issues and suggestions
        issues = []
        suggestions = []

        if not passed:
            issues.append("Text contains spam content")
            suggestions.append("Remove promotional or unsolicited content")
            suggestions.append("Make text more informative and relevant")

        # Create message
        if passed:
            if label == "ham":
                message = "Text passed spam validation"
            else:
                message = "Text contains promotional content but passes threshold"
        else:
            message = "Text contains spam content"

        # Calculate score (inverse of spam probability)
        # Higher score = less spammy (1.0 = definitely not spam)
        score = 1.0 - spam_probability

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={"label": label, "confidence": confidence, "probabilities": probabilities},
        )

    def train(self, texts: List[str], labels: List[str], save_path: Optional[str] = None) -> None:
        """
        Train the spam classifier on custom data.

        Args:
            texts: List of training texts
            labels: List of corresponding labels ("spam" or "ham")
            save_path: Optional path to save the trained model
        """
        # Train the classifier
        self.classifier.fit(texts, labels)

        # Save the model if requested
        if save_path:
            self.classifier._save_model(save_path)
