"""
Bias validator for detecting bias in text.

This module provides a validator that checks if text contains various forms of bias
using the BiasClassifier from the classifiers module.
"""

from typing import Dict, Any, List, Optional, Union
from ..types import ValidationResult
from ..classifiers.bias import BiasClassifier


class BiasValidator:
    """
    Validator that checks text for various forms of bias.

    This validator uses the BiasClassifier to detect different types of bias in text,
    including gender bias, racial bias, and other forms of discriminatory language.

    By default, it requires the 'scikit-learn' package to be installed.
    Install it with: pip install scikit-learn
    """

    def __init__(
        self,
        threshold: float = 0.7,
        allowed_bias_types: Optional[List[str]] = None,
        detect_all_bias_types: bool = True,
        model_path: Optional[str] = None,
        max_features: int = 3000,
        bias_types: Optional[List[str]] = None,
        bias_keywords: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the bias validator.

        Args:
            threshold: Threshold for bias detection (0.0 to 1.0)
            allowed_bias_types: List of bias types that are allowed (will not fail validation)
            detect_all_bias_types: If True, check for all bias types, not just the disallowed ones
            model_path: Path to a pre-trained model file
            max_features: Maximum number of features for the vectorizer
            bias_types: List of bias types to detect (defaults to BiasClassifier.DEFAULT_BIAS_TYPES)
            bias_keywords: Dictionary of bias types and associated keywords
        """
        self.threshold = threshold
        self.allowed_bias_types = allowed_bias_types or []
        self.detect_all_bias_types = detect_all_bias_types

        # Create the classifier
        self.classifier = BiasClassifier(
            bias_types=bias_types,
            bias_keywords=bias_keywords,
            threshold=threshold,
            model_path=model_path,
            max_features=max_features,
        )

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text for bias.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with bias detection results
        """
        if not text.strip():
            return ValidationResult(
                passed=True,
                message="Empty text passes bias validation",
                score=1.0,
                issues=[],
                suggestions=[],
                metadata={"bias_type": "neutral", "bias_features": {}},
            )

        # Use the classifier to detect bias
        classification = self.classifier.classify(text)

        # Get bias type and confidence
        bias_type = classification.label
        confidence = classification.confidence

        # Get additional information from metadata
        probabilities = classification.metadata.get("probabilities", {})
        bias_features = classification.metadata.get("bias_features", {})

        # Determine if validation passed
        # It passes if bias_type is neutral or in the allowed types
        passed = (
            bias_type == "neutral" or bias_type == "unknown" or bias_type in self.allowed_bias_types
        )

        # Create issues and suggestions
        issues = []
        suggestions = []

        if not passed:
            issues.append(f"Text contains {bias_type} bias")

            if bias_type == "gender":
                suggestions.append("Use gender-neutral language")
                suggestions.append("Avoid gender stereotypes")
            elif bias_type == "racial":
                suggestions.append("Use culturally sensitive language")
                suggestions.append("Avoid racial stereotypes")
            elif bias_type == "age":
                suggestions.append("Avoid age-based stereotypes")
            elif bias_type == "socioeconomic":
                suggestions.append("Use inclusive language regarding social class")
            elif bias_type == "religious":
                suggestions.append("Respect religious diversity")
            elif bias_type == "cultural":
                suggestions.append("Be mindful of cultural differences")
            elif bias_type == "educational":
                suggestions.append("Avoid assumptions about educational background")
            elif bias_type == "geographical":
                suggestions.append("Be inclusive of geographic diversity")
            elif bias_type == "political":
                suggestions.append("Use politically neutral language")
            else:
                suggestions.append(f"Reduce {bias_type} bias in the text")

        # Create message
        if passed:
            if bias_type == "neutral":
                message = "Text does not contain significant bias"
            else:
                message = f"Text contains {bias_type} bias but it's in the allowed categories"
        else:
            message = f"Text contains {bias_type} bias"

        # Calculate score (inverse of confidence if bias is not allowed)
        # Higher score = less biased (1.0 = no bias or allowed bias)
        if passed:
            score = 1.0
        else:
            # Score is inverse of confidence (higher confidence = lower score)
            score = 1.0 - confidence

        # Get probabilities for all bias types if requested
        detected_bias_types = {}
        if self.detect_all_bias_types:
            # Include all bias types with non-zero probability
            for bias_type, prob in probabilities.items():
                if prob > 0.1 and bias_type != "neutral":  # Only include significant probabilities
                    detected_bias_types[bias_type] = prob

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "bias_type": bias_type,
                "confidence": confidence,
                "bias_features": bias_features,
                "detected_bias_types": detected_bias_types,
            },
        )

    def train(self, texts: List[str], labels: List[str], save_path: Optional[str] = None) -> None:
        """
        Train the bias classifier on custom data.

        Args:
            texts: List of training texts
            labels: List of corresponding bias types
            save_path: Optional path to save the trained model
        """
        # Train the classifier
        self.classifier.fit(texts, labels)

        # Save the model if requested
        if save_path:
            self.classifier._save_model(save_path)
