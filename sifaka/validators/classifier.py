"""
Classifier validator for Sifaka.

This module provides a validator that uses classifiers to validate text.
It adapts classifiers to the validator interface, allowing them to be used
in validation chains.
"""

from typing import List, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass

from sifaka.results import ValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator
from sifaka.classifiers import Classifier, ClassificationResult


@dataclass
class ClassifierValidatorConfig:
    """
    Configuration for classifier validators.

    Attributes:
        threshold: Confidence threshold for accepting a classification (0.0 to 1.0).
        valid_labels: List of labels considered valid.
        invalid_labels: Optional list of labels considered invalid.
        extraction_function: Optional function to extract text for classification.
    """

    threshold: float = 0.5
    valid_labels: List[str] = None
    invalid_labels: Optional[List[str]] = None
    extraction_function: Optional[Callable[[str], str]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValidationError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")
        
        if self.valid_labels is None:
            self.valid_labels = []
        
        if not self.valid_labels and not self.invalid_labels:
            raise ValidationError("Either valid_labels or invalid_labels must be provided")


class ClassifierValidator:
    """
    Validator that uses a classifier for validation.

    This validator adapts a classifier to the validator interface, allowing
    classifiers to be used in validation chains. It validates text by classifying
    it and checking if the classification meets the configured criteria.

    Attributes:
        classifier: The classifier to use for validation.
        config: The validator configuration.
    """

    def __init__(
        self,
        classifier: Classifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        invalid_labels: Optional[List[str]] = None,
        extraction_function: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the classifier validator.

        Args:
            classifier: The classifier to use for validation.
            threshold: Confidence threshold for accepting a classification.
            valid_labels: List of labels considered valid.
            invalid_labels: Optional list of labels considered invalid.
            extraction_function: Optional function to extract text for classification.

        Raises:
            ValidationError: If the configuration is invalid.
        """
        self.classifier = classifier
        self.config = ClassifierValidatorConfig(
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text using the classifier.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text passed validation.

        Raises:
            ValidationError: If validation fails due to an error.
        """
        try:
            if not text:
                return ValidationResult(
                    passed=False,
                    message="Input text is empty",
                    details={"input_length": 0},
                )

            # Extract text to classify if an extraction function is provided
            text_to_classify = text
            if self.config.extraction_function:
                text_to_classify = self.config.extraction_function(text)

            # Classify the text
            result = self.classifier.classify(text_to_classify)

            # Check if the classification meets the criteria
            details = {
                "label": result.label,
                "confidence": result.confidence,
                "threshold": self.config.threshold,
                "valid_labels": self.config.valid_labels,
                "invalid_labels": self.config.invalid_labels,
            }

            # Check confidence threshold
            if result.confidence < self.config.threshold:
                return ValidationResult(
                    passed=False,
                    message=f"Classification confidence ({result.confidence:.2f}) below threshold ({self.config.threshold:.2f})",
                    details=details,
                )

            # Check if label is in valid labels
            if self.config.valid_labels and result.label in self.config.valid_labels:
                return ValidationResult(
                    passed=True,
                    message=f"Text classified as {result.label} with confidence {result.confidence:.2f}",
                    details=details,
                )

            # Check if label is in invalid labels
            if self.config.invalid_labels and result.label in self.config.invalid_labels:
                return ValidationResult(
                    passed=False,
                    message=f"Text classified as invalid label {result.label}",
                    details=details,
                )

            # If valid_labels is provided but label is not in it
            if self.config.valid_labels:
                return ValidationResult(
                    passed=False,
                    message=f"Text classified as {result.label}, which is not in the list of valid labels",
                    details=details,
                )

            # If invalid_labels is provided but label is not in it
            if self.config.invalid_labels:
                return ValidationResult(
                    passed=True,
                    message=f"Text classified as {result.label}, which is not in the list of invalid labels",
                    details=details,
                )

            # Default case (should not happen if config is valid)
            return ValidationResult(
                passed=True,
                message=f"Text classified as {result.label} with confidence {result.confidence:.2f}",
                details=details,
            )

        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}")


@register_validator("classifier")
def create_classifier_validator(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
    **options: Any,
) -> ClassifierValidator:
    """
    Create a classifier validator.

    This factory function creates a ClassifierValidator with the specified parameters.
    It is registered with the registry system for dependency injection.

    Args:
        classifier: The classifier to use for validation.
        threshold: Confidence threshold for accepting a classification.
        valid_labels: List of labels considered valid.
        invalid_labels: Optional list of labels considered invalid.
        extraction_function: Optional function to extract text for classification.
        **options: Additional options (ignored).

    Returns:
        A ClassifierValidator instance.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    return ClassifierValidator(
        classifier=classifier,
        threshold=threshold,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        extraction_function=extraction_function,
    )


def classifier_validator(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
) -> ClassifierValidator:
    """
    Create a classifier validator.

    This is a convenience function for creating a ClassifierValidator.

    Args:
        classifier: The classifier to use for validation.
        threshold: Confidence threshold for accepting a classification.
        valid_labels: List of labels considered valid.
        invalid_labels: Optional list of labels considered invalid.
        extraction_function: Optional function to extract text for classification.

    Returns:
        A ClassifierValidator instance.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    return ClassifierValidator(
        classifier=classifier,
        threshold=threshold,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        extraction_function=extraction_function,
    )
