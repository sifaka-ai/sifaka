"""Classifier validator for Sifaka.

This module provides a ClassifierValidator that uses machine learning classifiers
to validate text properties such as sentiment, toxicity, bias, or other characteristics.
It supports any classifier that follows the scikit-learn interface.

The ClassifierValidator is designed to leverage ML models for sophisticated text
validation beyond simple pattern matching.
"""

import time
from typing import Any, Callable, List, Optional, Protocol

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.shared import ClassifierValidatorBase

# Configure logger
logger = get_logger(__name__)


class Classifier(Protocol):
    """Protocol for classifiers that can be used with ClassifierValidator.

    This protocol defines the interface that classifiers must implement to be
    compatible with the ClassifierValidator. It follows the scikit-learn interface.
    """

    def predict(self, X: List[str]) -> List[Any]:
        """Predict class labels for samples.

        Args:
            X: List of text samples to classify.

        Returns:
            List of predicted class labels.
        """
        ...

    def predict_proba(self, X: List[str]) -> List[List[float]]:
        """Predict class probabilities for samples.

        Args:
            X: List of text samples to classify.

        Returns:
            List of probability arrays for each sample.
        """
        ...


class ClassifierValidator(ClassifierValidatorBase):
    """Validator that uses machine learning classifiers to validate text.

    This validator uses ML classifiers to validate text properties such as sentiment,
    toxicity, bias, or other characteristics. It supports any classifier that follows
    the scikit-learn interface. It extends ClassifierValidatorBase to reduce code duplication.

    Attributes:
        classifier: The ML classifier to use for validation.
        threshold: Confidence threshold for accepting a classification.
        valid_labels: List of labels considered valid.
        invalid_labels: List of labels considered invalid.
        extraction_function: Optional function to extract text for classification.
        name: The name of the validator.
    """

    def __init__(
        self,
        classifier: Classifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        invalid_labels: Optional[List[str]] = None,
        expected_label: Optional[str] = None,
        extraction_function: Optional[Callable[[str], str]] = None,
        name: str = "ClassifierValidator",
    ):
        """Initialize the validator.

        Args:
            classifier: The ML classifier to use for validation.
            threshold: Confidence threshold for accepting a classification.
            valid_labels: List of labels considered valid.
            invalid_labels: List of labels considered invalid.
            expected_label: Single expected label (converted to valid_labels).
            extraction_function: Optional function to extract text for classification.
            name: The name of the validator.

        Raises:
            ValidationError: If the configuration is invalid.
        """
        # Handle expected_label as alias for valid_labels
        if expected_label is not None:
            if valid_labels is not None:
                raise ValidationError(
                    message="Cannot specify both valid_labels and expected_label",
                    component="ClassifierValidator",
                    operation="initialization",
                    suggestions=["Use either valid_labels or expected_label, not both"],
                )
            valid_labels = [expected_label]
        if not (0.0 <= threshold <= 1.0):
            raise ValidationError(
                message=f"Threshold must be between 0.0 and 1.0, got {threshold}",
                component="ClassifierValidator",
                operation="initialization",
                suggestions=["Provide a threshold value between 0.0 and 1.0"],
            )

        if valid_labels is None and invalid_labels is None:
            raise ValidationError(
                message="Either valid_labels or invalid_labels must be specified",
                component="ClassifierValidator",
                operation="initialization",
                suggestions=[
                    "Provide valid_labels (list of acceptable labels)",
                    "Provide invalid_labels (list of unacceptable labels)",
                    "Provide both for more precise control",
                ],
            )

        # Initialize the base class
        super().__init__(
            classifier=classifier,
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            name=name,
        )

        # Store additional attributes specific to this implementation
        self.extraction_function = extraction_function

    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text using the classifier with custom text extraction.

        This method overrides the base class to add text extraction functionality
        while leveraging the shared validation patterns.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with the classification validation outcome.
        """
        start_time = time.time()

        # Extract text for classification if function provided
        text_to_classify = thought.text or ""
        if self.extraction_function:
            try:
                text_to_classify = self.extraction_function(thought.text or "")
            except Exception as e:
                logger.error(f"{self.name}: Text extraction failed: {e}")
                return self.create_validation_result(
                    passed=False,
                    message=f"Text extraction failed: {str(e)}",
                    issues=[f"Extraction function error: {str(e)}"],
                    suggestions=["Check the extraction function implementation"],
                )

        try:
            # Check if classifier has a classify method (test-style API)
            if hasattr(self.classifier, "classify"):
                result = self.classifier.classify(text_to_classify)
                # Handle both dict-style and object-style results
                if hasattr(result, "label") and hasattr(result, "confidence"):
                    # ClassificationResult object
                    predicted_label = result.label
                    max_confidence = result.confidence
                else:
                    # Dict-style result
                    predicted_label = result["label"]
                    max_confidence = result["confidence"]
            else:
                # Use sklearn-style API
                predictions = self.classifier.predict([text_to_classify])
                probabilities = self.classifier.predict_proba([text_to_classify])

                predicted_label = predictions[0]
                prediction_probs = probabilities[0]
                max_confidence = max(prediction_probs)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Check if confidence meets threshold
            if max_confidence < self.threshold:
                logger.debug(
                    f"{self.name}: Low confidence prediction ({max_confidence:.3f} < {self.threshold}) "
                    f"in {processing_time:.2f}ms"
                )
                return self.create_validation_result(
                    passed=False,
                    message=f"Classification confidence too low: {max_confidence:.3f}",
                    score=max_confidence,
                    issues=[
                        f"Prediction confidence ({max_confidence:.3f}) below threshold ({self.threshold})"
                    ],
                    suggestions=[
                        "Provide clearer, more definitive text",
                        "Consider lowering the confidence threshold",
                    ],
                )

            # Check against valid/invalid labels using base class logic
            is_valid = self._check_label_validity(predicted_label)

            if is_valid:
                logger.debug(
                    f"{self.name}: Validation passed - label '{predicted_label}' "
                    f"with confidence {max_confidence:.3f} in {processing_time:.2f}ms"
                )
                return self.create_validation_result(
                    passed=True,
                    message=f"Text classified as '{predicted_label}' with confidence {max_confidence:.3f}",
                    score=max_confidence,
                )
            else:
                logger.debug(
                    f"{self.name}: Validation failed - label '{predicted_label}' "
                    f"with confidence {max_confidence:.3f} in {processing_time:.2f}ms"
                )
                return self.create_validation_result(
                    passed=False,
                    message=f"Text classified as invalid label '{predicted_label}'",
                    score=1.0 - max_confidence,  # Invert score for invalid labels
                    issues=[f"Text classified as '{predicted_label}' which is not allowed"],
                    suggestions=[
                        f"Modify text to avoid classification as '{predicted_label}'",
                        f"Ensure text aligns with valid categories: {self.valid_labels}",
                    ],
                )

        except Exception as e:
            # Let the base class handle the error using the mixin
            return self.create_error_result(e, self.name, "classification")

    def _check_label_validity(self, label: str) -> bool:
        """Check if a predicted label is valid.

        Args:
            label: The predicted label to check.

        Returns:
            True if the label is valid, False otherwise.
        """
        # If invalid_labels is specified, check if label is in it
        if self.invalid_labels and label in self.invalid_labels:
            return False

        # If valid_labels is specified, check if label is in it
        if self.valid_labels and label not in self.valid_labels:
            return False

        # If only invalid_labels specified and label not in it, it's valid
        # If only valid_labels specified and label in it, it's valid
        # If both specified, label must be in valid and not in invalid
        return True

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        """Validate text using ML classifier asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync validate method but can be called concurrently with other validators.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # Classification is CPU-bound and fast, so we can just call the sync version
        # In a real implementation, you might want to run this in a thread pool for consistency
        return self.validate(thought)


def create_classifier_validator(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
    name: str = "ClassifierValidator",
) -> ClassifierValidator:
    """Create a classifier validator.

    Args:
        classifier: The ML classifier to use for validation.
        threshold: Confidence threshold for accepting a classification.
        valid_labels: List of labels considered valid.
        invalid_labels: List of labels considered invalid.
        extraction_function: Optional function to extract text for classification.
        name: The name of the validator.

    Returns:
        A ClassifierValidator instance.
    """
    return ClassifierValidator(
        classifier=classifier,
        threshold=threshold,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        extraction_function=extraction_function,
        name=name,
    )


def classifier_validator(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
) -> ClassifierValidator:
    """Create a classifier validator.

    This is a convenience function for creating a ClassifierValidator.

    Args:
        classifier: The ML classifier to use for validation.
        threshold: Confidence threshold for accepting a classification.
        valid_labels: List of labels considered valid.
        invalid_labels: List of labels considered invalid.
        extraction_function: Optional function to extract text for classification.

    Returns:
        A ClassifierValidator instance.
    """
    return create_classifier_validator(
        classifier=classifier,
        threshold=threshold,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        extraction_function=extraction_function,
        name="MLClassifierValidator",
    )
