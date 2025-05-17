"""
Classifier validator for Sifaka.

This module provides a validator that uses classifiers to validate text.
It adapts classifiers to the validator interface, allowing them to be used
in validation chains.
"""

import logging
import time
from typing import List, Optional, Any, Callable
from dataclasses import dataclass, field

from sifaka.results import ValidationResult as SifakaValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator
from sifaka.classifiers import Classifier
from sifaka.validators.base import BaseValidator
from sifaka.utils.error_handling import validation_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


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
    valid_labels: List[str] = field(default_factory=list)
    invalid_labels: Optional[List[str]] = None
    extraction_function: Optional[Callable[[str], str]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        try:
            # Validate threshold
            if self.threshold < 0.0 or self.threshold > 1.0:
                logger.error(f"Invalid threshold value: {self.threshold}")
                raise ValidationError(
                    message=f"Threshold must be between 0.0 and 1.0, got {self.threshold}",
                    component="ClassifierValidatorConfig",
                    operation="initialization",
                    suggestions=[
                        "Set threshold to a value between 0.0 and 1.0",
                        "Common values are 0.5 for balanced sensitivity, or 0.7 for higher confidence",
                    ],
                    metadata={
                        "threshold": self.threshold,
                        "valid_labels": self.valid_labels,
                        "invalid_labels": self.invalid_labels,
                    },
                )

            # Validate that at least one label list is provided
            if not self.valid_labels and not self.invalid_labels:
                logger.error("Neither valid_labels nor invalid_labels provided")
                raise ValidationError(
                    message="Either valid_labels or invalid_labels must be provided",
                    component="ClassifierValidatorConfig",
                    operation="initialization",
                    suggestions=[
                        "Provide a list of valid labels to accept",
                        "Provide a list of invalid labels to reject",
                        "For example, valid_labels=['positive'] or invalid_labels=['negative']",
                    ],
                    metadata={
                        "threshold": self.threshold,
                        "valid_labels": self.valid_labels,
                        "invalid_labels": self.invalid_labels,
                    },
                )

            # Log successful initialization
            logger.debug(
                f"Initialized ClassifierValidatorConfig with threshold={self.threshold}, "
                f"valid_labels={self.valid_labels}, invalid_labels={self.invalid_labels}"
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ClassifierValidatorConfig", operation="initialization")

            # Re-raise as ValidationError with more context if not already a ValidationError
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Failed to initialize ClassifierValidatorConfig: {str(e)}",
                    component="ClassifierValidatorConfig",
                    operation="initialization",
                    suggestions=[
                        "Check the threshold value",
                        "Ensure valid_labels or invalid_labels is provided",
                    ],
                    metadata={
                        "threshold": self.threshold,
                        "valid_labels": self.valid_labels,
                        "invalid_labels": self.invalid_labels,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            raise


class ClassifierValidator(BaseValidator):
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
        name: Optional[str] = None,
    ):
        """
        Initialize the classifier validator.

        Args:
            classifier: The classifier to use for validation.
            threshold: Confidence threshold for accepting a classification.
            valid_labels: List of labels considered valid.
            invalid_labels: Optional list of labels considered invalid.
            extraction_function: Optional function to extract text for classification.
            name: Optional name for the validator.

        Raises:
            ValidationError: If the configuration is invalid.
        """
        # Log initialization attempt
        logger.debug(
            f"Initializing ClassifierValidator with classifier={classifier.__class__.__name__}, "
            f"threshold={threshold}, valid_labels={valid_labels}, invalid_labels={invalid_labels}"
        )

        try:
            # Validate classifier
            if not classifier:
                logger.error("No classifier provided to ClassifierValidator")
                raise ValidationError(
                    message="Classifier must be provided",
                    component="ClassifierValidator",
                    operation="initialization",
                    suggestions=[
                        "Provide a valid classifier instance",
                        "Check that the classifier implements the Classifier protocol",
                    ],
                    metadata={
                        "threshold": threshold,
                        "valid_labels": valid_labels,
                        "invalid_labels": invalid_labels,
                    },
                )

            # Initialize the base validator with a name
            with validation_context(
                validator_name=name or f"{classifier.__class__.__name__}Validator",
                operation="initialization",
                message_prefix="Failed to initialize ClassifierValidator",
                suggestions=[
                    "Check the classifier implementation",
                    "Verify that the configuration parameters are valid",
                ],
                metadata={
                    "classifier_type": classifier.__class__.__name__,
                    "threshold": threshold,
                    "valid_labels": valid_labels,
                    "invalid_labels": invalid_labels,
                },
            ):
                super().__init__(name=name or f"{classifier.__class__.__name__}Validator")
                logger.debug(f"Successfully initialized base validator with name={self.name}")

            # Store the classifier
            self.classifier = classifier

            # Create and validate configuration
            with validation_context(
                validator_name=self.name,
                operation="configuration",
                message_prefix="Failed to create validator configuration",
                suggestions=[
                    "Check the threshold value",
                    "Ensure valid_labels or invalid_labels is provided",
                ],
                metadata={
                    "threshold": threshold,
                    "valid_labels": valid_labels,
                    "invalid_labels": invalid_labels,
                },
            ):
                # Ensure valid_labels is not None before passing to ClassifierValidatorConfig
                self.config = ClassifierValidatorConfig(
                    threshold=threshold,
                    valid_labels=valid_labels if valid_labels is not None else [],
                    invalid_labels=invalid_labels,
                    extraction_function=extraction_function,
                )
                logger.debug(f"Successfully created and validated configuration for {self.name}")

            # Log successful initialization
            logger.debug(
                f"Successfully initialized {self.name} with threshold={threshold}, "
                f"valid_labels={valid_labels}, invalid_labels={invalid_labels}"
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ClassifierValidator", operation="initialization")

            # Re-raise as ValidationError with more context if not already a ValidationError
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Failed to initialize ClassifierValidator: {str(e)}",
                    component="ClassifierValidator",
                    operation="initialization",
                    suggestions=[
                        "Check the classifier implementation",
                        "Verify that the configuration parameters are valid",
                        "Ensure the threshold is between 0.0 and 1.0",
                        "Provide either valid_labels or invalid_labels",
                    ],
                    metadata={
                        "classifier_type": classifier.__class__.__name__ if classifier else None,
                        "threshold": threshold,
                        "valid_labels": valid_labels,
                        "invalid_labels": invalid_labels,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            raise

    def _validate(self, text: str) -> SifakaValidationResult:
        """
        Validate text using the classifier.

        Args:
            text: The text to validate.

        Returns:
            A ValidationResult indicating whether the text passed validation.

        Raises:
            ValidationError: If validation fails due to an error.
        """
        start_time = time.time()

        # Log validation attempt
        logger.debug(
            f"{self.name}: Validating text of length {len(text)}, "
            f"threshold={self.config.threshold}, "
            f"valid_labels={self.config.valid_labels}, "
            f"invalid_labels={self.config.invalid_labels}"
        )

        try:
            # Extract text to classify if an extraction function is provided
            text_to_classify = text
            if self.config.extraction_function:
                with validation_context(
                    validator_name=self.name,
                    operation="text extraction",
                    message_prefix="Failed to extract text for classification",
                    suggestions=[
                        "Check the extraction function implementation",
                        "Verify that the text is in the expected format",
                    ],
                    metadata={
                        "text_length": len(text),
                        "classifier_type": self.classifier.__class__.__name__,
                    },
                ):
                    text_to_classify = self.config.extraction_function(text)
                    logger.debug(
                        f"{self.name}: Extracted text for classification, original length={len(text)}, extracted length={len(text_to_classify)}"
                    )

            # Classify the text
            with validation_context(
                validator_name=self.name,
                operation="classification",
                message_prefix=f"Failed to classify text using {self.classifier.__class__.__name__}",
                suggestions=[
                    "Check the classifier implementation",
                    "Verify that the text is in the expected format",
                    "Ensure the classifier is properly initialized",
                ],
                metadata={
                    "text_length": len(text_to_classify),
                    "classifier_type": self.classifier.__class__.__name__,
                    "threshold": self.config.threshold,
                },
            ):
                result = self.classifier.classify(text_to_classify)
                logger.debug(
                    f"{self.name}: Classified text as '{result.label}' with confidence {result.confidence:.2f}"
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Create details for the validation result
            details = {
                "validator_name": self.name,
                "classifier": self.classifier.__class__.__name__,
                "label": result.label,
                "confidence": result.confidence,
                "threshold": self.config.threshold,
                "valid_labels": self.config.valid_labels,
                "invalid_labels": self.config.invalid_labels,
                "processing_time_ms": processing_time,
            }

            # Add classifier-specific metadata if available
            if hasattr(result, "metadata") and result.metadata:
                details["classifier_metadata"] = result.metadata

            # Check confidence threshold
            if result.confidence < self.config.threshold:
                logger.debug(
                    f"{self.name}: Classification confidence {result.confidence:.2f} below threshold {self.config.threshold:.2f}"
                )

                # Calculate score as a ratio of confidence to threshold
                score = max(0.0, min(1.0, result.confidence / self.config.threshold))

                return SifakaValidationResult(
                    passed=False,
                    message=f"Classification confidence ({result.confidence:.2f}) below threshold ({self.config.threshold:.2f})",
                    details=details,
                    score=score,
                    issues=[
                        f"Classification confidence {result.confidence:.2f} is below threshold {self.config.threshold:.2f}"
                    ],
                    suggestions=[
                        f"Modify the text to increase {result.label} classification confidence",
                        "Consider lowering the confidence threshold if appropriate",
                    ],
                )

            # Check if label is in valid labels
            if self.config.valid_labels and result.label in self.config.valid_labels:
                logger.debug(
                    f"{self.name}: Text classified as '{result.label}', which is in valid labels"
                )

                # Calculate score based on confidence
                score = max(0.5, min(1.0, result.confidence))

                return SifakaValidationResult(
                    passed=True,
                    message=f"Text classified as '{result.label}' with confidence {result.confidence:.2f}",
                    details=details,
                    score=score,
                    issues=[],
                    suggestions=[],
                )

            # Check if label is in invalid labels
            if self.config.invalid_labels and result.label in self.config.invalid_labels:
                logger.debug(
                    f"{self.name}: Text classified as '{result.label}', which is in invalid labels"
                )

                # Calculate score inversely proportional to confidence
                score = max(0.0, min(0.5, 1.0 - result.confidence))

                return SifakaValidationResult(
                    passed=False,
                    message=f"Text classified as invalid label '{result.label}'",
                    details=details,
                    score=score,
                    issues=[f"Text classified as '{result.label}', which is not allowed"],
                    suggestions=[
                        f"Modify the text to avoid classification as: {result.label}",
                        "Consider rephrasing the content to change the classification",
                    ],
                )

            # If valid_labels is provided but label is not in it
            if self.config.valid_labels:
                valid_labels_str = ", ".join(self.config.valid_labels)
                logger.debug(
                    f"{self.name}: Text classified as '{result.label}', which is not in valid labels: {valid_labels_str}"
                )

                # Calculate score based on similarity to valid labels (placeholder logic)
                score = 0.0

                return SifakaValidationResult(
                    passed=False,
                    message=f"Text classified as '{result.label}', which is not in the list of valid labels: {valid_labels_str}",
                    details=details,
                    score=score,
                    issues=[
                        f"Text classified as '{result.label}', expected one of: {valid_labels_str}"
                    ],
                    suggestions=[
                        f"Modify the text to align with one of these classifications: {valid_labels_str}",
                        "Consider rephrasing the content to change the classification",
                    ],
                )

            # If invalid_labels is provided but label is not in it
            if self.config.invalid_labels:
                logger.debug(
                    f"{self.name}: Text classified as '{result.label}', which is not in invalid labels"
                )

                # Calculate score based on confidence and distance from invalid labels
                score = max(0.5, min(1.0, result.confidence))

                return SifakaValidationResult(
                    passed=True,
                    message=f"Text classified as '{result.label}', which is not in the list of invalid labels",
                    details=details,
                    score=score,
                    issues=[],
                    suggestions=[],
                )

            # Default case (should not happen if config is valid)
            logger.warning(
                f"{self.name}: Reached default case in validation logic, which should not happen with valid config"
            )

            return SifakaValidationResult(
                passed=True,
                message=f"Text classified as '{result.label}' with confidence {result.confidence:.2f}",
                details=details,
                score=result.confidence,
                issues=[],
                suggestions=[],
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ClassifierValidator", operation="validation")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Create error details
            error_details = {
                "validator_name": self.name,
                "classifier": self.classifier.__class__.__name__,
                "threshold": self.config.threshold,
                "valid_labels": self.config.valid_labels,
                "invalid_labels": self.config.invalid_labels,
                "text_length": len(text),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_time_ms": processing_time,
            }

            # Raise as ValidationError with more context
            raise ValidationError(
                message=f"Failed to validate text using classifier: {str(e)}",
                component="ClassifierValidator",
                operation="validation",
                suggestions=[
                    "Check the classifier implementation",
                    "Verify that the text is in the expected format",
                    "Ensure the classifier is properly initialized",
                    "Check if the text is too long or contains unsupported characters",
                ],
                metadata=error_details,
            )


@register_validator("classifier")
def create_classifier_validator(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
    name: Optional[str] = None,
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
        name: Optional name for the validator.
        **options: Additional options (ignored).

    Returns:
        A ClassifierValidator instance.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    start_time = time.time()

    # Log factory function call
    logger.debug(
        f"Creating classifier validator with classifier={classifier.__class__.__name__ if classifier else None}, "
        f"threshold={threshold}, valid_labels={valid_labels}, invalid_labels={invalid_labels}"
    )

    try:
        # Validate classifier
        if not classifier:
            logger.error("No classifier provided to create_classifier_validator")
            raise ValidationError(
                message="Classifier must be provided",
                component="ClassifierValidatorFactory",
                operation="creation",
                suggestions=[
                    "Provide a valid classifier instance",
                    "Check that the classifier implements the Classifier protocol",
                ],
                metadata={
                    "threshold": threshold,
                    "valid_labels": valid_labels,
                    "invalid_labels": invalid_labels,
                },
            )

        # Create the validator
        validator = ClassifierValidator(
            classifier=classifier,
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
            name=name or options.get("name"),
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(
            f"Successfully created classifier validator: {validator.name} in {processing_time:.2f}ms"
        )

        return validator

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="ClassifierValidatorFactory", operation="creation")

        # Re-raise as ValidationError with more context if not already a ValidationError
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create classifier validator: {str(e)}",
                component="ClassifierValidatorFactory",
                operation="creation",
                suggestions=[
                    "Check the classifier implementation",
                    "Verify that the threshold is between 0.0 and 1.0",
                    "Ensure valid_labels or invalid_labels is provided",
                    "Check that the extraction function is properly implemented",
                ],
                metadata={
                    "classifier_type": classifier.__class__.__name__ if classifier else None,
                    "threshold": threshold,
                    "valid_labels": valid_labels,
                    "invalid_labels": invalid_labels,
                    "has_extraction_function": extraction_function is not None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise


def classifier_validator(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
    name: Optional[str] = None,
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
        name: Optional name for the validator.

    Returns:
        A ClassifierValidator instance.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    start_time = time.time()

    # Log function call
    logger.debug(
        f"Creating classifier validator with classifier={classifier.__class__.__name__ if classifier else None}, "
        f"threshold={threshold}, valid_labels={valid_labels}, invalid_labels={invalid_labels}"
    )

    try:
        # Validate classifier
        if not classifier:
            logger.error("No classifier provided to classifier_validator")
            raise ValidationError(
                message="Classifier must be provided",
                component="ClassifierValidatorFunction",
                operation="creation",
                suggestions=[
                    "Provide a valid classifier instance",
                    "Check that the classifier implements the Classifier protocol",
                ],
                metadata={
                    "threshold": threshold,
                    "valid_labels": valid_labels,
                    "invalid_labels": invalid_labels,
                },
            )

        # Create the validator
        validator = ClassifierValidator(
            classifier=classifier,
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
            name=name,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(
            f"Successfully created classifier validator: {validator.name} in {processing_time:.2f}ms"
        )

        return validator

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="ClassifierValidatorFunction", operation="creation")

        # Re-raise as ValidationError with more context if not already a ValidationError
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create classifier validator: {str(e)}",
                component="ClassifierValidatorFunction",
                operation="creation",
                suggestions=[
                    "Check the classifier implementation",
                    "Verify that the threshold is between 0.0 and 1.0",
                    "Ensure valid_labels or invalid_labels is provided",
                    "Check that the extraction function is properly implemented",
                ],
                metadata={
                    "classifier_type": classifier.__class__.__name__ if classifier else None,
                    "threshold": threshold,
                    "valid_labels": valid_labels,
                    "invalid_labels": invalid_labels,
                    "has_extraction_function": extraction_function is not None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
