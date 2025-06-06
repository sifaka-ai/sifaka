"""Classifier-based validator for Sifaka.

This module provides a validator that uses classifiers to validate text
against specific classification criteria. For example, using a sentiment
classifier to ensure text has positive sentiment.
"""

from typing import List, Optional, Union
import asyncio

from sifaka.core.thought import SifakaThought
from sifaka.classifiers.base import BaseClassifier, ClassificationResult
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.base import BaseValidator, ValidationResult, TimingMixin

logger = get_logger(__name__)


class ClassifierValidator(BaseValidator, TimingMixin):
    """Validator that uses a classifier to validate text.

    This validator runs a classifier on text and validates the result
    against specified criteria such as required labels, confidence thresholds,
    or forbidden labels.

    Attributes:
        classifier: The classifier to use for validation
        threshold: Minimum confidence threshold for classification
        valid_labels: List of labels that are considered valid (None = all allowed)
        invalid_labels: List of labels that are considered invalid (None = none forbidden)
        strict: Whether to fail validation on any violation
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        invalid_labels: Optional[List[str]] = None,
        strict: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the classifier validator.

        Args:
            classifier: The classifier to use for validation
            threshold: Minimum confidence threshold for classification
            valid_labels: List of labels that are considered valid
            invalid_labels: List of labels that are considered invalid
            strict: Whether to fail validation on any violation
            name: Custom name for the validator
            description: Custom description for the validator

        Raises:
            ValidationError: If configuration is invalid
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValidationError(
                f"Threshold must be between 0.0 and 1.0, got {threshold}",
                error_code="invalid_config",
                context={"threshold": threshold},
                suggestions=["Use a threshold between 0.0 and 1.0"],
            )

        if valid_labels and invalid_labels:
            # Check for overlap
            overlap = set(valid_labels) & set(invalid_labels)
            if overlap:
                raise ValidationError(
                    f"Labels cannot be both valid and invalid: {overlap}",
                    error_code="invalid_config",
                    context={"overlap": list(overlap)},
                    suggestions=["Remove overlapping labels from one of the lists"],
                )

        # Set default name and description
        if name is None:
            name = f"classifier_{classifier.name}"

        if description is None:
            parts = []
            if valid_labels:
                parts.append(f"requires labels: {valid_labels}")
            if invalid_labels:
                parts.append(f"forbids labels: {invalid_labels}")
            if threshold > 0.0:
                parts.append(f"min confidence: {threshold}")

            if parts:
                description = f"Validates using {classifier.name} classifier ({', '.join(parts)})"
            else:
                description = f"Validates using {classifier.name} classifier"

        super().__init__(name=name, description=description)

        self.classifier = classifier
        self.threshold = threshold
        self.valid_labels = valid_labels
        self.invalid_labels = invalid_labels
        self.strict = strict

        logger.debug(
            f"Created ClassifierValidator",
            extra={
                "validator_name": self.name,
                "classifier_name": classifier.name,
                "threshold": self.threshold,
                "valid_labels": self.valid_labels,
                "invalid_labels": self.invalid_labels,
                "strict": self.strict,
            },
        )

    async def validate_async(self, thought: SifakaThought) -> ValidationResult:
        """Validate text using the classifier.

        Args:
            thought: The SifakaThought to validate

        Returns:
            ValidationResult with classifier-based validation information
        """
        # Check if we have text to validate
        text = thought.current_text
        if not text:
            logger.debug(
                f"Classifier validation failed: no text",
                extra={"validator": self.name, "thought_id": thought.id},
            )
            return self.create_empty_text_result()

        with self.time_operation("classifier_validation") as timer:
            try:
                # Run classification
                classification_result = await self.classifier.classify_async(text)

                # Validate classification result
                issues = []
                suggestions = []
                violations = 0

                # Check confidence threshold
                if classification_result.confidence < self.threshold:
                    violations += 1
                    issues.append(
                        f"Classification confidence {classification_result.confidence:.3f} "
                        f"below threshold {self.threshold}"
                    )
                    suggestions.append(
                        f"Improve text to increase {self.classifier.name} confidence"
                    )

                # Check valid labels
                if self.valid_labels and classification_result.label not in self.valid_labels:
                    violations += 1
                    issues.append(
                        f"Classification label '{classification_result.label}' "
                        f"not in valid labels: {self.valid_labels}"
                    )
                    suggestions.append(
                        f"Modify text to achieve one of: {', '.join(self.valid_labels)}"
                    )

                # Check invalid labels
                if self.invalid_labels and classification_result.label in self.invalid_labels:
                    violations += 1
                    issues.append(
                        f"Classification label '{classification_result.label}' "
                        f"is in forbidden labels: {self.invalid_labels}"
                    )
                    suggestions.append(f"Modify text to avoid: {', '.join(self.invalid_labels)}")

                # Determine if validation passed
                passed = violations == 0

                # Calculate score
                if passed:
                    score = 1.0
                elif self.strict:
                    score = 0.0
                else:
                    # Proportional score based on confidence and violations
                    confidence_score = classification_result.confidence
                    violation_penalty = violations * 0.3
                    score = max(0.1, confidence_score - violation_penalty)

                # Create result message
                if passed:
                    message = (
                        f"Classifier validation passed: {classification_result.label} "
                        f"(confidence: {classification_result.confidence:.3f})"
                    )
                else:
                    message = f"Classifier validation failed: {violations} violation(s)"

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)

                result = self.create_validation_result(
                    passed=passed,
                    message=message,
                    score=score,
                    issues=issues,
                    suggestions=suggestions,
                    metadata={
                        "classifier_name": self.classifier.name,
                        "classification_label": classification_result.label,
                        "classification_confidence": classification_result.confidence,
                        "classification_metadata": classification_result.metadata,
                        "threshold": self.threshold,
                        "valid_labels": self.valid_labels,
                        "invalid_labels": self.invalid_labels,
                        "violations": violations,
                        "strict_mode": self.strict,
                        "text_length": len(text),
                    },
                    processing_time_ms=processing_time,
                )

                logger.debug(
                    f"Classifier validation completed",
                    extra={
                        "validator": self.name,
                        "thought_id": thought.id,
                        "passed": passed,
                        "classifier": self.classifier.name,
                        "label": classification_result.label,
                        "confidence": classification_result.confidence,
                        "violations": violations,
                        "score": score,
                    },
                )

                return result

            except Exception as e:
                logger.error(
                    f"Classifier validation failed",
                    extra={
                        "validator": self.name,
                        "thought_id": thought.id,
                        "classifier": self.classifier.name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Classifier validation failed: {str(e)}",
                    validator_name=self.name,
                    validation_details={
                        "classifier": self.classifier.name,
                        "text_length": len(text),
                        "error_type": type(e).__name__,
                    },
                    context={
                        "validator": self.name,
                        "classifier": self.classifier.name,
                        "text_length": len(text),
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check classifier configuration",
                        "Verify classifier is properly initialized",
                        "Try with different text",
                    ],
                ) from e


def create_classifier_validator(
    classifier: BaseClassifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    strict: bool = True,
    name: Optional[str] = None,
) -> ClassifierValidator:
    """Create a classifier validator with the specified parameters.

    Args:
        classifier: The classifier to use for validation
        threshold: Minimum confidence threshold for classification
        valid_labels: List of labels that are considered valid
        invalid_labels: List of labels that are considered invalid
        strict: Whether to fail validation on any violation
        name: Custom name for the validator

    Returns:
        Configured ClassifierValidator instance
    """
    return ClassifierValidator(
        classifier=classifier,
        threshold=threshold,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        strict=strict,
        name=name,
    )


def sentiment_validator(
    required_sentiment: Optional[str] = None,
    forbidden_sentiments: Optional[List[str]] = None,
    min_confidence: float = 0.6,
    cached: bool = True,
    name: Optional[str] = None,
) -> ClassifierValidator:
    """Create a validator that checks text sentiment.

    Args:
        required_sentiment: Required sentiment ('positive', 'negative', 'neutral')
        forbidden_sentiments: List of forbidden sentiments
        min_confidence: Minimum confidence for sentiment detection
        cached: Whether to use cached sentiment classifier
        name: Custom name for the validator

    Returns:
        ClassifierValidator configured for sentiment validation
    """
    from sifaka.classifiers.sentiment import create_sentiment_classifier

    classifier = create_sentiment_classifier(cached=cached)

    # Set up valid/invalid labels
    valid_labels = None
    invalid_labels = None

    if required_sentiment:
        valid_labels = [required_sentiment]
    elif forbidden_sentiments:
        invalid_labels = forbidden_sentiments

    return create_classifier_validator(
        classifier=classifier,
        threshold=min_confidence,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        name=name or "sentiment_validation",
    )
