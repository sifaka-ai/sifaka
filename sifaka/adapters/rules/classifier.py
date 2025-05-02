"""
Adapter for using classifiers as rules.

This module provides adapters for using classifiers as validation rules.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Type, Union, runtime_checkable

from pydantic import BaseModel, Field, validate_arguments

from sifaka.rules.base import Rule, RuleResult
from sifaka.adapters.rules.base import Adaptable, BaseAdapter
from sifaka.classifiers.base import ClassificationResult


@runtime_checkable
class Classifier(Protocol):
    """Protocol for classifiers."""

    def classify(self, text: str) -> ClassificationResult: ...
    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def config(self) -> Any: ...


class ClassifierRuleConfig(BaseModel):
    """Configuration for classifier rules."""

    threshold: float = Field(
        0.5,
        description="Confidence threshold for accepting a classification",
        ge=0.0,
        le=1.0,
    )
    valid_labels: List[str] = Field(
        ...,
        description="List of valid labels",
    )
    invalid_labels: Optional[List[str]] = Field(
        None,
        description="List of invalid labels",
    )
    extraction_function: Optional[Callable[[str], str]] = Field(
        None,
        description="Function to extract text to classify from input",
    )


class ClassifierRule(Rule):
    """Rule that uses a classifier to validate input."""

    def __init__(
        self,
        classifier: Classifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        invalid_labels: Optional[List[str]] = None,
        extraction_function: Optional[Callable[[str], str]] = None,
        rule_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        severity: str = "error",
    ) -> None:
        """
        Initialize a classifier rule.

        Args:
            classifier: Classifier to use for validation
            threshold: Confidence threshold for accepting a classification
            valid_labels: List of valid labels
            invalid_labels: List of invalid labels
            extraction_function: Function to extract text to classify from input
            rule_id: Unique identifier for the rule
            name: Name of the rule
            description: Description of the rule
            severity: Severity of the rule
        """
        if not isinstance(classifier, Classifier):
            raise TypeError(f"Expected a Classifier, got {type(classifier)}")

        # Get labels from the classifier's config
        all_labels = getattr(classifier.config, "labels", [])

        if valid_labels is None and invalid_labels is None:
            raise ValueError("Either valid_labels or invalid_labels must be provided")

        if invalid_labels is not None and valid_labels is not None:
            raise ValueError("Only one of valid_labels or invalid_labels can be provided")

        # Derive valid labels from invalid labels if needed
        if valid_labels is None and invalid_labels is not None:
            valid_labels = [label for label in all_labels if label not in invalid_labels]

        # Set name and description based on classifier if not provided
        if name is None:
            name = f"{classifier.name} rule"
        if description is None:
            description = (
                f"Validates that text is classified as one of {valid_labels} "
                f"with confidence >= {threshold}"
            )

        super().__init__(
            rule_id=rule_id or f"classifier_{classifier.name}",
            name=name,
            description=description,
            severity=severity,
        )

        self.classifier = classifier
        self.config = ClassifierRuleConfig(
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )

    def validate(self, input_text: str, **kwargs) -> RuleResult:
        """
        Validate input using the classifier.

        Args:
            input_text: Input to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        # Extract text to classify
        text_to_classify = input_text
        if self.config.extraction_function:
            text_to_classify = self.config.extraction_function(input_text)

        # Get prediction using the classifier's classify method
        result = self.classifier.classify(text_to_classify)

        # Extract label and confidence
        label = result.label
        confidence = result.confidence

        # Check if label is valid
        is_valid_label = label in self.config.valid_labels
        passed = is_valid_label and confidence >= self.config.threshold

        # Prepare metadata
        metadata = {
            "label": label,
            "confidence": confidence,
            "threshold": self.config.threshold,
            "valid_labels": self.config.valid_labels,
            "classification_result": result,
        }

        # Create result message
        if passed:
            message = (
                f"Classified as '{label}' with confidence {confidence:.2f}, "
                f"which is >= threshold {self.config.threshold} and in valid labels"
            )
        else:
            if not is_valid_label:
                message = (
                    f"Classified as '{label}' which is not in valid labels {self.config.valid_labels}"
                )
            else:
                message = (
                    f"Classified as '{label}' with confidence {confidence:.2f}, "
                    f"which is < threshold {self.config.threshold}"
                )

        return RuleResult(
            passed=passed,
            rule_id=self.rule_id,
            message=message,
            severity=self.severity,
            metadata=metadata,
        )


class ClassifierAdapter(BaseAdapter):
    """Adapter for using a classifier as a rule."""

    def __init__(
        self,
        classifier: Classifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        invalid_labels: Optional[List[str]] = None,
        extraction_function: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Initialize a classifier adapter.

        Args:
            classifier: Classifier to use for validation
            threshold: Confidence threshold for accepting a classification
            valid_labels: List of valid labels
            invalid_labels: List of invalid labels
            extraction_function: Function to extract text to classify from input
        """
        super().__init__(classifier)
        self.rule = ClassifierRule(
            classifier=classifier,
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )

    def validate(self, input_text: str, **kwargs) -> RuleResult:
        """
        Validate input using the classifier.

        Args:
            input_text: Input to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        return self.rule.validate(input_text, **kwargs)


@validate_arguments
def create_classifier_rule(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    severity: str = "error",
) -> ClassifierRule:
    """
    Create a rule from a classifier.

    Args:
        classifier: Classifier to use for validation
        threshold: Confidence threshold for accepting a classification
        valid_labels: List of valid labels
        invalid_labels: List of invalid labels
        extraction_function: Function to extract text to classify from input
        rule_id: Unique identifier for the rule
        name: Name of the rule
        description: Description of the rule
        severity: Severity of the rule

    Returns:
        A rule that uses the classifier for validation
    """
    return ClassifierRule(
        classifier=classifier,
        threshold=threshold,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        extraction_function=extraction_function,
        rule_id=rule_id,
        name=name,
        description=description,
        severity=severity,
    )


# Type aliases for better documentation
ClassifierType = Type[Classifier]
ClassifierInstance = Classifier
