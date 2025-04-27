"""
Rule implementation that uses pluggable classifiers.
"""

from typing import Dict, Any, Optional, Union, List, Callable
from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import Rule, RuleResult
from sifaka.classifiers.base import Classifier, ClassificationResult


class ClassifierRule(Rule):
    """
    A rule that uses a classifier to validate output.

    This allows for flexible validation using any classifier implementation,
    from lightweight ML models to LLMs.

    Attributes:
        classifier: The classifier to use
        validation_fn: Optional function to convert classification to validation result
        threshold: Confidence threshold for validation (0-1)
        valid_labels: List of labels considered valid
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classifier: Classifier = Field(description="The classifier to use for validation")
    validation_fn: Optional[Callable[[ClassificationResult], RuleResult]] = Field(
        default=None, description="Optional function to convert classification to validation result"
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold for validation (0-1)"
    )
    valid_labels: List[Union[str, int, float, bool]] = Field(
        default_factory=list, description="List of labels considered valid"
    )

    def __init__(
        self,
        name: str,
        description: str,
        classifier: Classifier,
        config: Optional[Dict[str, Any]] = None,
        validation_fn: Optional[Callable[[ClassificationResult], RuleResult]] = None,
        threshold: float = 0.5,
        valid_labels: Optional[List[Union[str, int, float, bool]]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a classifier rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            classifier: The classifier to use
            config: Configuration for the rule
            validation_fn: Optional function to convert classification to validation result
            threshold: Confidence threshold for validation
            valid_labels: List of labels considered valid
            **kwargs: Additional arguments
        """
        # Inherit classifier's cost if not specified
        if "cost" not in kwargs and hasattr(classifier, "cost"):
            kwargs["cost"] = classifier.cost

        # Initialize parent class with all fields
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            **kwargs,
        )

        # Set classifier-specific fields
        self.classifier = classifier
        self.validation_fn = validation_fn
        self.threshold = threshold
        self.valid_labels = valid_labels or []

        # Warm up the classifier if needed
        self.classifier.warm_up()

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate output using the classifier.

        Args:
            output: The output to validate

        Returns:
            RuleResult with validation results

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        # Get classification
        result = self.classifier.classify(output)

        # Use custom validation function if provided
        if self.validation_fn is not None:
            validation_result = self.validation_fn(result)
            if not isinstance(validation_result, RuleResult):
                raise ValueError("Validation function must return a RuleResult")
            return validation_result

        # Default validation logic
        passed = result.confidence >= self.threshold and (
            not self.valid_labels or result.label in self.valid_labels
        )

        confidence_msg = f"confidence: {result.confidence:.2f}"
        label_msg = f"label: {result.label}"
        threshold_msg = f"threshold: {self.threshold}"
        valid_labels_msg = f"valid labels: {self.valid_labels}" if self.valid_labels else ""

        message = (
            f"Validation {'passed' if passed else 'failed'} " f"({confidence_msg}, {label_msg}"
        )
        if not passed:
            message += f", {threshold_msg}"
            if valid_labels_msg:
                message += f", {valid_labels_msg}"
        message += ")"

        return RuleResult(
            passed=passed,
            message=message,
            metadata={
                "confidence": result.confidence,
                "label": result.label,
                "classification_result": {
                    "confidence": result.confidence,
                    "label": result.label,
                    **(result.metadata or {}),
                },
            },
        )
