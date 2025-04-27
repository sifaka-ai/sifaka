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
        default=0.5, description="Confidence threshold for validation", ge=0.0, le=1.0
    )
    valid_labels: List[str] = Field(
        default_factory=list, description="List of labels considered valid"
    )

    def __init__(
        self,
        name: str,
        description: str,
        classifier: Classifier,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a classifier-based rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            classifier: The classifier to use
            config: Configuration dictionary containing:
                   - validation_fn: Optional function to convert classification to validation
                   - threshold: Confidence threshold (default: 0.5)
                   - valid_labels: List of labels considered valid
            **kwargs: Additional arguments
        """
        if not isinstance(classifier, Classifier):
            raise ValueError("classifier must be an instance of Classifier")

        # Extract configuration
        config = config or {}

        # Set validation function
        validation_fn = config.get("validation_fn")
        if validation_fn is not None:
            if not callable(validation_fn):
                raise ValueError("validation_fn must be callable")
            # Check function signature
            import inspect

            sig = inspect.signature(validation_fn)
            if len(sig.parameters) != 1:
                raise ValueError("validation_fn must take exactly one argument")
            param = list(sig.parameters.values())[0]
            if param.annotation != ClassificationResult:
                raise ValueError("validation_fn argument must be annotated as ClassificationResult")
            if sig.return_annotation not in (bool, RuleResult):
                raise ValueError("validation_fn must return bool or RuleResult")

        # Set threshold with validation
        threshold = config.get("threshold", 0.5)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a number between 0 and 1")

        # Set valid labels
        valid_labels = config.get("valid_labels", [])
        if not isinstance(valid_labels, list):
            raise ValueError("valid_labels must be a list")

        # Inherit classifier's cost if not specified
        if "cost" not in kwargs and hasattr(classifier, "cost"):
            kwargs["cost"] = classifier.cost

        # Initialize base rule with all fields
        super().__init__(
            name=name,
            description=description,
            config=config,
            classifier=classifier,
            validation_fn=validation_fn,
            threshold=threshold,
            valid_labels=valid_labels,
            **kwargs,
        )

        # Warm up the classifier
        self.classifier.warm_up()

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate output using the classifier.

        Args:
            output: The text to validate

        Returns:
            RuleResult with classification validation results
        """
        try:
            if not output:
                raise ValueError("Text cannot be empty")

            # Get classification result
            result = self.classifier.classify(output)

            # Use custom validation function if provided
            if self.validation_fn is not None:
                validation_result = self.validation_fn(result)
                if isinstance(validation_result, bool):
                    return RuleResult(
                        passed=validation_result,
                        message=("Validation passed" if validation_result else "Validation failed"),
                        metadata={
                            "classification": result.model_dump(),
                            "threshold": self.threshold,
                            "valid_labels": self.valid_labels,
                            "custom": True,
                        },
                    )
                elif isinstance(validation_result, RuleResult):
                    return validation_result
                else:
                    raise ValueError("validation_fn must return bool or RuleResult")

            # Default validation logic
            passed = result.confidence >= self.threshold and (
                not self.valid_labels or result.label in self.valid_labels
            )

            return RuleResult(
                passed=passed,
                message=(
                    f"Classification '{result.label}' with confidence {result.confidence:.2f}"
                    if passed
                    else f"Failed validation: {result.label} ({result.confidence:.2f})"
                ),
                metadata={
                    "classification": result.model_dump(),
                    "threshold": self.threshold,
                    "valid_labels": self.valid_labels,
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during classification: {str(e)}",
                metadata={"error": str(e)},
            )
