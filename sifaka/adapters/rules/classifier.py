"""
Adapter for using classifiers as rules.

This module provides adapters for using classifiers as validation rules.

## Architecture Overview

The classifier adapter follows an adapter pattern to convert classifiers into validation rules:

1. **Classifier Protocol**: Defines the expected interface for classifiers
2. **ClassifierAdapter**: Adapts classifiers to work as validators
3. **ClassifierRule**: Rule that uses a classifier for validation
4. **Factory Functions**: Simple creation patterns for classifier-based rules

## Component Lifecycle

### ClassifierAdapter
1. **Initialization**: Set up with classifier and configuration
2. **Validation**: Run classifier on input text
3. **Result Conversion**: Convert classification to standardized rule result

### ClassifierRule
1. **Initialization**: Set up with classifier and configuration
2. **Text Extraction**: Extract relevant text from input if needed
3. **Classification**: Apply classifier to extracted text
4. **Result Evaluation**: Determine if classification meets criteria

## Usage Examples

```python
from sifaka.adapters.rules import create_classifier_rule
from sifaka.classifiers.safety import SentimentClassifier

# Create a classifier
classifier = SentimentClassifier()

# Create a rule from the classifier
rule = create_classifier_rule(
    classifier=classifier,
    threshold=0.8,
    valid_labels=["positive", "neutral"],
    name="positive_sentiment_rule"
)

# Validate text
result = rule.validate("This is a great example!")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    TypeVar,
    Generic,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, Field, validate_arguments, validate_call

from sifaka.rules.base import Rule, RuleConfig, RuleResult, BaseValidator
from sifaka.adapters.rules.base import Adaptable, BaseAdapter, A
from sifaka.classifiers.base import ClassificationResult, ClassifierConfig

# Type for classifier
C = TypeVar("C", bound="Classifier")


@runtime_checkable
class Classifier(Protocol):
    """
    Protocol for classifiers.

    Classes implementing this protocol can classify text
    and provide standardized classification results.

    Lifecycle:
    1. Initialization: Set up with model and configuration
    2. Classification: Process input text and apply classification logic
    3. Result: Return standardized classification results

    Examples:
        ```python
        from sifaka.classifiers.base import ClassificationResult

        class SentimentClassifier:
            @property
            def name(self) -> str:
                return "sentiment_classifier"

            @property
            def description(self) -> str:
                return "Classifies text sentiment"

            @property
            def config(self) -> Any:
                return self._config

            def classify(self, text: str) -> ClassificationResult:
                # Apply classification logic
                if "good" in text.lower():
                    return ClassificationResult(
                        label="positive",
                        confidence=0.9,
                        metadata={"input_length": len(text)}
                    )
                else:
                    return ClassificationResult(
                        label="neutral",
                        confidence=0.7,
                        metadata={"input_length": len(text)}
                    )

            def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
                return [self.classify(text) for text in texts]
        ```
    """

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text and return a classification result.

        Args:
            text: The text to classify

        Returns:
            Classification result with label, confidence, and metadata
        """
        ...

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts and return classification results.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the classifier name.

        Returns:
            The name of the classifier
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the classifier description.

        Returns:
            The description of the classifier
        """
        ...

    @property
    def config(self) -> Any:
        """
        Get the classifier configuration.

        Returns:
            The configuration of the classifier
        """
        ...


class ClassifierRuleConfig(BaseModel):
    """
    Configuration for classifier rules.

    This class provides configuration options for rules that use classifiers
    for validation.

    Attributes:
        threshold: Confidence threshold for accepting a classification
        valid_labels: List of valid labels
        invalid_labels: Optional list of invalid labels
        extraction_function: Optional function to extract text to classify

    Examples:
        ```python
        from sifaka.adapters.rules.classifier import ClassifierRuleConfig

        # Create a configuration
        config = ClassifierRuleConfig(
            threshold=0.8,
            valid_labels=["positive", "neutral"],
            extraction_function=lambda text: text.split(":")[-1] if ":" in text else text
        )
        ```
    """

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
    """
    Rule that uses a classifier to validate input.

    This rule delegates validation to a classifier, converting its classification
    results into rule validation results.

    Lifecycle:
    1. Initialization: Set up with classifier and configuration
    2. Validation: Extract relevant text, classify it, and evaluate the result
    3. Result: Return validation result with comprehensive metadata

    Examples:
        ```python
        from sifaka.adapters.rules.classifier import ClassifierRule
        from sifaka.classifiers.safety import ToxicityClassifier

        # Create a classifier
        classifier = ToxicityClassifier()

        # Create a rule
        rule = ClassifierRule(
            classifier=classifier,
            threshold=0.7,
            valid_labels=["non_toxic"],
            name="toxicity_rule",
            description="Ensures text is not toxic"
        )

        # Validate text
        result = rule.validate("Hello world!")
        ```
    """

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
        config: Optional[RuleConfig] = None,
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
            config: Optional RuleConfig object

        Raises:
            TypeError: If classifier does not implement the Classifier protocol
            ValueError: If neither valid_labels nor invalid_labels is provided
            ValueError: If both valid_labels and invalid_labels are provided
        """
        if not isinstance(classifier, Classifier):
            raise TypeError(f"Expected a Classifier, got {type(classifier)}")

        # Get labels from the classifier's config
        all_labels = []
        if hasattr(classifier.config, "get"):
            all_labels = classifier.config.get("labels", [])
        elif hasattr(classifier.config, "labels"):
            all_labels = classifier.config.labels

        # Check for valid_labels or invalid_labels in config params if not provided directly
        if valid_labels is None and invalid_labels is None:
            if config and "valid_labels" in config.params:
                valid_labels = config.params["valid_labels"]
            elif config and "invalid_labels" in config.params:
                invalid_labels = config.params["invalid_labels"]
            else:
                raise ValueError("Either valid_labels or invalid_labels must be provided")

        if invalid_labels is not None and valid_labels is not None:
            raise ValueError("Only one of valid_labels or invalid_labels can be provided")

        # Derive valid labels from invalid labels if needed
        if valid_labels is None and invalid_labels is not None:
            valid_labels = [label for label in all_labels if label not in invalid_labels]

        # Get threshold from config if not provided directly
        if threshold == 0.5 and config and "threshold" in config.params:
            threshold = config.params["threshold"]

        # Set name and description based on classifier if not provided
        if name is None:
            name = f"{classifier.name} rule"
        if description is None:
            description = (
                f"Validates that text is classified as one of {valid_labels} "
                f"with confidence >= {threshold}"
            )

        # Use provided config or create a new one
        rule_config = config or RuleConfig()

        # Ensure params contains the necessary configuration
        rule_config = rule_config.with_params(
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
        )

        # Store essential attributes
        self._classifier = classifier
        self._rule_id = rule_id or f"classifier_{classifier.name}"
        self._severity = severity
        self._classifier_config = ClassifierRuleConfig(
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=rule_config,
        )

    @property
    def classifier(self) -> Classifier:
        """Get the classifier used by this rule."""
        return self._classifier

    @property
    def threshold(self) -> float:
        """Get the confidence threshold for this rule."""
        return self._classifier_config.threshold

    @property
    def valid_labels(self) -> List[str]:
        """Get the valid labels for this rule."""
        return self._classifier_config.valid_labels

    @property
    def rule_id(self) -> str:
        """
        Get the rule ID.

        Returns:
            The ID of the rule
        """
        return self._rule_id

    @property
    def severity(self) -> str:
        """
        Get the rule severity.

        Returns:
            The severity of the rule
        """
        return self._severity

    def _validate_text(self, text_to_classify: str) -> RuleResult:
        """
        Internal method to validate text using the classifier.

        Args:
            text_to_classify: Text to classify

        Returns:
            RuleResult with validation results

        Raises:
            Exception: Any exception from the classifier is caught by the caller
        """
        # Get prediction using the classifier's classify method
        result = self._classifier.classify(text_to_classify)

        # Extract label and confidence
        label = result.label
        confidence = result.confidence

        # Check if label is valid
        is_valid_label = label in self._classifier_config.valid_labels
        passed = is_valid_label and confidence >= self._classifier_config.threshold

        # Prepare metadata
        metadata = {
            "label": label,
            "confidence": confidence,
            "threshold": self._classifier_config.threshold,
            "valid_labels": self._classifier_config.valid_labels,
            "classification_result": (
                result.model_dump() if hasattr(result, "model_dump") else result
            ),
            "rule_id": self._rule_id,
            "severity": self._severity,
        }

        # Create result message
        if passed:
            message = (
                f"Classified as '{label}' with confidence {confidence:.2f}, "
                f"which is >= threshold {self._classifier_config.threshold} and in valid labels"
            )
        else:
            if not is_valid_label:
                message = f"Classified as '{label}' which is not in valid labels {self._classifier_config.valid_labels}"
                metadata["errors"] = [message]
            else:
                message = (
                    f"Classified as '{label}' with confidence {confidence:.2f}, "
                    f"which is < threshold {self._classifier_config.threshold}"
                )
                metadata["errors"] = [message]

        return RuleResult(
            passed=passed,
            message=message,
            metadata=metadata,
        )

    def validate(self, input_text: str, **kwargs) -> RuleResult:
        """
        Validate input using the classifier.

        This method extracts text to classify if needed, classifies it,
        and evaluates the classification against the rule criteria.

        Args:
            input_text: Input to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        # Handle empty input
        if not input_text:
            return RuleResult(
                passed=True,
                message="Empty text validation skipped",
                metadata={
                    "reason": "empty_input",
                    "rule_id": self._rule_id,
                    "severity": self._severity,
                },
            )

        # Extract text to classify
        text_to_classify = input_text
        if self._classifier_config.extraction_function:
            text_to_classify = self._classifier_config.extraction_function(input_text)

        try:
            return self._validate_text(text_to_classify)
        except Exception as e:
            # Handle classifier errors
            error_message = f"Classification error: {str(e)}"
            return RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error_type": type(e).__name__,
                    "rule_id": self._rule_id,
                    "severity": self._severity,
                    "errors": [error_message],
                },
            )

    def _create_default_validator(self) -> BaseValidator[str]:
        """Create a default validator for this rule.

        This method is required by the Rule abstract base class.

        Returns:
            A validator that uses the classifier for validation
        """

        # Create a simple validator that uses the classifier
        class ClassifierValidator(BaseValidator[str]):
            def __init__(self, rule: ClassifierRule):
                self._rule = rule

            def validate(self, input_text: str, **kwargs) -> RuleResult:
                # Handle empty text
                empty_result = self.handle_empty_text(input_text)
                if empty_result:
                    return empty_result

                return self._rule._validate_text(input_text)

        return ClassifierValidator(self)


class ClassifierAdapter(BaseAdapter[str, Classifier]):
    """
    Adapter for using a classifier as a validator.

    This adapter converts a classifier into a validator that can be used
    in the Sifaka rule system.

    Type Parameters:
        Classifier: The classifier type being adapted

    Lifecycle:
    1. Initialization: Set up with classifier and configuration
    2. Validation: Delegate to internal ClassifierRule
    3. Result: Return validation result

    Examples:
        ```python
        from sifaka.adapters.rules.classifier import ClassifierAdapter
        from sifaka.classifiers.safety import SentimentClassifier

        # Create a classifier
        classifier = SentimentClassifier()

        # Create an adapter
        adapter = ClassifierAdapter(
            classifier=classifier,
            threshold=0.8,
            valid_labels=["positive", "neutral"]
        )

        # Use the adapter for validation
        result = adapter.validate("This is a great example!")
        ```
    """

    def __init__(
        self,
        classifier: Classifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        invalid_labels: Optional[List[str]] = None,
        extraction_function: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Initialize with classifier and configuration.

        Args:
            classifier: The classifier to adapt
            threshold: Confidence threshold for accepting a classification
            valid_labels: List of valid labels
            invalid_labels: List of invalid labels
            extraction_function: Function to extract text to classify

        Raises:
            ValueError: If neither valid_labels nor invalid_labels is provided
            ValueError: If both valid_labels and invalid_labels are provided
        """
        super().__init__(classifier)

        # Create configuration
        self._config = ClassifierRuleConfig(
            threshold=threshold,
            valid_labels=valid_labels if valid_labels is not None else [],
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )

    @property
    def config(self) -> ClassifierRuleConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def classifier(self) -> Classifier:
        """Get the classifier being adapted."""
        return self.adaptee

    @property
    def valid_labels(self) -> List[str]:
        """Get the valid labels for this adapter."""
        return self._config.valid_labels

    @property
    def threshold(self) -> float:
        """Get the confidence threshold for this adapter."""
        return self._config.threshold

    def validate(self, input_text: str, **kwargs) -> RuleResult:
        """
        Validate input using the classifier.

        Args:
            input_text: Text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        # Handle empty text
        empty_result = self.handle_empty_text(input_text)
        if empty_result:
            return empty_result

        try:
            # Extract text to classify if needed
            text_to_classify = input_text
            if self._config.extraction_function:
                text_to_classify = self._config.extraction_function(input_text)

            # Get classification result
            result = self.classifier.classify(text_to_classify)

            # Extract label and confidence
            label = result.label
            confidence = result.confidence

            # Check if label is valid
            is_valid_label = label in self._config.valid_labels
            passed = is_valid_label and confidence >= self._config.threshold

            # Prepare metadata
            metadata = {
                "label": label,
                "confidence": confidence,
                "threshold": self._config.threshold,
                "valid_labels": self._config.valid_labels,
                "adaptee_name": self.classifier.name,
            }

            # Create result message
            if passed:
                message = (
                    f"Classified as '{label}' with confidence {confidence:.2f}, "
                    f"which is >= threshold {self._config.threshold} and in valid labels"
                )
            else:
                if not is_valid_label:
                    message = f"Classified as '{label}' which is not in valid labels {self._config.valid_labels}"
                else:
                    message = (
                        f"Classified as '{label}' with confidence {confidence:.2f}, "
                        f"which is < threshold {self._config.threshold}"
                    )

            return RuleResult(
                passed=passed,
                message=message,
                metadata=metadata,
            )
        except Exception as e:
            # Handle errors consistently
            return RuleResult(
                passed=False,
                message=f"Classification error: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "adaptee_name": self.classifier.name,
                },
            )


@validate_call(config=dict(arbitrary_types_allowed=True))
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
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> ClassifierRule:
    """
    Create a rule from a classifier.

    This factory function creates a ClassifierRule with the specified
    configuration options. It follows the standard Sifaka configuration pattern
    by using RuleConfig with params for all configuration options.

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
        config: Optional RuleConfig object
        **kwargs: Additional configuration options

    Returns:
        A rule that uses the classifier for validation

    Examples:
        ```python
        from sifaka.adapters.rules import create_classifier_rule
        from sifaka.classifiers.content import ToxicityClassifier
        from sifaka.rules.base import RuleConfig

        # Create a classifier
        classifier = ToxicityClassifier()

        # Create a rule from the classifier with direct parameters
        rule = create_classifier_rule(
            classifier=classifier,
            threshold=0.7,
            valid_labels=["safe"],
            name="safety_rule",
            description="Ensures text is safe and non-toxic"
        )

        # Create a rule with a RuleConfig
        rule_config = RuleConfig(
            priority="HIGH",
            cost=5,
            params={
                "threshold": 0.8,
                "valid_labels": ["safe"],
            }
        )
        rule = create_classifier_rule(
            classifier=classifier,
            config=rule_config,
            name="high_priority_safety_rule",
            description="High priority safety validation"
        )
        ```
    """
    # Import here to avoid circular imports
    from sifaka.utils import standardize_rule_config

    # Create standardized rule configuration
    rule_params = {
        "threshold": threshold,
        "valid_labels": valid_labels,
        "invalid_labels": invalid_labels,
        "extraction_function": extraction_function,
    }

    # Standardize the configuration
    rule_config = standardize_rule_config(config=config, params=rule_params, **kwargs)

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
        config=rule_config,
    )


def create_classifier_adapter(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ClassifierAdapter:
    """
    Create an adapter from a classifier.

    This factory function creates a ClassifierAdapter with the specified
    configuration options. It follows the standard Sifaka configuration pattern
    by using a consistent approach to configuration.

    Args:
        classifier: Classifier to use for validation
        threshold: Confidence threshold for accepting a classification
        valid_labels: List of valid labels
        invalid_labels: List of invalid labels
        extraction_function: Function to extract text to classify from input
        config: Optional configuration dictionary
        **kwargs: Additional configuration options

    Returns:
        An adapter that uses the classifier for validation

    Examples:
        ```python
        from sifaka.adapters.rules import create_classifier_adapter
        from sifaka.classifiers.content import SentimentClassifier

        # Create a classifier
        classifier = SentimentClassifier()

        # Create an adapter from the classifier with direct parameters
        adapter = create_classifier_adapter(
            classifier=classifier,
            threshold=0.8,
            valid_labels=["positive", "neutral"]
        )

        # Create an adapter with a configuration dictionary
        adapter = create_classifier_adapter(
            classifier=classifier,
            config={
                "threshold": 0.8,
                "valid_labels": ["positive", "neutral"]
            }
        )
        ```
    """
    # First check if the classifier is valid
    if not isinstance(classifier, Classifier):
        raise ValueError(f"Expected a Classifier, got {type(classifier)}")

    # Extract configuration from config dictionary if provided
    if config:
        threshold = config.get("threshold", threshold)
        valid_labels = config.get("valid_labels", valid_labels)
        invalid_labels = config.get("invalid_labels", invalid_labels)
        extraction_function = config.get("extraction_function", extraction_function)

    # Extract configuration from kwargs if provided
    threshold = kwargs.get("threshold", threshold)
    valid_labels = kwargs.get("valid_labels", valid_labels)
    invalid_labels = kwargs.get("invalid_labels", invalid_labels)
    extraction_function = kwargs.get("extraction_function", extraction_function)

    return ClassifierAdapter(
        classifier=classifier,
        threshold=threshold,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
        extraction_function=extraction_function,
    )


# Type aliases for better documentation
ClassifierType = Type[Classifier]
ClassifierInstance = Classifier


# Export public classes and functions
__all__ = [
    # Protocols
    "Classifier",
    # Configuration
    "ClassifierRuleConfig",
    # Core components
    "ClassifierRule",
    "ClassifierAdapter",
    # Factory functions
    "create_classifier_rule",
    "create_classifier_adapter",
    # Type aliases
    "ClassifierType",
    "ClassifierInstance",
]
