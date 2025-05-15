"""
Classifier Adapter

Adapter for using classifiers as rules in the Sifaka framework.

## Overview
This module provides adapters for using classifiers as validation rules. It enables
the integration of classification models into Sifaka's validation system, allowing
for sophisticated content analysis and validation.

## Components
1. **Classifier Protocol**: Defines the expected interface for classifiers
2. **ClassifierAdapter**: Adapts classifiers to work as validators
3. **ClassifierRule**: Rule that uses a classifier for validation
4. **Factory Functions**: Simple creation patterns for classifier-based rules

## Usage Examples
```python
from sifaka.adapters.classifier import create_classifier_rule
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
result = rule.validate("This is a great example!") if rule else ""
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

## Error Handling
- ConfigurationError: Raised when classifier configuration is invalid
- ValidationError: Raised when validation fails
- TypeError: Raised when input types are incompatible
- AdapterError: Raised for adapter-specific errors

## Configuration
- threshold: Confidence threshold for accepting classifications
- valid_labels: List of labels considered valid
- invalid_labels: Optional list of labels considered invalid
- extraction_function: Optional function to extract text for classification

## State Management
The module uses a standardized state management approach:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state object
- Clear separation of configuration and state
- Execution tracking for monitoring and debugging
"""

import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, TypeVar, runtime_checkable
from pydantic import BaseModel, Field, validate_call
from sifaka.rules.base import Rule, RuleConfig
from sifaka.core.results import RuleResult, ClassificationResult
from sifaka.utils.errors.base import ConfigurationError, ValidationError
from sifaka.adapters.base import BaseAdapter, AdapterError
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
C = TypeVar("C", bound="Classifier")


@runtime_checkable
class Classifier(Protocol):
    """
    Protocol for classifiers.

    ## Overview
    Classes implementing this protocol can classify text
    and provide standardized classification results.

    ## Architecture
    The protocol defines a minimal interface that classifiers must implement
    to be compatible with Sifaka's adapter system.

    ## Lifecycle
    1. Initialization: Set up with model and configuration
    2. Classification: Process input text and apply classification logic
    3. Result: Return standardized classification results

    ## Error Handling
    - ValueError: Raised when input text is invalid
    - RuntimeError: Raised when classification fails
    - TypeError: Raised when input types are incompatible

    ## Examples
    ```python
    from sifaka.classifiers.result import ClassificationResult

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
            if "good" in text.lower() if text else "":
                return ClassificationResult(
                    label="positive",
                    confidence=0.9,
                    metadata={"input_length": len(text))
                )
            else:
                return ClassificationResult(
                    label="neutral",
                    confidence=0.7,
                    metadata={"input_length": len(text))
                )

        def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
            return [self.classify(text) if self else "" for text in texts)
    ```
    """

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text and return a classification result.

        Args:
            text (str): The text to classify

        Returns:
            ClassificationResult: Classification result with label, confidence, and metadata

        Raises:
            ValueError: If input text is invalid
            RuntimeError: If classification fails
        """
        ...

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts and return classification results.

        Args:
            texts (List[str]): List of texts to classify

        Returns:
            List[ClassificationResult]: List of classification results

        Raises:
            ValueError: If any input text is invalid
            RuntimeError: If classification fails
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the classifier name.

        Returns:
            str: The name of the classifier
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the classifier description.

        Returns:
            str: The description of the classifier
        """
        ...

    @property
    def config(self) -> Any:
        """
        Get the classifier configuration.

        Returns:
            Any: The configuration of the classifier
        """
        ...


class ClassifierRuleConfig(BaseModel):
    """
    Configuration for classifier rules.

    ## Overview
    This class provides configuration options for rules that use classifiers
    for validation.

    ## Architecture
    The configuration follows a standard pattern with:
    1. Threshold for confidence scores
    2. Lists of valid and invalid labels
    3. Optional text extraction function

    ## Error Handling
    - ValueError: Raised when threshold is invalid
    - TypeError: Raised when label lists are invalid

    ## Examples
    ```python
    from sifaka.adapters.classifier import ClassifierRuleConfig

    # Create a configuration
    config = ClassifierRuleConfig(
        threshold=0.8,
        valid_labels=["positive", "neutral"],
        extraction_function=lambda text: text.split(":") if text else ""[-1] if ":" in text else text
    )
    ```

    Attributes:
        threshold (float): Confidence threshold for accepting a classification
        valid_labels (List[str]): List of valid labels
        invalid_labels (Optional[List[str]]): Optional list of invalid labels
        extraction_function (Optional[Callable[[str], str]]): Optional function to extract text to classify
    """

    threshold: float = Field(
        0.5, description="Confidence threshold for accepting a classification", ge=0.0, le=1.0
    )
    valid_labels: List[str] = Field(..., description="List of valid labels")
    invalid_labels: Optional[List[str]] = Field(None, description="List of invalid labels")
    extraction_function: Optional[Callable[[str], str]] = Field(
        None, description="Function to extract text to classify from input"
    )


class ClassifierRule(Rule):
    """
    Rule that uses a classifier for validation.

    ## Overview
    This rule adapts a classifier to function as a validation rule, allowing
    for sophisticated content analysis and validation based on classification
    results.

    ## Architecture
    The rule follows a standard pattern:
    1. Configuration with threshold and label lists
    2. Text extraction from input
    3. Classification using the classifier
    4. Validation based on classification results

    ## Error Handling
    - ConfigurationError: Raised when configuration is invalid
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible

    ## Examples
    ```python
    from sifaka.adapters.classifier import ClassifierRule
    from sifaka.classifiers.safety import SentimentClassifier

    # Create a classifier
    classifier = SentimentClassifier()

    # Create a rule
    rule = ClassifierRule(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["positive", "neutral"],
        name="positive_sentiment_rule"
    )

    # Validate text
    result = rule.validate("This is a great example!") if rule else ""
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```

    Attributes:
        classifier (Classifier): The classifier to use for validation
        threshold (float): Confidence threshold for accepting a classification
        valid_labels (List[str]): List of valid labels
        invalid_labels (Optional[List[str]]): Optional list of invalid labels
        extraction_function (Optional[Callable[[str], str]]): Optional function to extract text to classify
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
        Initialize the rule.

        Args:
            classifier: The classifier to use for validation
            threshold: Confidence threshold for accepting a classification
            valid_labels: List of valid labels
            invalid_labels: Optional list of invalid labels
            extraction_function: Optional function to extract text to classify
            rule_id: Optional rule ID
            name: Optional rule name
            description: Optional rule description
            severity: Rule severity level
            config: Optional rule configuration
        """
        super().__init__(
            rule_id=rule_id,
            name=name or classifier.name,
            description=description or classifier.description,
            severity=severity,
            config=config,
        )
        self._classifier = classifier
        self._config = ClassifierRuleConfig(
            threshold=threshold,
            valid_labels=valid_labels or [],
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )

    @property
    def classifier(self) -> Classifier:
        """
        Get the classifier.

        Returns:
            Classifier: The classifier used by this rule
        """
        return self._classifier

    @property
    def threshold(self) -> float:
        """
        Get the confidence threshold.

        Returns:
            float: The confidence threshold for accepting a classification
        """
        return self._config.threshold

    @property
    def valid_labels(self) -> List[str]:
        """
        Get the list of valid labels.

        Returns:
            List[str]: The list of valid labels
        """
        return self._config.valid_labels

    @property
    def rule_id(self) -> str:
        """
        Get the rule ID.

        Returns:
            str: The rule ID
        """
        return self._rule_id

    @property
    def severity(self) -> str:
        """
        Get the rule severity.

        Returns:
            str: The rule severity level
        """
        return self._severity

    def _validate_text(self, text_to_classify: str) -> RuleResult:
        """
        Validate text using the classifier.

        Args:
            text_to_classify: The text to classify

        Returns:
            RuleResult: The validation result

        Raises:
            ValidationError: If validation fails
        """
        try:
            result = self._classifier.classify(text_to_classify)
            if result.confidence < self.threshold:
                return RuleResult(
                    passed=False,
                    message=f"Classification confidence ({result.confidence:.2f}) below threshold ({self.threshold:.2f})",
                    metadata={"confidence": result.confidence, "threshold": self.threshold},
                )
            if result.label in self.valid_labels:
                return RuleResult(
                    passed=True,
                    message=f"Text classified as {result.label} with confidence {result.confidence:.2f}",
                    metadata={"label": result.label, "confidence": result.confidence},
                )
            if self._config.invalid_labels and result.label in self._config.invalid_labels:
                return RuleResult(
                    passed=False,
                    message=f"Text classified as invalid label {result.label}",
                    metadata={"label": result.label, "confidence": result.confidence},
                )
            return RuleResult(
                passed=False,
                message=f"Text classified as {result.label}, which is not in the list of valid labels",
                metadata={"label": result.label, "confidence": result.confidence},
            )
        except Exception as e:
            raise ValidationError(f"Classification failed: {str(e)}") from e

    def validate(self, input_text: str, **kwargs: Any) -> RuleResult:
        """
        Validate input text using the classifier.

        Args:
            input_text: The text to validate
            **kwargs: Additional keyword arguments

        Returns:
            RuleResult: The validation result

        Raises:
            ValidationError: If validation fails
        """
        try:
            if not input_text:
                return RuleResult(
                    passed=False,
                    message="Input text is empty",
                    metadata={"input_length": 0},
                )
            text_to_classify = input_text
            if self._config.extraction_function:
                text_to_classify = self._config.extraction_function(input_text)
            return self._validate_text(text_to_classify)
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e

    def _create_default_validator(self) -> Any:
        """
        Create a default validator for this rule.

        Returns:
            Any: A validator that uses this rule
        """

        class ClassifierValidator:
            def __init__(self, rule: ClassifierRule) -> None:
                self._rule = rule

            def validate(self, input_text: str, **kwargs: Any) -> RuleResult:
                return self._rule.validate(input_text, **kwargs)

        return ClassifierValidator(self)


class ClassifierAdapter(BaseAdapter[str, Classifier]):
    """
    Adapter for using classifiers as validators.

    ## Overview
    This adapter adapts a classifier to function as a validator, allowing
    for sophisticated content analysis and validation based on classification
    results.

    ## Architecture
    The adapter follows a standard pattern:
    1. Configuration with threshold and label lists
    2. Text extraction from input
    3. Classification using the classifier
    4. Validation based on classification results

    ## Error Handling
    - ConfigurationError: Raised when configuration is invalid
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible
    - AdapterError: Raised for adapter-specific errors

    ## Examples
    ```python
    from sifaka.adapters.classifier import ClassifierAdapter
    from sifaka.classifiers.safety import SentimentClassifier

    # Create a classifier
    classifier = SentimentClassifier()

    # Create an adapter
    adapter = ClassifierAdapter(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["positive", "neutral"],
        name="positive_sentiment_adapter"
    )

    # Validate text
    result = adapter.validate("This is a great example!") if adapter else ""
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```

    Attributes:
        classifier (Classifier): The classifier to use for validation
        threshold (float): Confidence threshold for accepting a classification
        valid_labels (List[str]): List of valid labels
        invalid_labels (Optional[List[str]]): Optional list of invalid labels
        extraction_function (Optional[Callable[[str], str]]): Optional function to extract text to classify
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
        Initialize the adapter.

        Args:
            classifier: The classifier to use for validation
            threshold: Confidence threshold for accepting a classification
            valid_labels: List of valid labels
            invalid_labels: Optional list of invalid labels
            extraction_function: Optional function to extract text to classify
        """
        super().__init__()
        self._classifier = classifier
        self._config = ClassifierRuleConfig(
            threshold=threshold,
            valid_labels=valid_labels or [],
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )

    @property
    def config(self) -> ClassifierRuleConfig:
        """
        Get the adapter configuration.

        Returns:
            ClassifierRuleConfig: The adapter configuration
        """
        return self._config

    @property
    def classifier(self) -> Classifier:
        """
        Get the classifier.

        Returns:
            Classifier: The classifier used by this adapter
        """
        return self._classifier

    @property
    def valid_labels(self) -> List[str]:
        """
        Get the list of valid labels.

        Returns:
            List[str]: The list of valid labels
        """
        return self._config.valid_labels

    @property
    def threshold(self) -> float:
        """
        Get the confidence threshold.

        Returns:
            float: The confidence threshold for accepting a classification
        """
        return self._config.threshold

    def _validate_impl(self, input_value: str, **kwargs: Any) -> RuleResult:
        """
        Validate input text using the classifier.

        Args:
            input_value: The text to validate
            **kwargs: Additional keyword arguments

        Returns:
            RuleResult: The validation result

        Raises:
            ValidationError: If validation fails
        """
        try:
            if not input_value:
                return RuleResult(
                    passed=False,
                    message="Input text is empty",
                    metadata={"input_length": 0},
                )
            text_to_classify = input_value
            if self._config.extraction_function:
                text_to_classify = self._config.extraction_function(input_value)
            result = self._classifier.classify(text_to_classify)
            if result.confidence < self.threshold:
                return RuleResult(
                    passed=False,
                    message=f"Classification confidence ({result.confidence:.2f}) below threshold ({self.threshold:.2f})",
                    metadata={"confidence": result.confidence, "threshold": self.threshold},
                )
            if result.label in self.valid_labels:
                return RuleResult(
                    passed=True,
                    message=f"Text classified as {result.label} with confidence {result.confidence:.2f}",
                    metadata={"label": result.label, "confidence": result.confidence},
                )
            if self._config.invalid_labels and result.label in self._config.invalid_labels:
                return RuleResult(
                    passed=False,
                    message=f"Text classified as invalid label {result.label}",
                    metadata={"label": result.label, "confidence": result.confidence},
                )
            return RuleResult(
                passed=False,
                message=f"Text classified as {result.label}, which is not in the list of valid labels",
                metadata={"label": result.label, "confidence": result.confidence},
            )
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e

    def _get_cache_key(self, input_value: str, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Get the cache key for the input value.

        Args:
            input_value: The input value
            kwargs: Additional keyword arguments

        Returns:
            Optional[str]: The cache key, or None if caching is not supported
        """
        if not self._config.extraction_function:
            return input_value
        return None

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the adapter's usage.

        Returns:
            Dict[str, Any]: Dictionary containing detailed statistics
        """
        return {
            "classifier_name": self._classifier.name,
            "classifier_description": self._classifier.description,
            "threshold": self.threshold,
            "valid_labels": self.valid_labels,
            "invalid_labels": self._config.invalid_labels,
            "has_extraction_function": self._config.extraction_function is not None,
        }


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
    Create a classifier rule.

    Args:
        classifier: The classifier to use for validation
        threshold: Confidence threshold for accepting a classification
        valid_labels: List of valid labels
        invalid_labels: Optional list of invalid labels
        extraction_function: Optional function to extract text to classify
        rule_id: Optional rule ID
        name: Optional rule name
        description: Optional rule description
        severity: Rule severity level
        config: Optional rule configuration
        **kwargs: Additional keyword arguments

    Returns:
        ClassifierRule: The created rule

    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
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
            config=config,
        )
    except Exception as e:
        raise ConfigurationError(f"Failed to create classifier rule: {str(e)}") from e


def create_classifier_adapter(
    classifier: Classifier,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    invalid_labels: Optional[List[str]] = None,
    extraction_function: Optional[Callable[[str], str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    initialize: bool = True,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ClassifierAdapter:
    """
    Create a classifier adapter.

    Args:
        classifier: The classifier to use for validation
        threshold: Confidence threshold for accepting a classification
        valid_labels: List of valid labels
        invalid_labels: Optional list of invalid labels
        extraction_function: Optional function to extract text to classify
        name: Optional adapter name
        description: Optional adapter description
        initialize: Whether to initialize the adapter
        config: Optional adapter configuration
        **kwargs: Additional keyword arguments

    Returns:
        ClassifierAdapter: The created adapter

    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        adapter = ClassifierAdapter(
            classifier=classifier,
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )
        if initialize:
            adapter.initialize()
        return adapter
    except Exception as e:
        raise ConfigurationError(f"Failed to create classifier adapter: {str(e)}") from e


ClassifierType = Type[Classifier]
ClassifierInstance = Classifier
