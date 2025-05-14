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
    1. Text extraction (if needed)
    2. Classification
    3. Result evaluation
    4. Rule result generation

    ## Lifecycle
    1. Initialization: Set up with classifier and configuration
    2. Text Extraction: Extract relevant text if needed
    3. Classification: Apply classifier to text
    4. Result Evaluation: Determine if classification meets criteria

    ## Error Handling
    - ConfigurationError: Raised when configuration is invalid
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible

    ## Examples
    ```python
    from sifaka.adapters.classifier import create_classifier_rule
    from sifaka.classifiers.safety import SentimentClassifier

    # Create a rule
    classifier = SentimentClassifier()
    rule = create_classifier_rule(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["positive", "neutral"]
    )

    # Use the rule
    result = rule.validate("This is a great example!") if rule else ""
    ```

    Attributes:
        classifier (Classifier): The classifier to use for validation
        threshold (float): Confidence threshold for accepting classifications
        valid_labels (List[str]): List of labels considered valid
        invalid_labels (Optional[List[str]]): Optional list of labels considered invalid
        extraction_function (Optional[Callable[[str], str]]): Optional function to extract text for classification
        rule_id (str): Unique identifier for the rule
        name (str): Human-readable name of the rule
        description (str): Description of the rule's purpose
        severity (str): Severity level of validation failures
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
        Initialize the classifier rule.

        Args:
            classifier: The classifier to use for validation
            threshold: Confidence threshold for accepting classifications
            valid_labels: List of labels considered valid
            invalid_labels: Optional list of labels considered invalid
            extraction_function: Optional function to extract text for classification
            rule_id: Unique identifier for the rule
            name: Human-readable name of the rule
            description: Description of the rule's purpose
            severity: Severity level of validation failures
            config: Optional rule configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(classifier, Classifier):
            raise TypeError(f"Expected a Classifier, got {type(classifier)}")
        all_labels = []
        if hasattr(classifier.config, "get"):
            all_labels = classifier.config.get("labels", [])
        elif hasattr(classifier.config, "labels") and classifier.config is not None:
            all_labels = classifier.config.labels
        if valid_labels is None and invalid_labels is None:
            if config is not None and hasattr(config, "params") and config.params is not None:
                if "valid_labels" in config.params:
                    valid_labels = config.params["valid_labels"]
                elif "invalid_labels" in config.params:
                    invalid_labels = config.params["invalid_labels"]
                else:
                    raise ValueError("Either valid_labels or invalid_labels must be provided")
            else:
                raise ValueError("Either valid_labels or invalid_labels must be provided")
        if invalid_labels is not None and valid_labels is not None:
            raise ValueError("Only one of valid_labels or invalid_labels can be provided")
        if valid_labels is None and invalid_labels is not None:
            valid_labels = [label for label in all_labels if label not in invalid_labels]
        if (
            threshold == 0.5
            and config is not None
            and hasattr(config, "params")
            and config.params is not None
            and "threshold" in config.params
        ):
            threshold = config.params["threshold"]
        if name is None:
            name = f"{classifier.name} rule"
        if description is None:
            description = f"Validates that text is classified as one of {valid_labels} with confidence >= {threshold}"
        rule_config = config or RuleConfig(name=name, description=description)
        # Create a new config with updated parameters
        rule_config = RuleConfig(
            name=rule_config.name,
            description=rule_config.description,
            params={
                "threshold": threshold,
                "valid_labels": valid_labels,
                "invalid_labels": invalid_labels,
            },
        )
        self._classifier = classifier
        self._rule_id = rule_id or f"classifier_{classifier.name}"
        self._severity = severity
        self._classifier_config = ClassifierRuleConfig(
            threshold=threshold,
            valid_labels=valid_labels if valid_labels is not None else [],
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )
        super().__init__(name=name, description=description, config=rule_config)
        self._validator = self._create_default_validator()

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
        result = self._classifier.classify(text_to_classify)
        label = result.label
        confidence = result.confidence
        is_valid_label = label in self._classifier_config.valid_labels
        passed = is_valid_label and confidence >= self._classifier_config.threshold
        metadata = {
            "label": label,
            "confidence": confidence,
            "threshold": self._classifier_config.threshold,
            "valid_labels": self._classifier_config.valid_labels,
            "classification_result": {
                "label": result.label,
                "confidence": result.confidence,
                "metadata": getattr(result, "metadata", {}),
            },
            "rule_id": self._rule_id,
            "severity": self._severity,
        }
        if passed:
            message = f"Classified as '{label}' with confidence {confidence:.2f}, which is >= threshold {self._classifier_config.threshold} and in valid labels"
        elif not is_valid_label:
            message = f"Classified as '{label}' which is not in valid labels {self._classifier_config.valid_labels}"
            metadata["errors"] = [message]
        else:
            message = f"Classified as '{label}' with confidence {confidence:.2f}, which is < threshold {self._classifier_config.threshold}"
            metadata["errors"] = [message]
        return RuleResult(passed=passed, message=message, metadata=metadata)

    def validate(self, input_text: str, **kwargs: Any) -> RuleResult:
        """
        Validate input text using the classifier.

        This method:
        1. Extracts relevant text if an extraction function is provided
        2. Applies the classifier to the text
        3. Evaluates if the classification meets the criteria
        4. Returns a standardized validation result

        Args:
            input_text: The text to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult containing validation outcome and metadata

        Raises:
            ValidationError: If validation fails due to an error
        """
        from sifaka.utils.text import handle_empty_text

        empty_result = handle_empty_text(
            input_text,
            passed=True,
            metadata={"rule_id": self._rule_id, "severity": self._severity},
            component_type="adapter",
        )
        if empty_result:
            return empty_result
        text_to_classify = input_text
        if self._classifier_config.extraction_function:
            text_to_classify = self._classifier_config.extraction_function(input_text)
        try:
            return self._validate_text(text_to_classify)
        except Exception as e:
            error_message = f"Classification error: {str(e)}"
            # Create a properly typed RuleResult
            from sifaka.core.results import create_rule_result

            return create_rule_result(
                passed=False,
                message=error_message,
                metadata={
                    "error_type": type(e).__name__,
                    "rule_id": self._rule_id,
                    "severity": self._severity,
                    "errors": [error_message],
                },
            )

    def _create_default_validator(self) -> Any:
        """Create a default validator for this rule.

        This method is required by the Rule abstract base class.

        Returns:
            A validator that uses the classifier for validation
        """

        class ClassifierValidator:

            def __init__(self, rule: ClassifierRule) -> None:
                self._rule = rule

            def validate(self, input_text: str, **kwargs: Any) -> RuleResult:
                from sifaka.utils.text import handle_empty_text

                empty_result = handle_empty_text(input_text)
                if empty_result:
                    return empty_result
                # Use create_rule_result to ensure proper typing
                from sifaka.core.results import create_rule_result

                result = self._rule._validate_text(input_text)
                return create_rule_result(
                    passed=result.passed, message=result.message, metadata=result.metadata
                )

        return ClassifierValidator(self)


class ClassifierAdapter(BaseAdapter[str, Classifier]):
    """
    Adapter for using classifiers as validators.

    ## Overview
    This adapter converts classifiers into validators, allowing them to be used
    in Sifaka's validation system.

    ## Architecture
    The adapter follows a standard pattern:
    1. Text extraction (if needed)
    2. Classification
    3. Result evaluation
    4. Rule result generation

    ## Lifecycle
    1. Initialization: Set up with classifier and configuration
    2. Text Extraction: Extract relevant text if needed
    3. Classification: Apply classifier to text
    4. Result Evaluation: Determine if classification meets criteria

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state object
    - Clear separation of configuration and state
    - State components:
      - adaptee: The classifier being adapted
      - config_cache: Configuration storage
      - execution_count: Number of validation executions
      - last_execution_time: Timestamp of last execution
      - avg_execution_time: Average execution time
      - error_count: Number of validation errors
      - cache: Temporary data storage for classification results

    ## Error Handling
    - ConfigurationError: Raised when configuration is invalid
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible
    - AdapterError: Raised for adapter-specific errors

    ## Examples
    ```python
    from sifaka.adapters.classifier import create_classifier_adapter
    from sifaka.classifiers.implementations.content.toxicity import create_toxicity_classifier

    # Create a toxicity classifier
    classifier = create_toxicity_classifier()

    # Create an adapter with basic configuration
    adapter = ClassifierAdapter(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["non_toxic"]
    )

    # Create an adapter with an extraction function
    def extract_message(text: Any) -> None:
        # Extract message content from a structured format
        if ":" in text:
            return (text.split("Content:")[1].strip()
        return text

    adapter = ClassifierAdapter(
        classifier=classifier,
        threshold=0.7,
        valid_labels=["non_toxic"],
        extraction_function=extract_message
    )
    ```

    Attributes:
        classifier (Classifier): The classifier to use for validation
        threshold (float): Confidence threshold for accepting classifications
        valid_labels (List[str]): List of labels considered valid
        invalid_labels (Optional[List[str]]): Optional list of labels considered invalid
        extraction_function (Optional[Callable[[str], str]]): Optional function to extract text for classification
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
        Initialize the classifier adapter.

        This method sets up the adapter with a classifier and configuration options.
        It validates the configuration and initializes the state manager.

        Args:
            classifier: The classifier to adapt
            threshold: Confidence threshold for accepting classifications (0.0 to 1.0)
            valid_labels: List of labels considered valid for validation to pass
            invalid_labels: Optional list of labels considered invalid (mutually exclusive with valid_labels)
            extraction_function: Optional function to extract text for classification

        Raises:
            ConfigurationError: If configuration is invalid
            ValueError: If both valid_labels and invalid_labels are provided
            TypeError: If classifier does not implement the Classifier protocol
            AdapterError: If initialization fails

        Example:
            ```python
            from sifaka.adapters.classifier import ClassifierAdapter
            from sifaka.classifiers.implementations.content.toxicity import create_toxicity_classifier

            # Create a toxicity classifier
            classifier = create_toxicity_classifier()

            # Create an adapter with basic configuration
            adapter = ClassifierAdapter(
                classifier=classifier,
                threshold=0.8,
                valid_labels=["non_toxic"]
            )
            ```
        """
        try:
            super().__init__(classifier)
            # Access state through the state manager
            state: Dict[str, Any] = (
                self._state_manager._state if hasattr(self._state_manager, "_state") else {}
            )
            config = ClassifierRuleConfig(
                threshold=threshold,
                valid_labels=valid_labels if valid_labels is not None else [],
                invalid_labels=invalid_labels,
                extraction_function=extraction_function,
            )
            if "config_cache" not in state:
                state["config_cache"] = {}
            state["config_cache"]["classifier_config"] = config
            self._state_manager.set_metadata("adapter_type", "classifier")
            self._state_manager.set_metadata("classifier_type", classifier.__class__.__name__)
            logger.debug(f"Initialized ClassifierAdapter for {classifier.name}")
        except Exception as e:
            error_info = handle_error(
                e, f"ClassifierAdapter:{getattr(classifier, 'name', 'unknown')}"
            )
            raise AdapterError(
                f"Failed to initialize ClassifierAdapter: {str(e)}", metadata=error_info
            ) from e

    @property
    def config(self) -> ClassifierRuleConfig:
        """
        Get the adapter configuration.

        Returns:
            The ClassifierRuleConfig containing threshold, valid_labels, and other settings

        Example:
            ```python
            adapter = ClassifierAdapter(classifier=my_classifier, threshold=0.8, valid_labels=["positive"])
            config = adapter.config
            print(f"Threshold: {config.threshold}")
            print(f"Valid labels: {config.valid_labels}")
            ```
        """
        # Access state through the state manager
        state: Dict[str, Any] = (
            self._state_manager._state if hasattr(self._state_manager, "_state") else {}
        )
        return (
            state.config_cache["classifier_config"]
            if hasattr(state, "config_cache")
            else ClassifierRuleConfig()
        )

    @property
    def classifier(self) -> Classifier:
        """
        Get the classifier being adapted.

        Returns:
            The underlying classifier instance that this adapter wraps

        Example:
            ```python
            adapter = ClassifierAdapter(classifier=my_classifier, threshold=0.8, valid_labels=["positive"])
            classifier = adapter.classifier
            print(f"Using classifier: {classifier.name}")
            ```
        """
        # Explicitly cast to Classifier to satisfy mypy
        return self.adaptee if isinstance(self.adaptee, Classifier) else self.adaptee

    @property
    def valid_labels(self) -> List[str]:
        """
        Get the valid labels for this adapter.

        Returns:
            List of labels that are considered valid for validation to pass

        Example:
            ```python
            adapter = ClassifierAdapter(classifier=my_classifier, valid_labels=["positive", "neutral"])
            print(f"Valid labels: {adapter.valid_labels}")
            ```
        """
        return self.config.valid_labels

    @property
    def threshold(self) -> float:
        """
        Get the confidence threshold for this adapter.

        Returns:
            The confidence threshold value (between 0.0 and 1.0)

        Example:
            ```python
            adapter = ClassifierAdapter(classifier=my_classifier, threshold=0.8)
            print(f"Confidence threshold: {adapter.threshold}")
            ```
        """
        return self.config.threshold

    def _validate_impl(self, input_value: str, **kwargs: Any) -> RuleResult:
        """
        Implementation of validation logic for the classifier adapter.

        This method is called by the base adapter's validate method.

        Args:
            input_value: The text to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult containing validation outcome and metadata

        Raises:
            ValidationError: If validation fails due to an error
            AdapterError: If adapter-specific error occurs
        """
        # Access state through the state manager
        state: Dict[str, Any] = (
            self._state_manager._state if hasattr(self._state_manager, "_state") else {}
        )
        config = self.config
        text_to_classify = input_value
        if config.extraction_function:
            text_to_classify = config.extraction_function(input_value)
        cache_key = self._get_cache_key(text_to_classify, kwargs)
        if cache_key and "cache" in state and cache_key in state["cache"]:
            cached_result = state["cache"][cache_key]
            logger.debug(f"Cache hit for classifier {self.classifier.name}")
            # Create a properly typed RuleResult
            from sifaka.core.results import create_rule_result

            return create_rule_result(
                passed=cached_result["passed"],
                message=cached_result["message"],
                metadata=cached_result["metadata"],
            )
        result = self.classifier.classify(text_to_classify)
        label = result.label
        confidence = result.confidence
        is_valid_label = label in config.valid_labels
        passed = is_valid_label and confidence >= config.threshold
        metadata = {
            "label": label,
            "confidence": confidence,
            "threshold": config.threshold,
            "valid_labels": config.valid_labels,
            "adaptee_name": self.classifier.name,
            "classification_time": time.time(),
        }
        if passed:
            message = f"Classified as '{label}' with confidence {confidence:.2f}, which is >= threshold {config.threshold} and in valid labels"
        elif not is_valid_label:
            message = f"Classified as '{label}' which is not in valid labels {config.valid_labels}"
        else:
            message = f"Classified as '{label}' with confidence {confidence:.2f}, which is < threshold {config.threshold}"
        # Create a properly typed RuleResult
        from sifaka.core.results import create_rule_result

        result_obj = create_rule_result(passed=passed, message=message, metadata=metadata)
        if cache_key and kwargs.get("cache_result", True):
            if "cache" not in state:
                state["cache"] = {}
            state["cache"][cache_key] = {"passed": passed, "message": message, "metadata": metadata}
        return result_obj

    def _get_cache_key(self, input_value: str, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Generate a cache key for the input value and kwargs.

        Args:
            input_value (str): The input text
            kwargs (Dict[str, Any]): Additional parameters

        Returns:
            Optional[str]: Cache key or None if caching is disabled
        """
        if kwargs.get("cache_result", True):
            threshold = kwargs.get("threshold", self.threshold)
            return f"{hash(input_value)}:{threshold}"
        return None

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about adapter usage.

        Returns:
            Dict[str, Any]: Dictionary with detailed usage statistics
        """
        stats: Dict[str, Any] = self.get_statistics()
        stats.update(
            {
                "threshold": self.threshold,
                "valid_labels": self.valid_labels,
                "classifier_name": self.classifier.name,
                "classifier_type": self._state_manager.get_metadata("classifier_type", "unknown"),
            }
        )
        return stats


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
    Factory function to create a classifier rule.

    ## Overview
    This function simplifies the creation of classifier rules by providing a
    consistent interface.

    Args:
        classifier (Classifier): The classifier to use for validation
        threshold (float): Confidence threshold for accepting classifications
        valid_labels (Optional[List[str]]): List of labels considered valid
        invalid_labels (Optional[List[str]]): Optional list of labels considered invalid
        extraction_function (Optional[Callable[[str], str]]): Optional function to extract text for classification
        rule_id (Optional[str]): Unique identifier for the rule
        name (Optional[str]): Human-readable name of the rule
        description (Optional[str]): Description of the rule's purpose
        severity (str): Severity level of validation failures
        config (Optional[RuleConfig]): Additional rule configuration
        **kwargs: Additional keyword arguments

    Returns:
        ClassifierRule: A configured classifier rule

    Raises:
        ConfigurationError: If configuration is invalid
        ValueError: If threshold is invalid
        TypeError: If label lists are invalid

    ## Examples
    ```python
    from sifaka.adapters.classifier import create_classifier_rule
    from sifaka.classifiers.safety import SentimentClassifier

    # Create a rule
    classifier = SentimentClassifier()
    rule = create_classifier_rule(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["positive", "neutral"]
    )
    ```
    """
    from sifaka.utils import standardize_rule_config

    rule_params = {
        "threshold": threshold,
        "valid_labels": valid_labels,
        "invalid_labels": invalid_labels,
        "extraction_function": extraction_function,
    }
    # Convert RuleConfig to Dict[str, Any] for standardize_rule_config
    config_dict = config.__dict__ if config else None
    rule_config = standardize_rule_config(config=config_dict, params=rule_params, **kwargs)
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
    name: Optional[str] = None,
    description: Optional[str] = None,
    initialize: bool = True,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ClassifierAdapter:
    """
    Factory function to create a classifier adapter.

    ## Overview
    This function simplifies the creation of classifier adapters by providing a
    consistent interface with standardized configuration options.

    ## Architecture
    The factory function follows a standard pattern:
    1. Validate inputs
    2. Extract configuration from various sources
    3. Create adapter instance
    4. Initialize if requested
    5. Return configured instance

    Args:
        classifier (Classifier): The classifier to use for validation
        threshold (float): Confidence threshold for accepting classifications
        valid_labels (Optional[List[str]]): List of labels considered valid
        invalid_labels (Optional[List[str]]): Optional list of labels considered invalid
        extraction_function (Optional[Callable[[str], str]]): Optional function to extract text for classification
        name (Optional[str]): Optional name for the adapter
        description (Optional[str]): Optional description for the adapter
        initialize (bool): Whether to initialize the adapter immediately
        config (Optional[Dict[str, Any]]): Additional adapter configuration
        **kwargs: Additional keyword arguments

    Returns:
        ClassifierAdapter: A configured classifier adapter

    Raises:
        ConfigurationError: If configuration is invalid
        ValueError: If threshold is invalid
        TypeError: If label lists are invalid
        AdapterError: If initialization fails

    ## Examples
    ```python
    from sifaka.adapters.classifier import create_classifier_adapter
    from sifaka.classifiers.safety import SentimentClassifier

    # Basic usage
    classifier = SentimentClassifier()
    adapter = create_classifier_adapter(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["positive", "neutral"]
    )

    # With custom name and description
    adapter = create_classifier_adapter(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["positive", "neutral"],
        name="sentiment_adapter",
        description="Adapter for sentiment classification"
    )

    # Without immediate initialization
    adapter = create_classifier_adapter(
        classifier=classifier,
        threshold=0.8,
        valid_labels=["positive", "neutral"],
        initialize=False
    )
    ```
    """
    try:
        if not isinstance(classifier, Classifier):
            raise ValueError(f"Expected a Classifier, got {type(classifier)}")
        if config:
            threshold = config.get("threshold", threshold)
            valid_labels = config.get("valid_labels", valid_labels)
            invalid_labels = config.get("invalid_labels", invalid_labels)
            extraction_function = config.get("extraction_function", extraction_function)
        threshold = kwargs.get("threshold", threshold)
        valid_labels = kwargs.get("valid_labels", valid_labels)
        invalid_labels = kwargs.get("invalid_labels", invalid_labels)
        extraction_function = kwargs.get("extraction_function", extraction_function)
        adapter = ClassifierAdapter(
            classifier=classifier,
            threshold=threshold,
            valid_labels=valid_labels,
            invalid_labels=invalid_labels,
            extraction_function=extraction_function,
        )
        if name:
            adapter._state_manager.set_metadata("name", name)
        if description:
            adapter._state_manager.set_metadata("description", description)
        if initialize:
            adapter.warm_up()
        logger.debug(f"Created ClassifierAdapter for {classifier.name}")
        return adapter
    except Exception as e:
        if isinstance(e, (ValueError, AdapterError)):
            raise
        error_info = handle_error(
            e, f"ClassifierAdapterFactory:{getattr(classifier, 'name', 'unknown')}"
        )
        raise AdapterError(
            f"Failed to create classifier adapter: {str(e)}", metadata=error_info
        ) from e


ClassifierType = Type[Classifier]
ClassifierInstance = Classifier
