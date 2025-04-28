"""
Rule implementation that uses pluggable classifiers.

.. deprecated:: 1.0.0
   This module is deprecated and will be removed in version 2.0.0.
   Use :mod:`sifaka.rules.adapters.classifier` instead.

Migration guide:
1. Replace imports:
   - Old: from sifaka.rules.classifier_rule import ClassifierRule, ClassifierProtocol
   - New: from sifaka.rules.adapters import ClassifierRule, create_classifier_rule

2. Update usage:
   - The new ClassifierRule has better configuration options and error handling
   - Use create_classifier_rule factory function for easier rule creation
"""

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import TypeGuard, List

from sifaka.classifiers.base import (
    ClassificationResult,
    ClassifierProtocol,
)
from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    ValidationError,
)


# Emit deprecation warning
warnings.warn(
    "The sifaka.rules.classifier_rule module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.adapters.classifier instead.",
    DeprecationWarning,
    stacklevel=2,
)


T = TypeVar("T", bound=ClassificationResult)


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Protocol for classifier components."""

    def classify(self, text: str) -> ClassificationResult: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...


# We don't need a separate ClassifierRuleConfig class anymore
# Instead, we'll use RuleConfig with params for consistency


class DefaultClassifierValidator(BaseValidator[str]):
    """Default validator that uses a classifier to validate text."""

    def __init__(
        self,
        config: RuleConfig,
        classifier: Optional[ClassifierProtocol] = None,
        validation_fn: Optional[Callable[[ClassificationResult], bool]] = None,
    ) -> None:
        """
        Initialize the validator.

        Args:
            config: Configuration for the validator
            classifier: Optional classifier to use (if not provided, will create from config)
            validation_fn: Optional function that determines if a classification result is valid

        Raises:
            ConfigurationError: If classifier is invalid
        """
        self._config = config

        # Create or validate classifier
        if classifier is None:
            classifier_name = config.params.get("classifier_name", "")
            if not classifier_name:
                raise ConfigurationError(
                    "Must provide either classifier or classifier_name in config params"
                )
            # Here you would create the classifier based on config.params.get("classifier_name")
            # For now we'll raise an error since we don't have the factory logic
            raise NotImplementedError("Classifier creation from config not implemented")
        else:
            self._validate_classifier(classifier)
            self._classifier = classifier

        # Set validation function
        self._validation_fn = validation_fn or self._default_validation_fn

    def _validate_classifier(self, classifier: Any) -> TypeGuard[ClassifierProtocol]:
        """Validate that a classifier implements the required protocol."""
        if not isinstance(classifier, ClassifierProtocol):
            raise ConfigurationError(
                f"Classifier must implement ClassifierProtocol, got {type(classifier)}"
            )
        return True

    def _default_validation_fn(self, result: ClassificationResult) -> bool:
        """Default validation function using threshold and valid labels."""
        threshold = self._config.params.get("threshold", 0.5)
        valid_labels = self._config.params.get("valid_labels", [])

        if result.confidence < threshold:
            return False
        if valid_labels and result.label not in valid_labels:
            return False
        return True

    @property
    def classifier(self) -> ClassifierProtocol:
        """Get the classifier."""
        return self._classifier

    @property
    def config(self) -> RuleConfig:
        """Get the configuration."""
        return self._config

    def validate(self, output: str, **_) -> RuleResult:
        """
        Validate text using the classifier.

        Args:
            output: The text to validate
            **_: Additional validation context (unused)

        Returns:
            RuleResult with validation results

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Classify the text
            result = self._classifier.classify(output)

            # Check if result meets validation criteria
            passed = self._validation_fn(result)

            # Build metadata
            metadata = {
                "classifier_name": self._classifier.name,
                "classifier_result": (
                    result.model_dump() if hasattr(result, "model_dump") else vars(result)
                ),
                "threshold": self._config.params.get("threshold", 0.5),
                "valid_labels": self._config.params.get("valid_labels", []),
            }

            # Return result
            return RuleResult(
                passed=passed,
                message=f"Classification {'passed' if passed else 'failed'}: {result.label}",
                metadata=metadata,
                score=result.confidence,
            )

        except Exception as e:
            raise ValidationError(f"Classification failed: {str(e)}") from e


class ClassifierRule(Rule[str, RuleResult, DefaultClassifierValidator, Any]):
    """Rule that uses a classifier to validate text."""

    def __init__(
        self,
        name: str = "classifier_rule",
        description: str = "Validates text using a classifier",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultClassifierValidator] = None,
        classifier: Optional[ClassifierProtocol] = None,
        validation_fn: Optional[Callable[[ClassificationResult], bool]] = None,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a classifier rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            classifier: Optional classifier to use if no validator provided
            validation_fn: Optional validation function if no validator provided
            threshold: Confidence threshold for validation (0-1)
            valid_labels: List of valid labels for validation

        Raises:
            ConfigurationError: If neither validator nor classifier is provided
        """
        # Store classifier and validation function for creating the default validator
        self._classifier = classifier
        self._validation_fn = validation_fn

        # Create config if not provided
        if config is None:
            params = {
                "threshold": threshold,
                "valid_labels": valid_labels or [],
            }
            config = RuleConfig(params=params)
        elif threshold != 0.5 or valid_labels is not None:
            # If config is provided but threshold or valid_labels are also provided,
            # update the config params
            params = dict(config.params)
            if threshold != 0.5:
                params["threshold"] = threshold
            if valid_labels is not None:
                params["valid_labels"] = valid_labels
            config = RuleConfig(
                priority=config.priority,
                cache_size=config.cache_size,
                cost=config.cost,
                params=params,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultClassifierValidator:
        """Create a default validator from config."""
        if self._classifier is None:
            raise ConfigurationError("Must provide a classifier")

        return DefaultClassifierValidator(self.config, self._classifier, self._validation_fn)

    @property
    def threshold(self) -> float:
        """Get the confidence threshold."""
        return self.config.params.get("threshold", 0.5)

    @property
    def valid_labels(self) -> List[str]:
        """Get the list of valid labels."""
        return self.config.params.get("valid_labels", [])

    @property
    def classifier(self) -> ClassifierProtocol:
        """Get the classifier."""
        return self._classifier


def create_classifier_rule(
    name: str = "classifier_rule",
    description: str = "Validates text using a classifier",
    config: Optional[Dict[str, Any]] = None,
    classifier: Optional[ClassifierProtocol] = None,
    validation_fn: Optional[Callable[[ClassificationResult], bool]] = None,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
) -> ClassifierRule:
    """
    Create a classifier rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary
        classifier: Optional classifier to use
        validation_fn: Optional validation function
        threshold: Confidence threshold for validation (0-1)
        valid_labels: List of valid labels for validation

    Returns:
        Configured ClassifierRule instance
    """
    # Create params dictionary from config or defaults
    params = {}
    if config:
        params.update(config)

    # Override with explicit parameters if provided
    if threshold != 0.5:
        params["threshold"] = threshold
    if valid_labels is not None:
        params["valid_labels"] = valid_labels

    # Create rule config
    rule_config = RuleConfig(params=params)

    return ClassifierRule(
        name=name,
        description=description,
        config=rule_config,
        classifier=classifier,
        validation_fn=validation_fn,
    )


# Export public classes and functions
__all__ = [
    "ClassifierRule",
    "ClassifierProtocol",
    "DefaultClassifierValidator",
    "create_classifier_rule",
]
