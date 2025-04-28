"""
Rule implementation that uses pluggable classifiers.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import TypeGuard

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

T = TypeVar("T", bound=ClassificationResult)


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Protocol for classifier components."""

    def classify(self, text: str) -> ClassificationResult: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...


@dataclass(frozen=True)
class ClassifierRuleConfig(RuleConfig):
    """Immutable configuration for classifier rules."""

    threshold: float = 0.5
    valid_labels: List[str] = field(default_factory=list)
    classifier_name: str = ""
    classifier_config: Optional[Dict[str, Any]] = None
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.threshold <= 1:
            raise ConfigurationError("Threshold must be between 0 and 1")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")

    def with_threshold(self, threshold: float) -> "ClassifierRuleConfig":
        """Create a new config with updated threshold."""
        return self.with_options(threshold=threshold)

    def with_labels(self, labels: List[str]) -> "ClassifierRuleConfig":
        """Create a new config with updated valid labels."""
        return self.with_options(valid_labels=labels)


class DefaultClassifierValidator(BaseValidator[str]):
    """Default validator that uses a classifier to validate text."""

    def __init__(
        self,
        config: ClassifierRuleConfig,
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
            if not config.classifier_name:
                raise ConfigurationError(
                    "Must provide either classifier or classifier_name in config"
                )
            # Here you would create the classifier based on config.classifier_name and config.classifier_config
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
        if result.confidence < self.config.threshold:
            return False
        if self.config.valid_labels and result.label not in self.config.valid_labels:
            return False
        return True

    @property
    def classifier(self) -> ClassifierProtocol:
        """Get the classifier."""
        return self._classifier

    @property
    def config(self) -> ClassifierRuleConfig:
        """Get the configuration."""
        return self._config

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate text using the classifier.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

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
                    result.model_dump() if hasattr(result, "model_dump") else result.dict()
                ),
                "threshold": self._config.threshold,
                "valid_labels": self._config.valid_labels,
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

        Raises:
            ConfigurationError: If neither validator nor classifier is provided
        """
        # Store classifier and validation function for creating the default validator
        self._classifier = classifier
        self._validation_fn = validation_fn

        # Store rule parameters
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultClassifierValidator:
        """Create a default validator from config."""
        if self._classifier is None:
            raise ConfigurationError("Must provide a classifier")

        rule_config = ClassifierRuleConfig(**self._rule_params)
        return DefaultClassifierValidator(rule_config, self._classifier, self._validation_fn)


def create_classifier_rule(
    name: str = "classifier_rule",
    description: str = "Validates text using a classifier",
    config: Optional[Dict[str, Any]] = None,
    classifier: Optional[ClassifierProtocol] = None,
    validation_fn: Optional[Callable[[ClassificationResult], bool]] = None,
) -> ClassifierRule:
    """
    Create a classifier rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary
        classifier: Optional classifier to use
        validation_fn: Optional validation function

    Returns:
        Configured ClassifierRule instance
    """
    if config is None:
        config = {
            "threshold": 0.5,
            "valid_labels": [],
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config)

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
    "ClassifierRuleConfig",
    "ClassifierProtocol",
    "DefaultClassifierValidator",
    "create_classifier_rule",
]
