"""
Rule implementation that uses pluggable classifiers.
"""

from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    Callable,
    Dict,
    Any,
    TypeVar,
    Generic,
    cast,
    Protocol,
    runtime_checkable,
    Final,
    Union,
    Type,
)
from typing_extensions import TypeGuard

from sifaka.classifiers.base import (
    Classifier,
    ClassificationResult,
    ClassifierConfig,
    ClassifierProtocol,
)
from sifaka.rules.base import (
    Rule,
    RuleResult,
    RuleValidator,
    RuleResultHandler,
    RuleConfig,
    ValidationError,
    ConfigurationError,
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

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.threshold <= 1:
            raise ConfigurationError("Threshold must be between 0 and 1")

    def with_threshold(self, threshold: float) -> "ClassifierRuleConfig":
        """Create a new config with updated threshold."""
        return self.with_options(threshold=threshold)

    def with_labels(self, labels: List[str]) -> "ClassifierRuleConfig":
        """Create a new config with updated valid labels."""
        return self.with_options(valid_labels=labels)


class ClassifierValidator(RuleValidator[str]):
    """Validator that uses a classifier to validate text."""

    def __init__(
        self,
        classifier: ClassifierProtocol,
        validation_fn: Callable[[ClassificationResult], bool],
        config: ClassifierRuleConfig,
    ) -> None:
        """
        Initialize the validator.

        Args:
            classifier: The classifier to use
            validation_fn: Function that determines if a classification result is valid
            config: Configuration for the validator

        Raises:
            ConfigurationError: If classifier is invalid
        """
        self._validate_classifier(classifier)
        self._classifier: Final[ClassifierProtocol] = classifier
        self._validation_fn: Final[Callable[[ClassificationResult], bool]] = validation_fn
        self._config: Final[ClassifierRuleConfig] = config

    def _validate_classifier(self, classifier: Any) -> TypeGuard[ClassifierProtocol]:
        """Validate that a classifier implements the required protocol."""
        if not isinstance(classifier, ClassifierProtocol):
            raise ConfigurationError(
                f"Classifier must implement ClassifierProtocol, got {type(classifier)}"
            )
        return True

    @property
    def classifier(self) -> ClassifierProtocol:
        """Get the classifier."""
        return self._classifier

    @property
    def config(self) -> ClassifierRuleConfig:
        """Get the configuration."""
        return self._config

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator accepts."""
        return str

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

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
                "classifier_result": result.dict(),
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


class ClassifierRule(Rule[str, RuleResult, ClassifierValidator, RuleResultHandler[RuleResult]]):
    """
    Rule that uses a classifier to validate text.

    This rule allows plugging in any classifier that implements the ClassifierProtocol
    and provides a validation function to determine if the classification result is valid.
    """

    def __init__(
        self,
        name: str,
        description: str,
        classifier: ClassifierProtocol,
        validation_fn: Optional[Callable[[ClassificationResult], bool]] = None,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        config: Optional[RuleConfig] = None,
        result_handler: Optional[RuleResultHandler[RuleResult]] = None,
    ) -> None:
        """
        Initialize a classifier rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            classifier: The classifier to use
            validation_fn: Optional function that determines if a classification result is valid
            threshold: Confidence threshold for validation
            valid_labels: List of valid labels
            config: Additional rule configuration
            result_handler: Optional handler for validation results

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Create config
        base_config = config or RuleConfig()
        rule_config = ClassifierRuleConfig(
            threshold=threshold,
            valid_labels=valid_labels or [],
            priority=base_config.priority,
            cache_size=base_config.cache_size,
            cost=base_config.cost,
            metadata=base_config.metadata,
        )

        # Create default validation function if none provided
        if validation_fn is None:
            validation_fn = lambda r: (
                r.confidence >= rule_config.threshold
                and (not rule_config.valid_labels or r.label in rule_config.valid_labels)
            )

        # Create validator
        validator = ClassifierValidator(
            classifier=classifier,
            validation_fn=validation_fn,
            config=rule_config,
        )

        super().__init__(
            name=name,
            description=description,
            validator=validator,
            config=rule_config,
            result_handler=result_handler,
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
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
        return self._validator.validate(output, **kwargs)

    @property
    def classifier(self) -> ClassifierProtocol:
        """Get the underlying classifier."""
        return self._validator.classifier

    @property
    def threshold(self) -> float:
        """Get the confidence threshold."""
        return cast(ClassifierRuleConfig, self.config).threshold

    @property
    def valid_labels(self) -> List[str]:
        """Get the list of valid labels."""
        return cast(ClassifierRuleConfig, self.config).valid_labels
