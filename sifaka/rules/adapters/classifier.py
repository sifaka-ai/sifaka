"""
Classifier adapter module for Sifaka.

This module provides adapters for using classifiers as validation rules.
It allows you to plug in any classifier from the classifiers module and
use it as a rule in validation pipelines.
"""

from typing import Any, Callable, Dict, Final, List, Optional, Type, Union

from sifaka.classifiers.base import Classification, Classifier, ClassifierConfig
from sifaka.rules.adapters.base import Adaptable, BaseAdapter
from sifaka.rules.base import (
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)


class ClassifierAdapter(BaseAdapter):
    """Adapter that enables a classifier to function as a validator."""

    def __init__(
        self,
        classifier: Classifier,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        validation_fn: Optional[Callable[[Classification], bool]] = None,
    ) -> None:
        """
        Initialize with a classifier.

        Args:
            classifier: Classifier to adapt
            threshold: Confidence threshold for validation (0.0 to 1.0)
            valid_labels: Optional list of valid classification labels
            validation_fn: Optional custom validation function
        """
        super().__init__(classifier)
        self._threshold = threshold
        self._valid_labels = valid_labels or []
        self._validation_fn = validation_fn or self._default_validation_fn

    @property
    def classifier(self) -> Classifier:
        """Get the classifier."""
        return self._adaptee  # type: ignore

    @property
    def threshold(self) -> float:
        """Get the threshold."""
        return self._threshold

    @property
    def valid_labels(self) -> List[str]:
        """Get the valid labels."""
        return self._valid_labels

    def _default_validation_fn(self, result: Classification) -> bool:
        """
        Default validation logic.

        Args:
            result: Classification result to validate

        Returns:
            True if result passes validation, False otherwise
        """
        # Check if confidence is above threshold
        if result.confidence < self.threshold:
            return False

        # If valid_labels is set, check that label is in the list
        if self.valid_labels and result.label not in self.valid_labels:
            return False

        return True

    def validate(self, text: str, **kwargs) -> RuleResult:
        """
        Validate text using the classifier.

        Args:
            text: Text to classify and validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results

        Raises:
            ValidationError: If validation fails
        """
        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")

            # Classify the text
            result = self.classifier.classify(text)

            # Check if result meets validation criteria
            passed = self._validation_fn(result)

            # Create metadata
            metadata = {
                "classifier": self.classifier.name,
                "label": result.label,
                "confidence": result.confidence,
                "threshold": self.threshold,
                "valid_labels": self.valid_labels,
                # Include full classification result when available
                "classification": (
                    result.model_dump() if hasattr(result, "model_dump") else vars(result)
                ),
            }

            return RuleResult(
                passed=passed,
                message=f"Classification {'passed' if passed else 'failed'}: {result.label} ({result.confidence:.2f})",
                metadata=metadata,
                score=result.confidence,
            )

        except Exception as e:
            raise ValidationError(f"Classification failed: {str(e)}") from e


class ClassifierRule(Rule[str, RuleResult, ClassifierAdapter, RuleResultHandler[RuleResult]]):
    """Rule that leverages a classifier for validation."""

    def __init__(
        self,
        classifier: Union[Type[Classifier], Classifier],
        name: Optional[str] = None,
        description: Optional[str] = None,
        classifier_config: Optional[Dict[str, Any]] = None,
        rule_config: Optional[RuleConfig] = None,
        threshold: float = 0.5,
        valid_labels: Optional[List[str]] = None,
        validation_fn: Optional[Callable[[Classification], bool]] = None,
    ) -> None:
        """
        Initialize with a classifier.

        Args:
            classifier: Classifier class or instance to use
            name: Name for the rule (defaults to classifier name + "Rule")
            description: Description for the rule (defaults to classifier description)
            classifier_config: Configuration for the classifier (only used if classifier is a class)
            rule_config: Configuration for the rule
            threshold: Confidence threshold for validation (0.0 to 1.0)
            valid_labels: List of valid labels for validation
            validation_fn: Custom validation function

        Raises:
            ConfigurationError: If classifier configuration is invalid
        """
        # Store validation parameters
        self._threshold = threshold
        self._valid_labels = valid_labels or []
        self._validation_fn = validation_fn

        # Instantiate classifier if needed
        if isinstance(classifier, type):
            # If classifier is a class, instantiate it with config
            classifier_class_name = classifier.__name__

            # Create classifier config if provided
            if classifier_config:
                # Extract labels if provided
                labels = classifier_config.pop("labels", [])
                # Create config with remaining params
                config = ClassifierConfig(labels=labels, params=classifier_config)
                self._classifier_instance = classifier(
                    name=classifier_config.get("name", f"{classifier_class_name.lower()}"),
                    description=classifier_config.get(
                        "description", f"{classifier_class_name} classifier"
                    ),
                    config=config,
                )
            else:
                # Create with default config
                self._classifier_instance = classifier(
                    name=f"{classifier_class_name.lower()}",
                    description=f"{classifier_class_name} classifier",
                )
        else:
            # If classifier is an instance, use it directly
            self._classifier_instance = classifier

        # Create rule name and description based on classifier if not provided
        if name is None:
            name = f"{self._classifier_instance.name}_rule"
        if description is None:
            description = f"Rule using {self._classifier_instance.name} classifier"

        # Create or update rule config
        final_rule_config = self._create_or_update_rule_config(rule_config)

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=final_rule_config,
            validator=None,
            result_handler=None,
        )

    def _create_or_update_rule_config(self, rule_config: Optional[RuleConfig]) -> RuleConfig:
        """
        Create or update rule configuration.

        Args:
            rule_config: Optional existing rule configuration

        Returns:
            Updated or new rule configuration
        """
        params = {
            "threshold": self._threshold,
            "valid_labels": self._valid_labels,
            "classifier_name": self._classifier_instance.name,
        }

        # If existing config provided, merge params
        if rule_config:
            # Start with existing params or empty dict
            merged_params = dict(rule_config.params or {})
            # Update with our params
            merged_params.update(params)
            # Create new config preserving other fields
            return RuleConfig(
                priority=rule_config.priority,
                cache_size=rule_config.cache_size,
                cost=rule_config.cost,
                params=merged_params,
            )
        else:
            # Create new config with our params
            return RuleConfig(params=params)

    @property
    def classifier(self) -> Classifier:
        """Get the classifier instance."""
        return self._classifier_instance

    def _create_default_validator(self) -> ClassifierAdapter:
        """
        Create default validator.

        Returns:
            Configured ClassifierAdapter
        """
        return ClassifierAdapter(
            classifier=self._classifier_instance,
            threshold=self._threshold,
            valid_labels=self._valid_labels,
            validation_fn=self._validation_fn,
        )


def create_classifier_rule(
    classifier: Union[Type[Classifier], Classifier],
    name: Optional[str] = None,
    description: Optional[str] = None,
    classifier_config: Optional[Dict[str, Any]] = None,
    rule_config: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
    valid_labels: Optional[List[str]] = None,
    validation_fn: Optional[Callable[[Classification], bool]] = None,
) -> ClassifierRule:
    """
    Create a classifier rule.

    Args:
        classifier: Classifier class or instance to use
        name: Name for the rule (defaults to classifier name + "Rule")
        description: Description for the rule (defaults to classifier description)
        classifier_config: Configuration for the classifier (only if classifier is a class)
        rule_config: Configuration for the rule
        threshold: Confidence threshold for validation (0.0 to 1.0)
        valid_labels: List of valid labels for validation
        validation_fn: Custom validation function

    Returns:
        Configured ClassifierRule
    """
    # Convert rule_config dict to RuleConfig if provided
    rule_config_obj = RuleConfig(params=rule_config) if rule_config else None

    return ClassifierRule(
        classifier=classifier,
        name=name,
        description=description,
        classifier_config=classifier_config,
        rule_config=rule_config_obj,
        threshold=threshold,
        valid_labels=valid_labels,
        validation_fn=validation_fn,
    )
