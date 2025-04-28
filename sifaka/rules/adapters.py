"""
Adapter module for using classifiers as rules.

.. deprecated:: 1.0.0
   This module is deprecated and will be removed in version 2.0.0.
   Use :mod:`sifaka.rules.adapters` instead.

Migration guide:
1. Replace imports:
   - Old: from sifaka.rules.adapters import ClassifierAdapter, ClassifierRuleAdapter
   - New: from sifaka.rules.adapters import ClassifierAdapter, ClassifierRule

2. Class name changes:
   - ClassifierRuleAdapter is now ClassifierRule
   - Interface has been improved with more options and better error handling
"""

import warnings
from typing import Any, Dict, Optional, Type, Union

from ..classifiers.base import Classifier, ClassifierConfig
from .base import BaseValidator, Rule, RuleConfig, RuleResult


# Emit deprecation warning
warnings.warn(
    "The sifaka.rules.adapters module is deprecated and will be removed in version 2.0.0. "
    "Use sifaka.rules.adapters package instead.",
    DeprecationWarning,
    stacklevel=2,
)


class ClassifierAdapter(BaseValidator[str]):
    """Validator that adapts a classifier for validation."""

    def __init__(self, classifier: Classifier) -> None:
        """Initialize with a classifier instance."""
        self._classifier = classifier

    @property
    def classifier(self) -> Classifier:
        """Get the classifier."""
        return self._classifier

    def validate(self, output: str, **_) -> RuleResult:
        """
        Validate output using the classifier.

        Args:
            output: The text to validate
            **_: Additional validation context (unused)

        Returns:
            RuleResult with validation results
        """
        try:
            result = self.classifier.classify(output)
            return RuleResult(
                passed=True,
                message=f"Classification successful: {result.label}",
                metadata={"classification_result": result},
                score=result.confidence,
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Classification failed: {str(e)}",
                metadata={"error": str(e)},
                score=0.0,
            )


class ClassifierRuleAdapter(Rule[str, RuleResult, ClassifierAdapter, Any]):
    """Adapter to use a classifier as a validation rule."""

    def __init__(
        self,
        classifier: Union[Type[Classifier], Classifier],
        classifier_config: Optional[Dict[str, Any]] = None,
        rule_config: Optional[RuleConfig] = None,
    ) -> None:
        """
        Initialize the adapter with a classifier class or instance and optional configs.

        Args:
            classifier: Either a Classifier class or instance
            classifier_config: Optional classifier configuration dictionary (only used if classifier is a class)
            rule_config: Optional rule configuration
        """
        # Store the classifier for creating the validator
        if isinstance(classifier, type):
            # If classifier is a class, instantiate it with config
            classifier_name = classifier.__name__

            # Create classifier config if provided
            if classifier_config:
                # Extract labels if provided
                labels = classifier_config.pop("labels", [])
                # Create config with remaining params
                config = ClassifierConfig(labels=labels, params=classifier_config)
                self._classifier_instance = classifier(
                    name=f"{classifier_name.lower()}",
                    description=f"{classifier_name} classifier",
                    config=config,
                )
            else:
                # Create with default config
                self._classifier_instance = classifier(
                    name=f"{classifier_name.lower()}", description=f"{classifier_name} classifier"
                )
        else:
            # If classifier is an instance, use it directly
            classifier_name = classifier.__class__.__name__
            self._classifier_instance = classifier

        # Initialize base class
        super().__init__(
            name=f"{classifier_name}Rule",
            description=f"Rule adapter for {classifier_name}",
            config=rule_config,
        )

    @property
    def classifier(self) -> Classifier:
        """Get the classifier instance."""
        return self._classifier_instance

    def _create_default_validator(self) -> ClassifierAdapter:
        """Create a default validator using the classifier."""
        return ClassifierAdapter(self._classifier_instance)
