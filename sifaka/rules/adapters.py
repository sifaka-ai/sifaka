from typing import Any, Optional, Type, Union

from ..classifiers.base import Classifier, ClassifierConfig
from .base import BaseValidator, Rule, RuleConfig, RuleResult


class ClassifierAdapter(BaseValidator[str]):
    """Validator that adapts a classifier for validation."""

    def __init__(self, classifier: Classifier) -> None:
        """Initialize with a classifier instance."""
        self._classifier = classifier

    @property
    def classifier(self) -> Classifier:
        """Get the classifier."""
        return self._classifier

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate output using the classifier."""
        try:
            result = self.classifier.classify(output)
            return RuleResult(
                passed=True,
                message=f"Classification successful: {result}",
                metadata={"classification_result": result},
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Classification failed: {str(e)}",
                metadata={"error": str(e)},
            )


class ClassifierRuleAdapter(Rule[str, RuleResult, ClassifierAdapter, Any]):
    """Adapter to use a classifier as a validation rule."""

    def __init__(
        self,
        classifier: Union[Type[Classifier], Classifier],
        config: Optional[ClassifierConfig] = None,
        rule_config: Optional[RuleConfig] = None,
    ) -> None:
        """
        Initialize the adapter with a classifier class or instance and optional configs.

        Args:
            classifier: Either a Classifier class or instance
            config: Optional classifier configuration (only used if classifier is a class)
            rule_config: Optional rule configuration
        """
        # Store the classifier for creating the validator
        if isinstance(classifier, type):
            # If classifier is a class, instantiate it
            classifier_name = classifier.__name__
            self._classifier_instance = classifier(config=config)
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

    def _create_default_validator(self) -> ClassifierAdapter:
        """Create a default validator using the classifier."""
        return ClassifierAdapter(self._classifier_instance)
