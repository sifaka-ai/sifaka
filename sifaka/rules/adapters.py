from typing import (
    Optional,
    Type,
    Union
)

from ..classifiers.base import Classifier, ClassifierConfig
from .base import Rule, RuleConfig, RuleResult

class ClassifierRuleAdapter(Rule[str, RuleResult, None, None]):
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
        if isinstance(classifier, type):
            # If classifier is a class, instantiate it
            classifier_name = classifier.__name__
            self.classifier = classifier(config=config)
        else:
            # If classifier is an instance, use it directly
            classifier_name = classifier.__class__.__name__
            self.classifier = classifier

        super().__init__(
            name=f"{classifier_name}Rule",
            description=f"Rule adapter for {classifier_name}",
            config=rule_config,
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
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
