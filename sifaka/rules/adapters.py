from typing import Any, Dict, Optional, Type, TypeVar, cast

from ..classifiers.base import Classifier, ClassifierConfig
from .base import Rule, RuleConfig, RuleResult


class ClassifierRuleAdapter(Rule[str, RuleResult, None, None]):
    """Adapter to use a classifier as a validation rule."""

    def __init__(
        self,
        classifier_cls: Type[Classifier],
        config: Optional[ClassifierConfig] = None,
        rule_config: Optional[RuleConfig] = None,
    ) -> None:
        """Initialize the adapter with a classifier class and optional configs."""
        super().__init__(
            name=f"{classifier_cls.__name__}Rule",
            description=f"Rule adapter for {classifier_cls.__name__}",
            config=rule_config,
        )
        self.classifier = classifier_cls(config=config)

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
