"""
Prohibited content validation rules for Sifaka.

This module provides validators and rules for checking text against prohibited content.

This module is now a thin wrapper around the ProfanityClassifier, which provides
more sophisticated prohibited content detection.

Usage Example:
    from sifaka.rules.content.prohibited import create_prohibited_content_rule

    # Create a prohibited content rule using the classifier adapter
    rule = create_prohibited_content_rule(
        terms=["inappropriate", "offensive", "vulgar"],
        threshold=0.5
    )

    # Validate text
    result = rule.validate("This is a test.")
"""

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.classifiers.base import ClassifierConfig
from sifaka.rules.adapters.classifier import create_classifier_rule
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.content.base import (
    ContentAnalyzer,
    ContentValidator,
    DefaultContentAnalyzer,
)


__all__ = [
    # Factory functions
    "create_prohibited_content_rule",
]


def create_prohibited_content_rule(
    name: str = "prohibited_content_rule",
    description: str = "Validates text for prohibited content",
    terms: Optional[List[str]] = None,
    threshold: float = 0.5,
    **kwargs,
) -> Rule[str, RuleResult, BaseValidator[str], RuleResultHandler[RuleResult]]:
    """
    Create a prohibited content rule using the classifier adapter.

    This factory function creates a configured prohibited content rule instance using the
    ProfanityClassifier through the classifier adapter.

    Args:
        name: The name of the rule
        description: Description of the rule
        terms: List of prohibited terms to check for
        threshold: Threshold for prohibited content detection (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured prohibited content rule instance
    """
    # Create the classifier with custom words if provided
    classifier = ProfanityClassifier(
        config=ClassifierConfig(
            labels=["clean", "profane", "unknown"], params={"custom_words": terms or []}
        )
    )

    # Extract rule_config from kwargs if present
    rule_config = kwargs.pop("rule_config", None)

    return create_classifier_rule(
        classifier=classifier,
        name=name,
        description=description,
        threshold=threshold,
        valid_labels=["clean"],
        rule_config=rule_config,
        **kwargs,
    )
