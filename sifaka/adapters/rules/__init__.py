"""
Adapter module for converting various components to Sifaka rules.

This module provides adapters for using different components as validation rules.
Currently supported adapters:
- Classifier adapters: Convert classifiers to rules
- Guardrails adapters: Use Guardrails validators as Sifaka rules

Example usage:
    >>> from sifaka.classifiers import SentimentClassifier
    >>> from sifaka.adapters.rules import create_classifier_rule
    >>>
    >>> # Create a classifier rule
    >>> rule = create_classifier_rule(
    >>>     classifier=SentimentClassifier(),
    >>>     threshold=0.7,
    >>>     valid_labels=["positive"]
    >>> )
    >>>
    >>> # Use the rule like any other rule
    >>> result = rule.validate("This is great!")  # True if classified as "positive" with confidence >= 0.7

    >>> # Example using Guardrails validator (if installed)
    >>> from guardrails.hub import RegexMatch
    >>> from sifaka.adapters.rules import create_guardrails_rule
    >>>
    >>> # Create a rule from a Guardrails validator
    >>> regex_rule = create_guardrails_rule(
    >>>     guardrails_validator=RegexMatch(regex=r"\d{3}-\d{3}-\d{4}"),
    >>>     rule_id="phone_format"
    >>> )
"""

from sifaka.adapters.rules.base import Adaptable, BaseAdapter
from sifaka.adapters.rules.classifier import (
    ClassifierAdapter,
    ClassifierRule,
    create_classifier_rule,
)

# Try to import Guardrails adapter if available
try:
    from sifaka.adapters.rules.guardrails_adapter import (
        GuardrailsValidatorAdapter,
        create_guardrails_rule,
    )

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

# Export public classes and functions
__all__ = [
    "Adaptable",
    "BaseAdapter",
    "ClassifierAdapter",
    "ClassifierRule",
    "create_classifier_rule",
]

# Add Guardrails adapter to exports if available
if GUARDRAILS_AVAILABLE:
    __all__.extend(
        [
            "GuardrailsValidatorAdapter",
            "create_guardrails_rule",
        ]
    )

"""
Adapters for using various validator systems with Sifaka.

This package provides adapters that allow validators from other systems
to be used within Sifaka's rule-based validation framework.
"""

# Import adapter factory functions for easier access
try:
    from .guardrails_adapter import create_guardrails_rule
except ImportError:
    # The adapter might not be available if the required dependencies aren't installed
    pass
