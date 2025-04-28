"""
Adapter module for converting various components to Sifaka rules.

This module provides adapters for using different components as validation rules.
Currently supported adapters:
- Classifier adapters: Convert classifiers to rules

Example usage:
    >>> from sifaka.classifiers import SentimentClassifier
    >>> from sifaka.rules.adapters import create_classifier_rule
    >>>
    >>> # Create a classifier rule
    >>> rule = create_classifier_rule(
    >>>     classifier=SentimentClassifier(),
    >>>     threshold=0.7,
    >>>     valid_labels=["positive"]
    >>> )
    >>>
    >>> # Use the rule like any other rule
    >>> result = rule.validate("This is great!")
    >>> print(result.passed)  # True if classified as "positive" with confidence >= 0.7
"""

from sifaka.rules.adapters.base import Adaptable, BaseAdapter
from sifaka.rules.adapters.classifier import (
    ClassifierAdapter,
    ClassifierRule,
    create_classifier_rule,
)

# Export public classes and functions
__all__ = [
    "Adaptable",
    "BaseAdapter",
    "ClassifierAdapter",
    "ClassifierRule",
    "create_classifier_rule",
]
