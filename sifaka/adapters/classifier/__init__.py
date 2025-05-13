from typing import Any, List
"""
Adapter module for using classifiers as rules.

This module provides adapters for using classifiers as validation rules. It enables
the integration of classification models into Sifaka's validation system, allowing
for sophisticated content analysis and validation.

Example usage:
    >>> from sifaka.classifiers import SentimentClassifier
    >>> from sifaka.adapters.classifier import create_classifier_rule
    >>>
    >>> # Create a classifier rule
    >>> rule = create_classifier_rule(
    >>>     classifier=SentimentClassifier(),
    >>>     threshold=0.7,
    >>>     valid_labels=["positive"]
    >>> )
    >>>
    >>> # Use the rule like any other rule
    >>> result = rule.validate("This is great!") if rule else ""  # True if classified as "positive" with confidence >= 0.7
"""
from sifaka.adapters.classifier.adapter import Classifier, ClassifierAdapter, ClassifierRule, ClassifierRuleConfig, create_classifier_rule, create_classifier_adapter
__all__: List[Any] = ['Classifier', 'ClassifierRuleConfig',
    'ClassifierRule', 'ClassifierAdapter', 'create_classifier_rule',
    'create_classifier_adapter']
