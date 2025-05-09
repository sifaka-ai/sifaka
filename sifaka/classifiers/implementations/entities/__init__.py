"""
Entity analysis classifiers.

This package provides classifiers for analyzing entities in text, such as
named entity recognition (NER).
"""

from .ner import NERClassifier, create_ner_classifier

__all__ = [
    # Classes
    "NERClassifier",
    # Factory functions
    "create_ner_classifier",
]
