from typing import Any, List
"""
Entity analysis classifiers.

This package provides classifiers for analyzing entities in text, such as
named entity recognition (NER).
"""
from .ner import NERClassifier, create_ner_classifier
__all__: List[Any] = ['NERClassifier', 'create_ner_classifier']
