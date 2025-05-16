from typing import Any, List
"""
Text properties classifiers.

This package provides classifiers for analyzing text properties such as
readability, language, topic, and genre.
"""
from .genre import GenreClassifier, create_genre_classifier
from .language import LanguageClassifier, create_language_classifier
from .readability import ReadabilityClassifier, create_readability_classifier
from .topic import TopicClassifier, create_topic_classifier
__all__: List[Any] = ['GenreClassifier', 'LanguageClassifier',
    'ReadabilityClassifier', 'TopicClassifier', 'create_genre_classifier',
    'create_language_classifier', 'create_readability_classifier',
    'create_topic_classifier']
