"""
Classifier implementations.

This package provides concrete implementations of classifiers in the Sifaka framework.
"""

# Content analysis classifiers
from .content.bias import BiasDetector, create_bias_detector
from .content.profanity import ProfanityClassifier, create_profanity_classifier
from .content.sentiment import SentimentClassifier, create_sentiment_classifier
from .content.spam import SpamClassifier, create_spam_classifier
from .content.toxicity import ToxicityClassifier, create_toxicity_classifier

# Text properties classifiers
from .properties.genre import GenreClassifier, create_genre_classifier
from .properties.language import LanguageClassifier, create_language_classifier
from .properties.readability import ReadabilityClassifier, create_readability_classifier
from .properties.topic import TopicClassifier, create_topic_classifier

# Entity analysis classifiers
from .entities.ner import NERClassifier, create_ner_classifier

__all__ = [
    # Content analysis classifiers
    "BiasDetector",
    "ProfanityClassifier",
    "SentimentClassifier",
    "SpamClassifier",
    "ToxicityClassifier",
    # Text properties classifiers
    "GenreClassifier",
    "LanguageClassifier",
    "ReadabilityClassifier",
    "TopicClassifier",
    # Entity analysis classifiers
    "NERClassifier",
    # Factory functions
    "create_bias_detector",
    "create_profanity_classifier",
    "create_sentiment_classifier",
    "create_spam_classifier",
    "create_toxicity_classifier",
    "create_genre_classifier",
    "create_language_classifier",
    "create_readability_classifier",
    "create_topic_classifier",
    "create_ner_classifier",
]
