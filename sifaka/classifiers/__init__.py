"""Classifiers for Sifaka.

This module provides text classifiers for analyzing various aspects of generated text.
These classifiers are designed to work with the new PydanticAI-based architecture.
"""

from .base import BaseClassifier, CachedClassifier, ClassificationResult
from .sentiment import SentimentClassifier, CachedSentimentClassifier, create_sentiment_classifier
from .toxicity import ToxicityClassifier, CachedToxicityClassifier, create_toxicity_classifier
from .spam import SpamClassifier, CachedSpamClassifier, create_spam_classifier
from .language import LanguageClassifier, CachedLanguageClassifier, create_language_classifier

__all__ = [
    # Base classes
    "BaseClassifier",
    "CachedClassifier",
    "ClassificationResult",
    # Sentiment classification
    "SentimentClassifier",
    "CachedSentimentClassifier",
    "create_sentiment_classifier",
    # Toxicity classification
    "ToxicityClassifier",
    "CachedToxicityClassifier",
    "create_toxicity_classifier",
    # Spam classification
    "SpamClassifier",
    "CachedSpamClassifier",
    "create_spam_classifier",
    # Language classification
    "LanguageClassifier",
    "CachedLanguageClassifier",
    "create_language_classifier",
]
