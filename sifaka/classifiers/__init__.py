"""Classifiers for Sifaka.

This module provides text classifiers for analyzing various aspects of generated text.
These classifiers are designed to work with the new PydanticAI-based architecture.
"""

from .base import BaseClassifier, CachedClassifier, ClassificationResult
from .emotion import CachedEmotionClassifier, EmotionClassifier, create_emotion_classifier
from .intent import CachedIntentClassifier, IntentClassifier, create_intent_classifier
from .language import CachedLanguageClassifier, LanguageClassifier, create_language_classifier
from .readability import (
    CachedReadabilityClassifier,
    ReadabilityClassifier,
    create_readability_classifier,
)
from .sentiment import CachedSentimentClassifier, SentimentClassifier, create_sentiment_classifier
from .spam import CachedSpamClassifier, SpamClassifier, create_spam_classifier
from .toxicity import CachedToxicityClassifier, ToxicityClassifier, create_toxicity_classifier

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
    # Readability classification
    "ReadabilityClassifier",
    "CachedReadabilityClassifier",
    "create_readability_classifier",
    # Emotion classification
    "EmotionClassifier",
    "CachedEmotionClassifier",
    "create_emotion_classifier",
    # Intent classification
    "IntentClassifier",
    "CachedIntentClassifier",
    "create_intent_classifier",
]
