"""
Content analysis classifiers.

This package provides classifiers for analyzing content characteristics such as
sentiment, toxicity, profanity, spam, and bias.
"""

from typing import Any, List
from .bias import BiasDetector, create_bias_detector
from .profanity import ProfanityClassifier, create_profanity_classifier
from .sentiment import SentimentClassifier, create_sentiment_classifier
from .spam import SpamClassifier, create_spam_classifier
from .toxicity import ToxicityClassifier, create_toxicity_classifier

__all__: List[Any] = [
    "BiasDetector",
    "ProfanityClassifier",
    "SentimentClassifier",
    "SpamClassifier",
    "ToxicityClassifier",
    "create_bias_detector",
    "create_profanity_classifier",
    "create_sentiment_classifier",
    "create_spam_classifier",
    "create_toxicity_classifier",
]
