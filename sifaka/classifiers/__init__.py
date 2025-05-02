"""
Sifaka Classifiers Package

This package provides a collection of text classifiers that analyze content for various
characteristics. Each classifier follows a consistent interface and can be used
independently or integrated with rules.

Architecture Overview:
- Each classifier inherits from the base BaseClassifier class
- Classifiers return ClassificationResult objects with consistent structure
- Classifiers can be combined with rules using ClassifierRule

Available Classifiers:
1. Base Classes:
   - BaseClassifier: Abstract base class for all classifiers
   - Classifier: Alias for BaseClassifier
   - ClassificationResult: Standard result format
   - ClassifierConfig: Configuration for classifiers
   - ClassifierProtocol: Protocol for classifier implementations
   - TextProcessor: Utility for text preprocessing

2. Content Analysis:
   - SentimentClassifier: Analyzes text sentiment (positive/negative/neutral)
   - ProfanityClassifier: Detects profane or inappropriate language
   - ToxicityClassifier: Identifies toxic content
   - SpamClassifier: Detects spam content in text
   - BiasDetector: Identifies various forms of bias in text

3. Text Properties:
   - ReadabilityClassifier: Evaluates reading difficulty level
   - LanguageClassifier: Identifies the language of text
   - TopicClassifier: Identifies topics in text using LDA
   - GenreClassifier: Categorizes text into genres (news, fiction, academic, etc.)

4. Entity Analysis:
   - NERClassifier: Identifies named entities (people, organizations, locations, etc.)

Usage Example:
    from sifaka.classifiers import SentimentClassifier, create_sentiment_classifier

    # Create classifiers directly
    sentiment = SentimentClassifier()

    # Or use factory functions (recommended)
    sentiment = create_sentiment_classifier(
        positive_threshold=0.1,
        negative_threshold=-0.1
    )

    # Analyze text
    sentiment_result = sentiment.classify("This is fantastic!")

    # Use with rules
    from sifaka.rules import ClassifierRule
    sentiment_rule = ClassifierRule(classifier=sentiment)
"""

from .base import (
    BaseClassifier,
    Classifier,
    ClassificationResult,
    ClassifierConfig,
    ClassifierProtocol,
    TextProcessor,
    TextClassifier,
)
from .bias import BiasDetector
from .genre import GenreClassifier
from .language import LanguageClassifier
from .ner import NERClassifier, create_ner_classifier
from .profanity import ProfanityClassifier, create_profanity_classifier
from .readability import ReadabilityClassifier
from .sentiment import SentimentClassifier, create_sentiment_classifier
from .spam import SpamClassifier, create_spam_classifier
from .topic import TopicClassifier
from .toxicity import ToxicityClassifier, create_toxicity_classifier

__all__ = [
    # Base Classes
    "BaseClassifier",
    "Classifier",
    "ClassificationResult",
    "ClassifierConfig",
    "ClassifierProtocol",
    "TextProcessor",
    "TextClassifier",

    # Content Analysis Classes
    "SentimentClassifier",
    "ProfanityClassifier",
    "ToxicityClassifier",
    "SpamClassifier",
    "BiasDetector",

    # Text Properties Classes
    "ReadabilityClassifier",
    "LanguageClassifier",
    "TopicClassifier",
    "GenreClassifier",

    # Entity Analysis
    "NERClassifier",

    # Factory Functions
    "create_sentiment_classifier",
    "create_toxicity_classifier",
    "create_spam_classifier",
    "create_profanity_classifier",
    "create_ner_classifier",
]
