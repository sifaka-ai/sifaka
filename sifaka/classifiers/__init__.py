"""
Sifaka Classifiers Package

This package provides a collection of text classifiers that analyze content for various
characteristics. Each classifier follows a consistent interface and can be used
independently or integrated with rules.

Architecture Overview:
- Classifiers use composition over inheritance with implementation objects
- Classifiers return ClassificationResult objects with consistent structure
- Classifiers can be combined with rules using ClassifierRule

Available Classifiers:
1. Base Classes:
   - Classifier: Main classifier class using composition
   - ClassifierImplementation: Protocol for classifier implementations
   - ClassificationResult: Standard result format
   - ClassifierConfig: Configuration for classifiers
   - ClassifierProtocol: Protocol for classifier interfaces
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
   - Named Entity Recognition: Identifies named entities (people, organizations, locations, etc.)

Usage Example:
    from sifaka.classifiers import create_sentiment_classifier, create_toxicity_classifier

    # Create classifiers using factory functions (recommended)
    sentiment = create_sentiment_classifier(
        positive_threshold=0.1,
        negative_threshold=-0.1,
        cache_size=100
    )

    # Analyze text
    sentiment_result = sentiment.classify("This is fantastic!")
    print(f"Sentiment: {sentiment_result.label}, Confidence: {sentiment_result.confidence:.2f}")
    print(f"Compound score: {sentiment_result.metadata['compound_score']:.2f}")

    # Create another classifier
    toxicity = create_toxicity_classifier(
        general_threshold=0.5,
        cache_size=100
    )

    # Use with rules
    from sifaka.adapters.classifier import create_classifier_rule
    sentiment_rule = create_classifier_rule(
        classifier=sentiment,
        name="sentiment_rule",
        description="Ensures text has positive sentiment",
        threshold=0.6,
        valid_labels=["positive"]
    )
"""

from .base import (
    Classifier,
    ClassificationResult,
    ClassifierConfig,
    ClassifierProtocol,
    ClassifierImplementation,
    TextProcessor,
    TextClassifier,
)
from .bias import BiasDetector, create_bias_detector
from .genre import GenreClassifier, create_genre_classifier
from .language import LanguageClassifier, create_language_classifier
from .ner import create_ner_classifier, create_ner_classifier_with_custom_engine
from .profanity import ProfanityClassifier, create_profanity_classifier
from .readability import ReadabilityClassifier, create_readability_classifier
from .sentiment import SentimentClassifier, create_sentiment_classifier
from .spam import SpamClassifier, create_spam_classifier
from .topic import create_topic_classifier, create_pretrained_topic_classifier
from .toxicity import ToxicityClassifier, create_toxicity_classifier

__all__ = [
    # Base Classes
    "Classifier",
    "ClassificationResult",
    "ClassifierConfig",
    "ClassifierProtocol",
    "ClassifierImplementation",
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
    "GenreClassifier",
    # Factory Functions
    "create_sentiment_classifier",
    "create_toxicity_classifier",
    "create_spam_classifier",
    "create_profanity_classifier",
    "create_ner_classifier",
    "create_ner_classifier_with_custom_engine",
    "create_bias_detector",
    "create_language_classifier",
    "create_genre_classifier",
    "create_readability_classifier",
    "create_topic_classifier",
    "create_pretrained_topic_classifier",
]
