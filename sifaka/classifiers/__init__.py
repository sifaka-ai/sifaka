"""
Sifaka Classifiers Package

A comprehensive package for text classification in the Sifaka framework.

## Overview
This package provides a collection of text classifiers that analyze content for various
characteristics. Each classifier follows a consistent interface and can be used
independently or integrated with rules and critics in the Sifaka framework.

## Components
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

## Usage Examples
```python
from sifaka.classifiers import create_sentiment_classifier, create_toxicity_classifier

# Create classifiers using factory functions
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
```

## Error Handling
The package handles errors by:
- Validating input parameters
- Providing clear error messages
- Supporting graceful fallbacks
- Including error details in results

## Configuration
The package supports various configuration options:
- Factory functions for easy instantiation
- Flexible parameter customization
- Caching strategies
- Integration with rules and critics
"""

# Base components
from .base import BaseClassifier, Classifier, TextClassifier

# Models
from .models import ClassificationResult

# Configuration
from .config import standardize_classifier_config
from .config import ClassifierConfig

# Interfaces
from .interfaces.classifier import ClassifierProtocol, TextProcessor

# Managers
from .managers.state import StateManager

# Strategies
from .strategies.caching import CachingStrategy

# Factory functions
from .factories import create_classifier_by_name

# Implementations
from .implementations import (
    # Content analysis classifiers
    BiasDetector,
    ProfanityClassifier,
    SentimentClassifier,
    SpamClassifier,
    ToxicityClassifier,
    # Text properties classifiers
    GenreClassifier,
    LanguageClassifier,
    ReadabilityClassifier,
    TopicClassifier,
    # Entity analysis classifiers
    NERClassifier,
    # Factory functions
    create_bias_detector,
    create_profanity_classifier,
    create_sentiment_classifier,
    create_spam_classifier,
    create_toxicity_classifier,
    create_genre_classifier,
    create_language_classifier,
    create_readability_classifier,
    create_topic_classifier,
    create_ner_classifier,
)

__all__ = [
    # Base Classes
    "BaseClassifier",
    "Classifier",
    "TextClassifier",
    # Models
    "ClassificationResult",
    # Configuration
    "ClassifierConfig",
    "standardize_classifier_config",
    # Interfaces
    "ClassifierProtocol",
    "TextProcessor",
    # Managers
    "StateManager",
    # Strategies
    "CachingStrategy",
    # Factory Functions
    "create_classifier_by_name",
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
    # Specialized Factory Functions
    "create_sentiment_classifier",
    "create_toxicity_classifier",
    "create_spam_classifier",
    "create_profanity_classifier",
    "create_ner_classifier",
    "create_bias_detector",
    "create_language_classifier",
    "create_genre_classifier",
    "create_readability_classifier",
    "create_topic_classifier",
]
