"""
Sifaka Classifiers Package

This package provides a collection of text classifiers that analyze content for various
characteristics. Each classifier follows a consistent interface and can be used
independently or integrated with rules.

Architecture Overview:
- Each classifier inherits from the base Classifier class
- Classifiers return ClassificationResult objects with consistent structure
- Classifiers can be combined with rules using ClassifierRule

Available Classifiers:
1. Base Classes:
   - Classifier: Abstract base class for all classifiers
   - ClassificationResult: Standard result format
   - ClassifierConfig: Configuration for classifiers
   - ClassifierProtocol: Protocol for classifier implementations
   - TextProcessor: Utility for text preprocessing

2. Content Analysis:
   - SentimentClassifier: Analyzes text sentiment (positive/negative/neutral)
   - ProfanityClassifier: Detects profane or inappropriate language
   - ToxicityClassifier: Identifies toxic content

3. Text Properties:
   - ReadabilityClassifier: Evaluates reading difficulty level
   - LanguageClassifier: Identifies the language of text

4. Model-Based:
   - LLMClassifier: Uses LLMs for customizable classification tasks

Usage Example:
    from sifaka.classifiers import SentimentClassifier, ReadabilityClassifier

    # Create classifiers
    sentiment = SentimentClassifier()
    readability = ReadabilityClassifier()

    # Analyze text
    sentiment_result = sentiment.classify("This is fantastic!")
    readability_result = readability.classify("The quantum mechanical model...")

    # Use with rules
    from sifaka.rules import ClassifierRule
    sentiment_rule = ClassifierRule(classifier=sentiment)
"""

from .base import (
    Classifier,
    ClassificationResult,
    ClassifierConfig,
    ClassifierProtocol,
    TextProcessor,
)
from .sentiment import SentimentClassifier
from .profanity import ProfanityClassifier
from .readability import ReadabilityClassifier
from .toxicity import ToxicityClassifier
from .language import LanguageClassifier
from .llm import LLMClassifier

__all__ = [
    # Base
    "Classifier",
    "ClassificationResult",
    "ClassifierConfig",
    "ClassifierProtocol",
    "TextProcessor",
    # Content Analysis
    "SentimentClassifier",
    "ProfanityClassifier",
    "ToxicityClassifier",
    # Text Properties
    "ReadabilityClassifier",
    "LanguageClassifier",
    # Model-Based
    "LLMClassifier",
]
