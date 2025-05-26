"""Classifiers for Sifaka.

This module provides text classification capabilities for the Sifaka framework.
Classifiers can be used standalone for text analysis or integrated with validators
for content validation.

Available classifiers:
- BiasClassifier: Detects bias in text using machine learning
- LanguageClassifier: Detects the language of text
- ProfanityClassifier: Detects profanity and inappropriate language
- SentimentClassifier: Analyzes sentiment (positive/negative/neutral)
- SpamClassifier: Detects spam content
- ToxicityClassifier: Detects toxic language

Example:
    ```python
    from sifaka.classifiers import BiasClassifier, create_bias_validator

    # Use classifier standalone
    classifier = BiasClassifier()
    result = classifier.classify("This text might contain bias.")
    print(f"Label: {result.label}, Confidence: {result.confidence}")

    # Use with validator
    validator = create_bias_validator(threshold=0.7)
    validation_result = validator.validate(thought)
    ```
"""

from typing import List

# Import base classes and protocols
from sifaka.classifiers.base import CachedTextClassifier, ClassificationResult, TextClassifier

# Import classifier implementations
__all__: List[str] = [
    "ClassificationResult",
    "TextClassifier",
    "CachedTextClassifier",
]

# Import classifiers - these should always work since they have fallbacks
from sifaka.classifiers.bias import (
    BiasClassifier,
    CachedBiasClassifier,
    create_bias_validator,
    create_cached_bias_validator,
)
from sifaka.classifiers.language import LanguageClassifier, create_language_validator
from sifaka.classifiers.profanity import ProfanityClassifier, create_profanity_validator
from sifaka.classifiers.sentiment import (
    CachedSentimentClassifier,
    SentimentClassifier,
    create_cached_sentiment_validator,
    create_sentiment_validator,
)
from sifaka.classifiers.spam import SpamClassifier, create_spam_validator
from sifaka.classifiers.toxicity import (
    CachedToxicityClassifier,
    ToxicityClassifier,
    create_cached_toxicity_validator,
    create_toxicity_validator,
)

__all__.extend(
    [
        "BiasClassifier",
        "create_bias_validator",
        "CachedBiasClassifier",
        "create_cached_bias_validator",
        "LanguageClassifier",
        "create_language_validator",
        "ProfanityClassifier",
        "create_profanity_validator",
        "SentimentClassifier",
        "create_sentiment_validator",
        "CachedSentimentClassifier",
        "create_cached_sentiment_validator",
        "SpamClassifier",
        "create_spam_validator",
        "ToxicityClassifier",
        "create_toxicity_validator",
        "CachedToxicityClassifier",
        "create_cached_toxicity_validator",
    ]
)
