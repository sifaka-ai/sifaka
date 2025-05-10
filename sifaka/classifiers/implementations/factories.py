"""
Classifier Implementation Factories

This module provides factory functions for creating classifiers with existing implementations.
These factories simplify the creation of classifiers with the new system.

## Usage Examples
```python
from sifaka.classifiers.implementations.factories import create_toxicity_classifier

# Create toxicity classifier
classifier = create_toxicity_classifier(
    general_threshold=0.5,
    cache_enabled=True,
    cache_size=100
)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```
"""

from typing import Any, Dict, List, Optional

from .. import Classifier
from ..factories import create_classifier
from .adapters import (
    ToxicityClassifierAdapter,
    SentimentClassifierAdapter,
    ProfanityClassifierAdapter,
)


def create_toxicity_classifier(
    name: str = "toxicity_classifier",
    description: str = "Detects toxic content using Detoxify",
    general_threshold: float = 0.5,
    severe_toxic_threshold: float = 0.7,
    threat_threshold: float = 0.7,
    cache_enabled: bool = True,
    cache_size: int = 100,
    min_confidence: float = 0.0,
    **kwargs: Any,
) -> Classifier:
    """
    Create a toxicity classifier.
    
    Args:
        name: Classifier name
        description: Classifier description
        general_threshold: Threshold for general toxicity
        severe_toxic_threshold: Threshold for severe toxicity
        threat_threshold: Threshold for threats
        cache_enabled: Whether to enable result caching
        cache_size: Maximum number of cached results
        min_confidence: Minimum confidence threshold
        **kwargs: Additional parameters
        
    Returns:
        A toxicity classifier
    """
    # Create implementation adapter
    implementation = ToxicityClassifierAdapter(
        name=name,
        description=description,
        general_threshold=general_threshold,
        severe_toxic_threshold=severe_toxic_threshold,
        threat_threshold=threat_threshold,
        **kwargs
    )
    
    # Create classifier
    return create_classifier(
        implementation=implementation,
        name=name,
        description=description,
        cache_enabled=cache_enabled,
        cache_size=cache_size,
        min_confidence=min_confidence,
        labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_toxic"],
        **kwargs
    )


def create_sentiment_classifier(
    name: str = "sentiment_classifier",
    description: str = "Classifies text sentiment using VADER",
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
    cache_enabled: bool = True,
    cache_size: int = 100,
    min_confidence: float = 0.0,
    **kwargs: Any,
) -> Classifier:
    """
    Create a sentiment classifier.
    
    Args:
        name: Classifier name
        description: Classifier description
        positive_threshold: Threshold for positive sentiment
        negative_threshold: Threshold for negative sentiment
        cache_enabled: Whether to enable result caching
        cache_size: Maximum number of cached results
        min_confidence: Minimum confidence threshold
        **kwargs: Additional parameters
        
    Returns:
        A sentiment classifier
    """
    # Create implementation adapter
    implementation = SentimentClassifierAdapter(
        name=name,
        description=description,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        **kwargs
    )
    
    # Create classifier
    return create_classifier(
        implementation=implementation,
        name=name,
        description=description,
        cache_enabled=cache_enabled,
        cache_size=cache_size,
        min_confidence=min_confidence,
        labels=["positive", "negative", "neutral", "unknown"],
        **kwargs
    )


def create_profanity_classifier(
    name: str = "profanity_classifier",
    description: str = "Detects profanity in text",
    threshold: float = 0.5,
    cache_enabled: bool = True,
    cache_size: int = 100,
    min_confidence: float = 0.0,
    **kwargs: Any,
) -> Classifier:
    """
    Create a profanity classifier.
    
    Args:
        name: Classifier name
        description: Classifier description
        threshold: Threshold for profanity detection
        cache_enabled: Whether to enable result caching
        cache_size: Maximum number of cached results
        min_confidence: Minimum confidence threshold
        **kwargs: Additional parameters
        
    Returns:
        A profanity classifier
    """
    # Create implementation adapter
    implementation = ProfanityClassifierAdapter(
        name=name,
        description=description,
        threshold=threshold,
        **kwargs
    )
    
    # Create classifier
    return create_classifier(
        implementation=implementation,
        name=name,
        description=description,
        cache_enabled=cache_enabled,
        cache_size=cache_size,
        min_confidence=min_confidence,
        labels=["profane", "clean"],
        **kwargs
    )


# Add more factory functions for other classifier implementations as needed
