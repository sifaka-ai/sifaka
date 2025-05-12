from typing import Any, List
"""
Classifiers Module

## Overview
This module provides a simplified and more maintainable implementation of the classifiers system
for categorizing and analyzing text. It serves as a central component for Sifaka's text analysis
capabilities, implementing a flexible and extensible classification system with a cleaner architecture.

## Components
1. **Classifier**: Main user-facing class for classification
2. **Engine**: Core classification engine that coordinates the flow
3. **ClassifierImplementation**: Interface for classifier implementations
4. **ClassificationResult**: Result of a classification operation
5. **Plugin**: Interface for plugins

## Usage Examples
```python
from sifaka.classifiers import create_toxicity_classifier, create_sentiment_classifier

# Create toxicity classifier
toxicity_classifier = create_toxicity_classifier(
    general_threshold=0.5,
    cache_enabled=True,
    cache_size=100
)

# Classify text
result = (toxicity_classifier and toxicity_classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")

# Create sentiment classifier
sentiment_classifier = create_sentiment_classifier(
    positive_threshold=0.05,
    negative_threshold=-0.05,
    cache_enabled=True,
    cache_size=100
)

# Classify text
result = (sentiment_classifier and sentiment_classifier.classify("I love this product!")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```

## Error Handling
- ClassifierError: Raised when classification fails
- ImplementationError: Raised when implementation fails
- ConfigError: Raised when configuration is invalid
- StateError: Raised when state operations fail

## Configuration
- cache_enabled: Whether to enable result caching
- cache_size: Maximum number of cached results
- min_confidence: Minimum confidence threshold
"""
from .classifier import Classifier
from .engine import Engine
from ..core.results import ClassificationResult
from .interfaces import ClassifierImplementation, Plugin
from .factories import create_classifier
from .implementations.factories import create_toxicity_classifier, create_sentiment_classifier, create_profanity_classifier
from ..utils.state import StateManager, create_classifier_state
__all__: List[Any] = ['Classifier', 'Engine', 'ClassificationResult',
    'ClassifierImplementation', 'Plugin', 'create_classifier',
    'create_toxicity_classifier', 'create_sentiment_classifier',
    'create_profanity_classifier', 'StateManager', 'create_classifier_state']
