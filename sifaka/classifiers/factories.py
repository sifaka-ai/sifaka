"""
Classifier Factories Module

This module provides factory functions for creating classifiers and components.
These factories simplify the creation of classifiers with sensible defaults.

## Factory Functions
1. **create_classifier**: Creates a classifier with the specified implementation
2. **create_implementation_adapter**: Creates an implementation adapter for existing classifiers

## Usage Examples
```python
from sifaka.classifiers.factories import create_classifier
from sifaka.classifiers.implementations.content.toxicity import ToxicityClassifier

# Create classifier implementation
implementation = ToxicityClassifier()

# Create classifier using factory
classifier = create_classifier(
    implementation=implementation,
    name="toxicity_classifier",
    description="Detects toxic content in text",
    cache_enabled=True,
    cache_size=100,
    min_confidence=0.7
)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```
"""

from typing import Any, Dict, Optional

from .classifier import Classifier
from .interfaces import ClassifierImplementation
from .config import ClassifierConfig
from .adapters import ImplementationAdapter


def create_classifier(
    implementation: Any,
    name: str = "classifier",
    description: str = "Sifaka classifier for text classification",
    cache_enabled: bool = True,
    cache_size: int = 100,
    min_confidence: float = 0.5,
    async_enabled: bool = False,
    labels: Optional[list] = None,
    **kwargs: Any,
) -> Classifier:
    """
    Create a classifier with the specified implementation.

    This factory function creates a classifier with the specified implementation,
    automatically adapting it to the required interface if needed.

    Args:
        implementation: The classifier implementation to use
        name: Classifier name
        description: Classifier description
        cache_enabled: Whether to enable result caching
        cache_size: Maximum number of cached results
        min_confidence: Minimum confidence threshold
        async_enabled: Whether to enable asynchronous execution
        labels: List of valid labels
        **kwargs: Additional parameters for the configuration

    Returns:
        A classifier instance
    """
    # Adapt implementation if needed
    adapted_implementation = implementation
    if not isinstance(implementation, ClassifierImplementation):
        adapted_implementation = ImplementationAdapter(implementation)

    # Create configuration
    config = ClassifierConfig(
        name=name,
        description=description,
        cache_enabled=cache_enabled,
        cache_size=cache_size,
        min_confidence=min_confidence,
        async_enabled=async_enabled,
        labels=labels or [],
        params=kwargs,
    )

    # Create classifier
    return Classifier(
        implementation=adapted_implementation,
        config=config,
        name=name,
        description=description,
    )
