"""
Classifier Factories Module

This module provides factory functions for creating classifier instances.
These factories handle the creation and configuration of different types
of classifiers, ensuring consistent initialization and setup.

## Overview
The factory functions provide a convenient way to create and configure
classifiers, handling the details of initialization, configuration,
and setup. They ensure that classifiers are created with the correct
settings and dependencies.

## Factory Functions
1. **create_classifier**: Creates a classifier with the given configuration
2. **create_implementation**: Creates a classifier implementation
3. **create_adapter**: Creates a classifier adapter

## Usage Examples
```python
from sifaka.classifiers.factories import create_classifier
from sifaka.utils.config.classifiers import ClassifierConfig

# Create classifier with configuration
classifier = create_classifier(
    name="sentiment_classifier",
    description="Detects sentiment in text",
    labels=["positive", "negative", "neutral"],
    cache_size=100,
    min_confidence=0.7
)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```

## Error Handling
The factory functions provide robust error handling:
- ClassifierError: Raised when classifier creation fails
- ConfigurationError: Raised for invalid configuration
- ImplementationError: Raised when implementation creation fails

## Configuration
The factory functions support configuration through the ClassifierConfig class:
- cache_enabled: Whether to enable result caching
- cache_size: Maximum number of cached results
- min_confidence: Minimum confidence threshold
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, cast
from .interfaces import ClassifierImplementation
from .classifier import Classifier
from .adapters import ImplementationAdapter
from ..utils.config import ClassifierConfig
from ..utils.errors import ClassifierError

# Define type variables
L = TypeVar("L")  # Label type
M = TypeVar("M", bound=Dict[str, Any])  # Metadata type


def create_classifier(
    name: str,
    description: str,
    implementation: ClassifierImplementation,
    cache_size: int = 100,
    min_confidence: float = 0.7,
) -> Classifier:
    """
    Create a classifier with the given configuration.

    This function creates a Classifier instance with the specified configuration
    and implementation. It handles the initialization and setup of the
    classifier, ensuring consistent behavior.

    Args:
        name: The name of the classifier
        description: A description of the classifier's purpose
        implementation: The classifier implementation to use
        cache_size: Maximum number of cached results
        min_confidence: Minimum confidence threshold (stored in implementation)

    Returns:
        A configured Classifier instance

    Raises:
        ClassifierError: If classifier creation fails
        ConfigurationError: If configuration is invalid
    """
    config: ClassifierConfig = ClassifierConfig(
        cache_enabled=True,
        cache_size=cache_size,
    )

    # Configure the implementation with min_confidence if it supports it
    if hasattr(implementation, "min_confidence"):
        implementation.min_confidence = min_confidence

    return Classifier(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )


def create_implementation(
    implementation_type: Type[ClassifierImplementation],
    **kwargs: Any,
) -> ClassifierImplementation:
    """
    Create a classifier implementation.

    This function creates an instance of the specified implementation type
    with the given configuration. It handles the initialization and setup
    of the implementation, ensuring consistent behavior.

    Args:
        implementation_type: The type of implementation to create
        **kwargs: Additional arguments for the implementation

    Returns:
        A configured ClassifierImplementation instance

    Raises:
        ImplementationError: If implementation creation fails
        ConfigurationError: If configuration is invalid
    """
    try:
        return implementation_type(**kwargs)
    except Exception as e:
        raise ClassifierError(f"Failed to create implementation: {str(e)}") from e


def create_adapter(
    implementation: Any,
    adapter_type: Type[ImplementationAdapter] = ImplementationAdapter,
) -> ImplementationAdapter:
    """
    Create a classifier adapter.

    This function creates an adapter for the given implementation, allowing
    it to be used with the Sifaka classifiers system. It handles the initialization
    and setup of the adapter, ensuring consistent behavior.

    Args:
        implementation: The implementation to adapt
        adapter_type: The type of adapter to create

    Returns:
        A configured ImplementationAdapter instance

    Raises:
        ImplementationError: If adapter creation fails
    """
    try:
        return adapter_type(implementation)
    except Exception as e:
        raise ClassifierError(f"Failed to create adapter: {str(e)}") from e
