"""
Classifier Factories Module

This module provides factory functions for creating classifier instances
with various configurations and implementations.

## Overview
Factory functions simplify the creation of classifier instances by providing
sensible defaults, validating configurations, and ensuring proper initialization.
They abstract away the complexity of classifier construction and provide a
consistent interface for creating different types of classifiers.

## Factory Functions
1. **create_classifier**: Create a classifier with a custom implementation
   - Configures caching, confidence threshold, and other settings
   - Initializes the classifier with proper state management
   - Returns a ready-to-use classifier instance

## Usage Examples
```python
from sifaka.classifiers.factories import create_classifier
from sifaka.classifiers.implementations.content import ToxicityClassifier

# Create a toxicity classifier with default configuration
classifier = create_classifier(
    name="content_moderation",
    description="Moderates content for toxicity",
    implementation=ToxicityClassifier()
)

# Use the classifier
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```

## Configuration Options
The factory functions support various configuration options:
- **cache_size**: Maximum number of cached results
- **min_confidence**: Minimum confidence threshold
- **name**: Name of the classifier
- **description**: Description of the classifier
"""

from typing import Any, Dict, Optional, TypeVar, Type
from .classifier import Classifier
from .adapters import ImplementationAdapter
from ..utils.config import ClassifierConfig
from sifaka.interfaces.classifier import (
    ClassifierImplementationProtocol as ClassifierImplementation,
)
from ..utils.errors import ClassifierError
from sifaka.core.registry import register_classifier_factory

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


# This function is registered with the registry in the classifiers implementation module
# to enable registry-based classifier creation
register_classifier_factory("base", create_classifier)


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
