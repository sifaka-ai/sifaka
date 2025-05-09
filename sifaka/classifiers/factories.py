"""
Classifier Factory Module

A module that provides factory functions for creating Sifaka classifiers.

## Overview
This module provides factory functions for creating various types of classifiers
in the Sifaka framework, supporting content analysis, text properties, and entity
analysis classifiers.

## Components
- create_classifier_by_name: Main factory function for creating classifiers by name
- Specialized factory functions for each classifier type

## Usage Examples
```python
from sifaka.classifiers.factories import create_classifier_by_name

# Create a sentiment classifier
sentiment = create_classifier_by_name(
    name="sentiment",
    cache_size=100,
    min_confidence=0.6
)

# Create a toxicity classifier with specific configuration
toxicity = create_classifier_by_name(
    name="toxicity",
    config={
        "cache_size": 200,
        "params": {
            "threshold": 0.7
        }
    }
)

# Create a language classifier
language = create_classifier_by_name(
    name="language",
    min_confidence=0.8
)
```

## Error Handling
The factory handles errors by:
- Validating classifier names
- Providing clear error messages for unknown types
- Supporting flexible configuration options

## Configuration
The factory supports various classifier types:
- Content analysis (bias, profanity, sentiment, spam, toxicity)
- Text properties (genre, language, readability, topic)
- Entity analysis (NER)
"""

from typing import Any, Dict, List, Optional, Union

from .config import standardize_classifier_config
from .implementations.content.bias import BiasDetector, create_bias_detector
from .implementations.content.profanity import ProfanityClassifier, create_profanity_classifier
from .implementations.content.sentiment import SentimentClassifier, create_sentiment_classifier
from .implementations.content.spam import SpamClassifier, create_spam_classifier
from .implementations.content.toxicity import ToxicityClassifier, create_toxicity_classifier
from .implementations.properties.genre import GenreClassifier, create_genre_classifier
from .implementations.properties.language import LanguageClassifier, create_language_classifier
from .implementations.properties.readability import (
    ReadabilityClassifier,
    create_readability_classifier,
)
from .implementations.properties.topic import TopicClassifier, create_topic_classifier
from .implementations.entities.ner import NERClassifier, create_ner_classifier


def create_classifier_by_name(
    name: str, config: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Any:
    """
    Create a classifier by name.

    This factory function creates a classifier instance based on the provided name,
    delegating to the appropriate specialized factory function for each classifier type.

    ## Architecture
    The factory follows a registry pattern:
    - Maintains a mapping of names to factory functions
    - Delegates creation to specialized factories
    - Supports flexible configuration options

    ## Lifecycle
    1. Name Validation: Check if the classifier type is recognized
    2. Configuration: Process and validate configuration options
    3. Creation: Delegate to specialized factory function
    4. Return: Return configured classifier instance

    ## Error Handling
    - Validates classifier names against known types
    - Provides clear error messages for unknown types
    - Supports flexible configuration options

    Args:
        name (str): Name of the classifier type to create
        config (Optional[Dict[str, Any]]): Optional configuration dictionary
        **kwargs (Any): Additional configuration options

    Returns:
        Any: A configured classifier instance

    Raises:
        ValueError: If the classifier type is not recognized

    Example:
        ```python
        from sifaka.classifiers.factories import create_classifier_by_name

        # Create a sentiment classifier
        sentiment = create_classifier_by_name(
            name="sentiment",
            cache_size=100,
            min_confidence=0.6
        )

        # Create a toxicity classifier with specific configuration
        toxicity = create_classifier_by_name(
            name="toxicity",
            config={
                "cache_size": 200,
                "params": {
                    "threshold": 0.7
                }
            }
        )

        # Create a language classifier
        language = create_classifier_by_name(
            name="language",
            min_confidence=0.8
        )
        ```
    """
    # Map of classifier names to factory functions
    factory_map = {
        # Content analysis classifiers
        "bias": create_bias_detector,
        "profanity": create_profanity_classifier,
        "sentiment": create_sentiment_classifier,
        "spam": create_spam_classifier,
        "toxicity": create_toxicity_classifier,
        # Text properties classifiers
        "genre": create_genre_classifier,
        "language": create_language_classifier,
        "readability": create_readability_classifier,
        "topic": create_topic_classifier,
        # Entity analysis classifiers
        "ner": create_ner_classifier,
    }

    # Check if the classifier name is recognized
    if name not in factory_map:
        raise ValueError(
            f"Unknown classifier type: {name}. " f"Available types: {', '.join(factory_map.keys())}"
        )

    # Call the appropriate factory function
    return factory_map[name](config=config, **kwargs)
