"""
Factory functions for classifiers.

This module provides factory functions for creating classifiers in the Sifaka framework.
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
from .implementations.properties.readability import ReadabilityClassifier, create_readability_classifier
from .implementations.properties.topic import TopicClassifier, create_topic_classifier
from .implementations.entities.ner import NERClassifier, create_ner_classifier


def create_classifier_by_name(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Any:
    """
    Create a classifier by name.

    This factory function creates a classifier instance based on the provided name.
    It delegates to the appropriate specialized factory function for each classifier type.

    Args:
        name: Name of the classifier type to create
        config: Optional configuration dictionary
        **kwargs: Additional configuration options

    Returns:
        A classifier instance

    Raises:
        ValueError: If the classifier type is not recognized

    Examples:
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
            f"Unknown classifier type: {name}. "
            f"Available types: {', '.join(factory_map.keys())}"
        )

    # Call the appropriate factory function
    return factory_map[name](config=config, **kwargs)
