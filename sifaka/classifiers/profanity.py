"""
Profanity Classifier Module

A module that provides profanity detection functionality for text content.

## Overview
This module implements a classifier for detecting profanity in text, providing
a standardized interface for profanity detection that can be used by rules and critics
in the Sifaka framework.

## Components
- ProfanityClassifier: Main classifier class for profanity detection
- create_profanity_classifier: Factory function for creating configured instances

## Usage Examples
```python
from sifaka.classifiers.profanity import create_profanity_classifier

# Create a profanity classifier
classifier = create_profanity_classifier(
    name="my_profanity_classifier",
    description="Custom profanity classifier",
    min_confidence=0.8,
    cache_size=200
)

# Classify text
result = classifier.classify("This is a test message")
print(f"Profanity: {result.label}, Confidence: {result.confidence}")
```

## Error Handling
The classifier handles errors by:
- Validating input text before processing
- Returning valid results even when errors occur
- Including error details in metadata when issues arise

## Configuration
The classifier can be configured with:
- min_confidence: Minimum confidence threshold (default: 0.7)
- cache_size: Size of the classification cache (default: 100)
- Additional parameters can be passed through kwargs
"""

from typing import Any, Dict, List, Optional

from .base import BaseClassifier
from .models import ClassificationResult, ClassifierConfig


class ProfanityClassifier(BaseClassifier[str, str]):
    """
    Classifier for detecting profanity in text.

    This classifier analyzes text input and determines whether it contains profanity,
    providing a confidence score and metadata about the classification.

    ## Architecture
    The classifier follows the BaseClassifier architecture:
    - Inherits from BaseClassifier[str, str]
    - Uses a simple mock implementation for demonstration
    - Supports caching and confidence thresholds

    ## Lifecycle
    1. Initialization: Configure with name, description, and optional config
    2. Warm-up: No-op in mock implementation
    3. Classification: Process text and return profanity results
    4. Result Handling: Return standardized ClassificationResult

    ## Error Handling
    - Validates input text before processing
    - Returns valid results even when errors occur
    - Includes error details in metadata when issues arise

    ## Examples
    ```python
    from sifaka.classifiers.profanity import ProfanityClassifier

    # Create classifier
    classifier = ProfanityClassifier(
        name="profanity",
        description="Detects profanity",
        config=ClassifierConfig[str](
            labels=["profane", "clean"],
            min_confidence=0.7
        )
    )

    # Classify text
    result = classifier.classify("This is a test message")
    print(f"Profanity: {result.label}, Confidence: {result.confidence}")
    ```

    Attributes:
        name (str): Name of the classifier
        description (str): Description of the classifier
        config (ClassifierConfig): Configuration for the classifier
    """

    def __init__(
        self,
        name: str = "profanity",
        description: str = "Detects profanity in text",
        config: Optional[ClassifierConfig[str]] = None,
    ):
        """
        Initialize a profanity classifier.

        Creates a new profanity classifier with the specified name, description,
        and optional configuration. If no configuration is provided, creates a
        default configuration with profane/clean labels.

        Args:
            name (str): Name of the classifier
            description (str): Description of the classifier
            config (Optional[ClassifierConfig[str]]): Optional configuration

        Returns:
            None

        Example:
            ```python
            classifier = ProfanityClassifier(
                name="my_profanity",
                description="Custom profanity detector",
                config=ClassifierConfig[str](
                    labels=["profane", "clean"],
                    min_confidence=0.8
                )
            )
            ```
        """
        if config is None:
            config = ClassifierConfig[str](
                labels=["profane", "clean"],
                min_confidence=0.7,
                cache_size=100,
            )
        super().__init__(name=name, description=description, config=config)

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Classify text as profane or clean.

        Implements the core classification logic for determining if text contains
        profanity. This is a mock implementation that always returns "clean"
        with high confidence.

        Args:
            text (str): The text to classify

        Returns:
            ClassificationResult[str]: Classification result containing:
                - label: Either "profane" or "clean"
                - confidence: Confidence score between 0 and 1
                - metadata: Additional information about the classification

        Example:
            ```python
            result = classifier._classify_impl_uncached("This is a test message")
            print(f"Label: {result.label}, Confidence: {result.confidence}")
            ```
        """
        # This is a mock implementation
        return ClassificationResult[str](
            label="clean",
            confidence=0.9,
            metadata={"mock": True},
        )

    def warm_up(self) -> None:
        """
        Warm up the classifier.

        Prepares the classifier for use by loading any necessary resources.
        This is a no-op in the mock implementation, but real implementations
        would use this to load models or other resources.

        Args:
            None

        Returns:
            None

        Example:
            ```python
            classifier = ProfanityClassifier()
            classifier.warm_up()  # Prepare resources
            ```
        """
        pass


def create_profanity_classifier(
    name: str = "profanity",
    description: str = "Detects profanity in text",
    min_confidence: float = 0.7,
    cache_size: int = 100,
    **kwargs: Any,
) -> ProfanityClassifier:
    """
    Create a profanity classifier.

    Factory function that creates and configures a new ProfanityClassifier instance
    with the specified parameters.

    Args:
        name (str): Name of the classifier
        description (str): Description of the classifier
        min_confidence (float): Minimum confidence threshold
        cache_size (int): Size of the classification cache
        **kwargs (Any): Additional parameters to pass to the classifier

    Returns:
        ProfanityClassifier: Configured classifier instance

    Example:
        ```python
        classifier = create_profanity_classifier(
            name="my_profanity",
            description="Custom profanity detector",
            min_confidence=0.8,
            cache_size=200
        )
        ```
    """
    config = ClassifierConfig[str](
        labels=["profane", "clean"],
        min_confidence=min_confidence,
        cache_size=cache_size,
        params=kwargs,
    )
    return ProfanityClassifier(name=name, description=description, config=config)
