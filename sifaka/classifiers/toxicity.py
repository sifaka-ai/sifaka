"""
Toxicity Classifier Module

A module that provides toxicity classification functionality for text content.

## Overview
This module implements a classifier for detecting toxic content in text, providing
a standardized interface for toxicity detection that can be used by rules and critics
in the Sifaka framework.

## Components
- ToxicityClassifier: Main classifier class for toxicity detection
- create_toxicity_classifier: Factory function for creating configured instances

## Usage Examples
```python
from sifaka.classifiers.toxicity import create_toxicity_classifier

# Create a toxicity classifier
classifier = create_toxicity_classifier(
    name="my_toxicity_classifier",
    description="Custom toxicity classifier",
    min_confidence=0.8,
    cache_size=200
)

# Classify text
result = classifier.classify("This is a test message")
print(f"Toxicity: {result.label}, Confidence: {result.confidence}")
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


class ToxicityClassifier(BaseClassifier[str, str]):
    """
    Classifier for detecting toxic content in text.

    This classifier analyzes text input and determines whether it contains toxic content,
    providing a confidence score and metadata about the classification.

    ## Architecture
    The classifier follows the BaseClassifier architecture:
    - Inherits from BaseClassifier[str, str]
    - Uses a simple mock implementation for demonstration
    - Supports caching and confidence thresholds

    ## Lifecycle
    1. Initialization: Configure with name, description, and optional config
    2. Warm-up: No-op in mock implementation
    3. Classification: Process text and return toxicity results
    4. Result Handling: Return standardized ClassificationResult

    ## Error Handling
    - Validates input text before processing
    - Returns valid results even when errors occur
    - Includes error details in metadata when issues arise

    ## Examples
    ```python
    from sifaka.classifiers.toxicity import ToxicityClassifier

    # Create classifier
    classifier = ToxicityClassifier(
        name="toxicity",
        description="Detects toxic content",
        config=ClassifierConfig[str](
            labels=["toxic", "non_toxic"],
            min_confidence=0.7
        )
    )

    # Classify text
    result = classifier.classify("This is a test message")
    print(f"Toxicity: {result.label}, Confidence: {result.confidence}")
    ```

    Attributes:
        name (str): Name of the classifier
        description (str): Description of the classifier
        config (ClassifierConfig): Configuration for the classifier
    """

    def __init__(
        self,
        name: str = "toxicity",
        description: str = "Detects toxic content in text",
        config: Optional[ClassifierConfig[str]] = None,
    ):
        """
        Initialize a toxicity classifier.

        Creates a new toxicity classifier with the specified name, description,
        and optional configuration. If no configuration is provided, creates a
        default configuration with toxic/non-toxic labels.

        Args:
            name (str): Name of the classifier
            description (str): Description of the classifier
            config (Optional[ClassifierConfig[str]]): Optional configuration

        Returns:
            None

        Example:
            ```python
            classifier = ToxicityClassifier(
                name="my_toxicity",
                description="Custom toxicity detector",
                config=ClassifierConfig[str](
                    labels=["toxic", "non_toxic"],
                    min_confidence=0.8
                )
            )
            ```
        """
        if config is None:
            config = ClassifierConfig[str](
                labels=["toxic", "non_toxic"],
                min_confidence=0.7,
                cache_size=100,
            )
        super().__init__(name=name, description=description, config=config)

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Classify text as toxic or non-toxic.

        Implements the core classification logic for determining if text contains
        toxic content. This is a mock implementation that always returns "non_toxic"
        with high confidence.

        Args:
            text (str): The text to classify

        Returns:
            ClassificationResult[str]: Classification result containing:
                - label: Either "toxic" or "non_toxic"
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
            label="non_toxic",
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
            classifier = ToxicityClassifier()
            classifier.warm_up()  # Prepare resources
            ```
        """
        pass


def create_toxicity_classifier(
    name: str = "toxicity",
    description: str = "Detects toxic content in text",
    min_confidence: float = 0.7,
    cache_size: int = 100,
    **kwargs: Any,
) -> ToxicityClassifier:
    """
    Create a toxicity classifier.

    Factory function that creates and configures a new ToxicityClassifier instance
    with the specified parameters.

    Args:
        name (str): Name of the classifier
        description (str): Description of the classifier
        min_confidence (float): Minimum confidence threshold
        cache_size (int): Size of the classification cache
        **kwargs (Any): Additional parameters to pass to the classifier

    Returns:
        ToxicityClassifier: Configured classifier instance

    Example:
        ```python
        classifier = create_toxicity_classifier(
            name="my_toxicity",
            description="Custom toxicity detector",
            min_confidence=0.8,
            cache_size=200
        )
        ```
    """
    config = ClassifierConfig[str](
        labels=["toxic", "non_toxic"],
        min_confidence=min_confidence,
        cache_size=cache_size,
        params=kwargs,
    )
    return ToxicityClassifier(name=name, description=description, config=config)
