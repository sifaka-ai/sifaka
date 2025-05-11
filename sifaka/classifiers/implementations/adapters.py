"""
Classifier Implementation Adapters

This module provides adapter classes for existing classifier implementations to work with the new
classifiers system. These adapters implement the ClassifierImplementation interface while
delegating to the existing classifier implementations.

## Usage Examples
```python
from sifaka.classifiers import Classifier
from sifaka.classifiers.implementations.adapters import ToxicityClassifierAdapter

# Create adapter
implementation = ToxicityClassifierAdapter()

# Create classifier
classifier = Classifier(implementation=implementation)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..interfaces import ClassifierImplementation
from ..result import ClassificationResult
from sifaka.classifiers.errors import ImplementationError, safely_execute

# Import existing implementations
from .content.toxicity import ToxicityClassifier as OldToxicityClassifier
from .content.sentiment import SentimentClassifier as OldSentimentClassifier
from .content.profanity import ProfanityClassifier as OldProfanityClassifier

# Uncomment these imports when the corresponding classes are implemented
# from .content.spam import SpamClassifier as OldSpamClassifier
# from .content.bias import BiasClassifier as OldBiasClassifier
# from .properties.language import LanguageClassifier as OldLanguageClassifier
# from .properties.readability import ReadabilityClassifier as OldReadabilityClassifier
# from .properties.genre import GenreClassifier as OldGenreClassifier
# from .properties.topic import TopicClassifier as OldTopicClassifier
# from .entities.ner import NERClassifier as OldNERClassifier


class ToxicityClassifierAdapter(ClassifierImplementation):
    """Adapter for the ToxicityClassifier."""

    def __init__(self, **kwargs):
        """
        Initialize the adapter.

        Args:
            **kwargs: Arguments to pass to the ToxicityClassifier constructor
        """
        self._classifier = OldToxicityClassifier.create(
            name=kwargs.get("name", "toxicity_classifier"),
            description=kwargs.get("description", "Detects toxic content using Detoxify"),
            labels=kwargs.get(
                "labels",
                [
                    "toxic",
                    "severe_toxic",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_hate",
                    "non_toxic",
                ],
            ),
            **kwargs,
        )

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """

        def classify_operation():
            return self._classifier.classify(text)

        result = safely_execute(
            operation=classify_operation,
            component_name="toxicity_classifier_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )

        # Convert result to ClassificationResult
        return ClassificationResult(
            label=result.label,
            confidence=result.confidence,
            metadata=result.metadata,
            issues=getattr(result, "issues", []),
            suggestions=getattr(result, "suggestions", []),
        )

    async def classify_async(self, text: str) -> ClassificationResult:
        """
        Classify the given text asynchronously.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """
        # Fall back to running synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, text)


class SentimentClassifierAdapter(ClassifierImplementation):
    """Adapter for the SentimentClassifier."""

    def __init__(self, **kwargs):
        """
        Initialize the adapter.

        Args:
            **kwargs: Arguments to pass to the SentimentClassifier constructor
        """
        self._classifier = OldSentimentClassifier.create(
            name=kwargs.get("name", "sentiment_classifier"),
            description=kwargs.get("description", "Classifies text sentiment using VADER"),
            labels=kwargs.get("labels", ["positive", "negative", "neutral", "unknown"]),
            **kwargs,
        )

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """

        def classify_operation():
            return self._classifier.classify(text)

        result = safely_execute(
            operation=classify_operation,
            component_name="sentiment_classifier_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )

        # Convert result to ClassificationResult
        return ClassificationResult(
            label=result.label,
            confidence=result.confidence,
            metadata=result.metadata,
            issues=getattr(result, "issues", []),
            suggestions=getattr(result, "suggestions", []),
        )

    async def classify_async(self, text: str) -> ClassificationResult:
        """
        Classify the given text asynchronously.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """
        # Fall back to running synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, text)


class ProfanityClassifierAdapter(ClassifierImplementation):
    """Adapter for the ProfanityClassifier."""

    def __init__(self, **kwargs):
        """
        Initialize the adapter.

        Args:
            **kwargs: Arguments to pass to the ProfanityClassifier constructor
        """
        self._classifier = OldProfanityClassifier.create(
            name=kwargs.get("name", "profanity_classifier"),
            description=kwargs.get("description", "Detects profanity in text"),
            labels=kwargs.get("labels", ["profane", "clean"]),
            **kwargs,
        )

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """

        def classify_operation():
            return self._classifier.classify(text)

        result = safely_execute(
            operation=classify_operation,
            component_name="profanity_classifier_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )

        # Convert result to ClassificationResult
        return ClassificationResult(
            label=result.label,
            confidence=result.confidence,
            metadata=result.metadata,
            issues=getattr(result, "issues", []),
            suggestions=getattr(result, "suggestions", []),
        )

    async def classify_async(self, text: str) -> ClassificationResult:
        """
        Classify the given text asynchronously.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """
        # Fall back to running synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, text)


# Add more adapters for other classifier implementations as needed
