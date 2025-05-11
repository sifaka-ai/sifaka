"""
Classifier Interfaces Module

This module defines the core interfaces for the Sifaka classifiers system.
These interfaces establish a common contract for component behavior,
enabling better modularity, extensibility, and interoperability.

## Interface Hierarchy

1. **ClassifierImplementation**: Interface for classifier implementations
2. **Plugin**: Interface for plugins

## Usage Examples

```python
from typing import Dict, Any
from sifaka.classifiers.interfaces import ClassifierImplementation
from sifaka.classifiers.result import ClassificationResult

class SentimentClassifier(ClassifierImplementation):
    def classify(self, text: str) -> ClassificationResult:
        # Simple implementation that checks for positive/negative words
        positive_words = ["good", "great", "excellent", "happy"]
        negative_words = ["bad", "terrible", "awful", "sad"]

        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)

        if positive_count > negative_count:
            return ClassificationResult(
                label="positive",
                confidence=0.8,
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )
        elif negative_count > positive_count:
            return ClassificationResult(
                label="negative",
                confidence=0.8,
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )
        else:
            return ClassificationResult(
                label="neutral",
                confidence=0.6,
                metadata={"positive_words": positive_count, "negative_words": negative_count}
            )
```
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class ClassifierImplementation(Protocol):
    """Interface for classifier implementations."""

    @abstractmethod
    def classify(self, text: str) -> "ClassificationResult":
        """
        Classify the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        pass

    @abstractmethod
    async def classify_async(self, text: str) -> "ClassificationResult":
        """
        Classify the given text asynchronously.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        pass


# Import the core Plugin interface
from sifaka.core.interfaces import Plugin as CorePlugin


@runtime_checkable
class Plugin(CorePlugin, Protocol):
    """
    Interface for classifier plugins.

    This interface extends the core Plugin interface with classifier-specific
    functionality. It ensures that classifier plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """

    pass
