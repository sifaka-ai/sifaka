"""
Classifier Interfaces Module

This module defines the core interfaces for the Sifaka classifiers system.
These interfaces establish a common contract for component behavior,
enabling better modularity, extensibility, and interoperability.

## Overview
Interfaces are a critical part of the Sifaka architecture, providing clear
contracts that components must adhere to. This module defines the interfaces
specific to the classifiers system, ensuring consistent behavior across
different classifier implementations and plugins.

## Interface Hierarchy
1. **ClassifierImplementation**: Core interface for classifier implementations
   - Defines the contract for text classification functionality
   - Ensures consistent error handling and result formatting

2. **Plugin**: Interface for classifier plugins
   - Extends the core Plugin interface with classifier-specific functionality
   - Enables discovery and registration of classifier plugins
   - Ensures consistent plugin behavior across the framework

## Architecture
The interfaces follow a protocol-based approach using Python's typing.Protocol:
- Runtime checkable for dynamic type verification
- Abstract methods define required functionality
- Clear separation between implementation and interface

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

        text_lower = text.lower() if text else ""
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

## Error Handling
The interfaces define consistent error handling patterns:
- ImplementationError for classification failures
- Clear error propagation through the system
- Standardized error reporting in results
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

# Import ClassificationResult to fix the forward reference
from sifaka.core.results import ClassificationResult


@runtime_checkable
class ClassifierImplementation(Protocol):
    """
    Interface for classifier implementations.

    This protocol defines the essential methods that all classifier implementations
    must provide. It establishes a common contract for classification functionality,
    ensuring consistent behavior across different implementations.

    ## Architecture
    The ClassifierImplementation protocol:
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Ensures consistent result types and error handling

    ## Lifecycle
    1. **Implementation**: Create a class that implements all required methods
    2. **Verification**: The implementation can be verified at runtime
    3. **Usage**: The implementation is used by Classifier instances
    4. **Error Handling**: Implementations should raise ImplementationError for failures

    ## Examples
    ```python
    @runtime_checkable
    class ClassifierImplementation(Protocol):
        def classify(self, text: str) -> ClassificationResult:
            ...

    # Check if an object implements the protocol
    if isinstance(obj, ClassifierImplementation):
        result = obj.classify("Some text") if obj else ""
    ```
    """

    @abstractmethod
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the given text.

        This method processes the input text and returns a classification result
        with a label, confidence score, and optional metadata. It is the primary
        classification method that all implementations must provide.

        Args:
            text: The text to classify, which can be any string content
                  that the implementation can process

        Returns:
            The classification result containing:
            - label: The classification label (e.g., "positive", "toxic")
            - confidence: A confidence score between 0.0 and 1.0
            - metadata: Optional additional information about the classification
            - issues: Any issues encountered during classification
            - suggestions: Suggestions for improving the input

        Raises:
            ImplementationError: If classification fails due to implementation errors,
                                 invalid input, or other issues
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

    ## Architecture
    The Plugin protocol:
    - Extends the core Plugin interface from sifaka.core.interfaces
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Provides a consistent plugin interface across the framework

    ## Lifecycle
    1. **Implementation**: Create a plugin class that implements this interface
    2. **Registration**: Register the plugin with the plugin registry
    3. **Discovery**: The plugin is discovered by the framework
    4. **Usage**: The plugin is used by classifier components

    ## Examples
    ```python
    @runtime_checkable
    class Plugin(CorePlugin, Protocol):
        # Inherits methods from CorePlugin
        pass

    class MyClassifierPlugin(Plugin):
        def get_name(self) -> str:
            return "my_classifier_plugin"

        def get_version(self) -> str:
            return "1.0.0"

        def initialize(self) -> None:
            # Initialize plugin resources
            pass
    ```
    """

    pass
