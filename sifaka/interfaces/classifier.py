"""
Classifier Interfaces Module

This module defines the core interfaces for classifiers in the Sifaka framework.
These interfaces establish a common contract for component behavior,
enabling better modularity, extensibility, and interoperability.

## Overview
Interfaces are a critical part of the Sifaka architecture, providing clear
contracts that components must adhere to. This module defines the interfaces
specific to the classifiers system, ensuring consistent behavior across
different classifier implementations and plugins.

## Interface Hierarchy
1. **ClassifierImplementationProtocol**: Core interface for classifier implementations
   - Defines the contract for text classification functionality
   - Ensures consistent error handling and result formatting

2. **ClassifierPluginProtocol**: Interface for classifier plugins
   - Extends the core Plugin interface with classifier-specific functionality
   - Enables discovery and registration of classifier plugins
   - Ensures consistent plugin behavior across the framework
"""

from abc import abstractmethod
from typing import Any, Dict, List, Protocol, TypeVar, Union, runtime_checkable

from sifaka.core.results import ClassificationResult
from sifaka.interfaces.core import PluginProtocol as CorePlugin
from .component import ComponentProtocol


@runtime_checkable
class ClassifierImplementationProtocol(ComponentProtocol, Protocol):
    """
    Interface for classifier implementations.

    This protocol defines the essential methods that all classifier implementations
    must provide. It establishes a common contract for classification functionality,
    ensuring consistent behavior across different implementations.

    ## Architecture
    The ClassifierImplementationProtocol protocol:
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Ensures consistent result types and error handling

    ## Lifecycle
    1. **Implementation**: Create a class that implements all required methods
    2. **Verification**: The implementation can be verified at runtime
    3. **Usage**: The implementation is used by Classifier instances
    4. **Error Handling**: Implementations should raise ImplementationError for failures
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


@runtime_checkable
class ClassifierPluginProtocol(CorePlugin, Protocol):
    """
    Interface for classifier plugins.

    This interface extends the core Plugin interface with classifier-specific
    functionality. It ensures that classifier plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.

    ## Architecture
    The ClassifierPluginProtocol protocol:
    - Extends the core Plugin interface from sifaka.core.interfaces
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Provides a consistent plugin interface across the framework
    """

    pass
