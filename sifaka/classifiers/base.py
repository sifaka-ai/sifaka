"""
Base Classes for Sifaka Classifiers

This module provides the foundational components for the Sifaka classifiers system,
including base classes, protocols, and utility functions for text classification.

## Overview
The classifier base module serves as the foundation for all text classification
components in Sifaka. It defines abstract base classes and utility functions that
standardize how classifiers are implemented, configured, and used throughout the
system. This ensures consistent behavior and error handling across different
classifier implementations.

## Components
1. **BaseClassifier**: Abstract base class for all classifiers
2. **BaseClassifierImplementation**: Abstract base class for classifier implementations
3. **create_base_classification_result**: Utility function for creating standardized results
4. **safely_classify**: Utility function for safe classification with error handling

## Architecture
The classifier system follows a layered architecture:
1. **User Interface Layer**: Classifier class with public methods
2. **Base Layer**: BaseClassifier and BaseClassifierImplementation abstract classes
3. **Implementation Layer**: Concrete classifier implementations
4. **State Management Layer**: StateManager for tracking state and statistics

## Usage Examples
```python
from sifaka.classifiers.base import BaseClassifierImplementation
from sifaka.classifiers.result import ClassificationResult

class MyClassifierImplementation(BaseClassifierImplementation):
    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        # Implement classification logic
        return ClassificationResult(
            label="example",
            confidence=0.8,
            metadata={"source": "custom_implementation"}
        )

    async def _classify_async_impl_uncached(self, text: str) -> ClassificationResult:
        # Implement async classification logic
        return ClassificationResult(
            label="example",
            confidence=0.8,
            metadata={"source": "custom_implementation"}
        )
```

## Error Handling
The module provides standardized error handling through:
- ClassifierError: Base exception for all classifier errors
- ImplementationError: Raised when implementation-specific errors occur
- safely_classify: Utility function for safe classification with error handling
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, Union, runtime_checkable

from pydantic import BaseModel, Field

from sifaka.core.base import BaseComponent
from sifaka.utils.state import StateManager, create_classifier_state
from sifaka.utils.errors.component import ClassifierError, ImplementationError
from sifaka.utils.common import update_statistics
from sifaka.utils.logging import get_logger
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig

# Configure logger
logger = get_logger(__name__)

# Type variables
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
L = TypeVar("L")  # Label type


class BaseClassifier(BaseComponent[T, ClassificationResult], Generic[T, L]):
    """
    Abstract base class for all classifiers.

    This class provides a foundation for implementing classifiers that can
    categorize text into predefined labels. It implements common functionality
    and enforces a consistent interface for all classifier implementations.

    ## Architecture
    The BaseClassifier class:
    - Extends BaseComponent for consistent component interfaces
    - Uses StateManager for state tracking and statistics
    - Provides abstract methods for classifier-specific functionality
    - Implements common error handling and caching patterns

    ## Lifecycle
    1. **Initialization**: Set up classifier with configuration
    2. **Classification**: Process text through the implementation
    3. **Result Handling**: Return standardized ClassificationResult
    4. **State Management**: Track statistics and cache results
    5. **Error Handling**: Handle and track errors

    ## Examples
    ```python
    from sifaka.classifiers.base import BaseClassifier
    from sifaka.classifiers.result import ClassificationResult

    class MyClassifier(BaseClassifier[str, str]):
        def _classify_impl(self, text: str) -> ClassificationResult:
            # Implement classification logic
            return ClassificationResult(
                label="example",
                confidence=0.8,
                metadata={"source": "custom_classifier"}
            )
    ```
    """

    def __init__(
        self,
        config: Optional[Optional[ClassifierConfig]] = None,
        name: str = "classifier",
        description: str = "Sifaka classifier for text classification",
    ):
        """
        Initialize the base classifier.

        Args:
            config: Classifier configuration
            name: Classifier name
            description: Classifier description
        """
        self._name = name
        self._description = description
        self._config = config or ClassifierConfig()

        # Create state manager using the standardized state management
        self._state_manager = create_classifier_state()

        # Initialize state
        if self._state_manager:
            self._state_manager.update("name", name)
            self._state_manager.update("description", description)
            self._state_manager.update("config", self._config)
            self._state_manager.update("initialized", True)
            self._state_manager.update("execution_count", 0)
            self._state_manager.update("result_cache", {})

        # Set metadata
        if self._state_manager:
            self._state_manager.set_metadata("component_type", "classifier")
            self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """
        Get classifier name.

        Returns:
            The name of the classifier
        """
        return self._name

    @property
    def description(self) -> str:
        """
        Get classifier description.

        Returns:
            The description of the classifier
        """
        return self._description

    @property
    def config(self) -> ClassifierConfig:
        """
        Get classifier configuration.

        Returns:
            The current configuration of the classifier
        """
        return self._config

    def update_config(self, config: ClassifierConfig) -> None:
        """
        Update classifier configuration.

        This method updates the classifier's configuration and ensures the state
        manager is updated with the new configuration.

        Args:
            config: New classifier configuration
        """
        self._config = config
        if self._state_manager:
            self._state_manager.update("config", config)

    @abstractmethod
    def classify(self, text: T) -> ClassificationResult[L, Any]:
        """
        Classify the given text.

        This method processes the input text and returns a classification result.
        It must be implemented by subclasses to provide specific classification logic.

        Args:
            text: The text to classify

        Returns:
            The classification result with label, confidence, and metadata

        Raises:
            ClassifierError: If classification fails
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier statistics.

        This method returns a comprehensive dictionary of classifier statistics,
        including execution counts, success/failure rates, timing information,
        and error details.

        Returns:
            Dictionary with classifier statistics
        """
        return {
            "name": self._name,
            "execution_count": (
                self._state_manager.get("execution_count", 0) if self._state_manager else 0
            ),
            "success_count": (
                self._state_manager.get_metadata("success_count", 0) if self._state_manager else 0
            ),
            "failure_count": (
                self._state_manager.get_metadata("failure_count", 0) if self._state_manager else 0
            ),
            "error_count": (
                self._state_manager.get_metadata("error_count", 0) if self._state_manager else 0
            ),
            "avg_execution_time": (
                self._state_manager.get_metadata("avg_execution_time", 0)
                if self._state_manager
                else 0
            ),
            "max_execution_time": (
                self._state_manager.get_metadata("max_execution_time", 0)
                if self._state_manager
                else 0
            ),
            "last_execution_time": (
                self._state_manager.get_metadata("last_execution_time", 0)
                if self._state_manager
                else 0
            ),
            "last_error": (
                self._state_manager.get_metadata("last_error", None)
                if self._state_manager
                else None
            ),
            "last_error_time": (
                self._state_manager.get_metadata("last_error_time", None)
                if self._state_manager
                else None
            ),
            "cache_size": (
                len(self._state_manager.get("result_cache", {})) if self._state_manager else 0
            ),
            "label_stats": (
                self._state_manager.get_metadata("label_stats", {}) if self._state_manager else {}
            ),
        }

    def clear_cache(self) -> None:
        """
        Clear the classifier result cache.

        This method removes all cached classification results, which can be
        useful when changing configuration or when memory usage needs to be reduced.
        """
        if self._state_manager:
            self._state_manager.update("result_cache", {})
        logger.debug("Classifier cache cleared")

    def reset_state(self) -> None:
        """
        Reset classifier state.

        This method resets all state information, including execution counts,
        statistics, and the result cache. It then re-initializes the state with
        the current classifier configuration.
        """
        if self._state_manager:
            self._state_manager.reset()

            # Re-initialize state
            self._state_manager.update("name", self._name)
            self._state_manager.update("description", self._description)
            self._state_manager.update("config", self._config)
            self._state_manager.update("initialized", True)
            self._state_manager.update("execution_count", 0)
            self._state_manager.update("result_cache", {})

        logger.debug("Classifier state reset")


class BaseClassifierImplementation(ABC, Generic[L]):
    """
    Abstract base class for classifier implementations.

    This class provides a foundation for implementing the actual classification logic
    that is used by the Classifier class. It handles common implementation concerns
    like caching, state tracking, and error handling.

    ## Architecture
    The BaseClassifierImplementation class:
    - Uses the Template Method pattern for classification
    - Provides caching through the _classify_impl method
    - Delegates actual classification to _classify_impl_uncached
    - Supports both synchronous and asynchronous classification

    ## Lifecycle
    1. **Initialization**: Set up implementation with configuration
    2. **Classification**: Process text through the _classify_impl method
    3. **Caching**: Cache results for improved performance
    4. **Error Handling**: Handle and propagate errors consistently

    ## Examples
    ```python
    from sifaka.classifiers.base import BaseClassifierImplementation
    from sifaka.classifiers.result import ClassificationResult

    class MyImplementation(BaseClassifierImplementation[str]):
        def _classify_impl_uncached(self, text: str) -> ClassificationResult:
            # Implement classification logic
            return ClassificationResult(
                label="example",
                confidence=0.8,
                metadata={"source": "custom_implementation"}
            )
    ```
    """

    def __init__(
        self,
        config: Optional[Optional[ClassifierConfig]] = None,
        state_manager: Optional[Optional[StateManager]] = None,
    ):
        """
        Initialize the classifier implementation.

        Args:
            config: Configuration for the implementation
            state_manager: State manager for tracking state and statistics
        """
        self._config = config or ClassifierConfig()
        self._state_manager = state_manager or create_classifier_state()

    def classify(self, text: str) -> ClassificationResult[L, Any]:
        """
        Classify the given text.

        This method delegates to _classify_impl, which handles caching and
        then calls _classify_impl_uncached for the actual classification.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        return self._classify_impl(text)

    async def classify_async(self, text: str) -> ClassificationResult[L, Any]:
        """
        Classify the given text asynchronously.

        This method delegates to _classify_async_impl, which handles caching and
        then calls _classify_async_impl_uncached for the actual classification.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        return await self._classify_async_impl(text)

    def _classify_impl(self, text: str) -> ClassificationResult[L, Any]:
        """
        Implement classification with caching.

        This method handles caching of classification results and delegates
        to _classify_impl_uncached for the actual classification.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        # Check cache if enabled
        if self._config and hasattr(self._config, "cache_enabled") and self._config.cache_enabled:
            if self._state_manager:
                cache = self._state_manager.get("result_cache", {})
                if text in cache:
                    return cache[text]

        # Perform classification
        result = self._classify_impl_uncached(text)

        # Cache result if enabled
        if (
            self._config
            and hasattr(self._config, "cache_enabled")
            and self._config.cache_enabled
            and self._state_manager
        ):
            cache = self._state_manager.get("result_cache", {})
            cache[text] = result
            self._state_manager.update("result_cache", cache)

        return result

    async def _classify_async_impl(self, text: str) -> ClassificationResult[L, Any]:
        """
        Implement asynchronous classification with caching.

        This method handles caching of classification results and delegates
        to _classify_async_impl_uncached for the actual classification.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        # Check cache if enabled
        if self._config and hasattr(self._config, "cache_enabled") and self._config.cache_enabled:
            if self._state_manager:
                cache = self._state_manager.get("result_cache", {})
                if text in cache:
                    return cache[text]

        # Perform classification
        result = await self._classify_async_impl_uncached(text)

        # Cache result if enabled
        if (
            self._config
            and hasattr(self._config, "cache_enabled")
            and self._config.cache_enabled
            and self._state_manager
        ):
            cache = self._state_manager.get("result_cache", {})
            cache[text] = result
            self._state_manager.update("result_cache", cache)

        return result

    @abstractmethod
    def _classify_impl_uncached(self, text: str) -> ClassificationResult[L, Any]:
        """
        Implement classification without caching.

        This method must be implemented by subclasses to provide the actual
        classification logic.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        pass

    async def _classify_async_impl_uncached(self, text: str) -> ClassificationResult[L, Any]:
        """
        Implement asynchronous classification without caching.

        This method can be overridden by subclasses to provide asynchronous
        classification logic. The default implementation calls the synchronous
        version.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """
        # Default implementation calls the synchronous version
        return self._classify_impl_uncached(text)


def create_base_classification_result(
    label: Any,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
    issues: Optional[Optional[List[str]]] = None,
    suggestions: Optional[Optional[List[str]]] = None,
) -> ClassificationResult:
    """
    Create a standardized classification result.

    This utility function creates a ClassificationResult with the given parameters,
    ensuring consistent result creation across the framework.

    Args:
        label: The classification label
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional result metadata
        issues: List of issues found during classification
        suggestions: List of improvement suggestions

    Returns:
        Standardized ClassificationResult
    """
    return ClassificationResult(
        label=label,
        confidence=confidence,
        metadata=metadata or {},
        issues=issues or [],
        suggestions=suggestions or [],
    )


def safely_classify(
    operation: callable,
    classifier_name: str,
    component_name: str,
) -> ClassificationResult:
    """
    Safely execute a classification operation with error handling.

    This utility function wraps classification operations with standardized
    error handling, ensuring consistent error reporting and fallback behavior.

    Args:
        operation: The classification operation to execute
        classifier_name: Name of the classifier for error reporting
        component_name: Name of the component for error reporting

    Returns:
        The classification result or a fallback result in case of error

    Raises:
        ClassifierError: If classification fails and no fallback is available
    """
    try:
        return operation()
    except Exception as e:
        logger.error(
            f"Classification error in {component_name} ({classifier_name}): {str(e)}",
            exc_info=True,
        )

        # Wrap in ClassifierError if not already
        if not isinstance(e, ClassifierError):
            error = ClassifierError(
                f"Classification failed in {component_name} ({classifier_name}): {str(e)}"
            )
            error.__cause__ = e
            raise error
        raise e
