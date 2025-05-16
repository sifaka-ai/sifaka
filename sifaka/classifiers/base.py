"""
Classifier Base Module

This module provides the base classes for the Sifaka classifiers system.
These classes implement common functionality shared across different
classifier implementations.

## Overview
The base classes provide a foundation for building classifier implementations,
with shared functionality for caching, error handling, and result formatting.
They ensure consistent behavior across different classifier types while
allowing for implementation-specific customization.

## Base Classes
1. **BaseClassifier**: Abstract base class for all classifiers
2. **CachedClassifier**: Base class for classifiers with result caching
3. **ConfigurableClassifier**: Base class for classifiers with configuration

## Architecture
The base classes follow a layered architecture:
1. **Interface Layer**: Implements the ClassifierImplementation interface
2. **Caching Layer**: Optional caching of classification results
3. **Configuration Layer**: Configuration management and validation
4. **Error Handling Layer**: Standardized error handling and reporting

## Usage Examples
```python
from sifaka.classifiers.base import BaseClassifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig

class MyClassifier(BaseClassifier):
    def classify(self, text: str) -> ClassificationResult:
        # Implementation-specific classification logic
        return ClassificationResult(
            label="positive",
            confidence=0.8,
            metadata={"source": "my_classifier"}
        )

# Create classifier with configuration
classifier = MyClassifier(
    config=ClassifierConfig(
        cache_enabled=True,
        cache_size=100,
        min_confidence=0.7
    )
)

# Classify text
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```

## Error Handling
The base classes provide robust error handling:
- ClassifierError: Base class for classifier errors
- ImplementationError: Raised when classification fails
- ConfigurationError: Raised for invalid configuration
- Automatic error tracking and statistics

## Configuration
The base classes support configuration through the ClassifierConfig class:
- cache_enabled: Whether to enable result caching
- cache_size: Maximum number of cached results
- min_confidence: Minimum confidence threshold
"""

from typing import Any, Dict, Generic, Optional, TypeVar, cast, Union
from ..core.results import ClassificationResult
from ..utils.config import ClassifierConfig
from ..utils.errors import ClassifierError
from ..utils.errors import safely_execute_component_operation as safely_execute
from ..utils.errors.results import ErrorResult
from sifaka.interfaces.classifier import (
    ClassifierImplementationProtocol as ClassifierImplementation,
)

# Define type variables for label and metadata types
L = TypeVar("L")
M = TypeVar("M", bound=Dict[str, Any])


class BaseClassifier(ClassifierImplementation, Generic[L, M]):
    """
    Base class for all classifiers.

    This class provides the foundation for building classifier implementations,
    with shared functionality for error handling, result formatting, and
    configuration management.

    ## Architecture
    The BaseClassifier class:
    - Implements the ClassifierImplementation interface
    - Provides configuration management
    - Handles errors with standardized error classes
    - Formats results consistently

    ## Lifecycle
    1. **Initialization**: Set up classifier with configuration
    2. **Classification**: Process text through implementation
    3. **Error Handling**: Handle and track errors
    4. **Result Formatting**: Format results consistently

    ## Examples
    ```python
    class MyClassifier(BaseClassifier):
        def classify(self, text: str) -> ClassificationResult:
            # Implementation-specific classification logic
            return ClassificationResult(
                label="positive",
                confidence=0.8,
                metadata={"source": "my_classifier"}
            )
    ```
    """

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        """
        Initialize the base classifier.

        This method sets up the classifier with the provided configuration.
        It initializes the internal state and validates the configuration.

        Args:
            config: Classifier configuration with settings for caching,
                   confidence thresholds, etc.
        """
        self._config = config or ClassifierConfig()

    def classify(self, text: str) -> ClassificationResult[L, M]:
        """
        Classify the given text.

        This method processes the input text and returns a classification result.
        It handles error handling and result formatting.

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
            ClassifierError: If classification fails due to implementation errors,
                            invalid input, or other issues
        """

        def classify_operation() -> ClassificationResult[L, M]:
            return self._classify_impl(text)

        try:
            result = safely_execute(
                operation=classify_operation,
                component_name=self.__class__.__name__,
                component_type="ClassifierImplementation",
                error_class=ClassifierError,
            )
            return cast(ClassificationResult[L, M], result)
        except Exception as e:
            raise ClassifierError(f"Classification failed: {str(e)}") from e

    def _classify_impl(self, text: str) -> ClassificationResult[L, M]:
        """
        Implementation-specific classification logic.

        This method should be implemented by subclasses to provide the actual
        classification functionality. It is called by the public classify method
        after error handling and result formatting.

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
            ClassifierError: If classification fails due to implementation errors,
                            invalid input, or other issues
        """
        raise NotImplementedError("Subclasses must implement _classify_impl")


class CachedClassifier(BaseClassifier[L, M]):
    """
    Base class for classifiers with result caching.

    This class extends BaseClassifier with caching functionality, allowing
    for improved performance by caching classification results.

    ## Architecture
    The CachedClassifier class:
    - Extends BaseClassifier with caching functionality
    - Uses a simple in-memory cache with configurable size
    - Handles cache invalidation and updates
    - Maintains cache statistics

    ## Lifecycle
    1. **Initialization**: Set up classifier with configuration
    2. **Classification**: Process text through implementation
    3. **Caching**: Cache results for improved performance
    4. **Error Handling**: Handle and track errors
    5. **Result Formatting**: Format results consistently

    ## Examples
    ```python
    class MyCachedClassifier(CachedClassifier):
        def _classify_impl(self, text: str) -> ClassificationResult:
            # Implementation-specific classification logic
            return ClassificationResult(
                label="positive",
                confidence=0.8,
                metadata={"source": "my_classifier"}
            )
    ```
    """

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        """
        Initialize the cached classifier.

        This method sets up the classifier with the provided configuration.
        It initializes the internal state, including the result cache.

        Args:
            config: Classifier configuration with settings for caching,
                   confidence thresholds, etc.
        """
        super().__init__(config)
        self._cache: Dict[str, ClassificationResult[L, M]] = {}

    def classify(self, text: str) -> ClassificationResult[L, M]:
        """
        Classify the given text.

        This method processes the input text and returns a classification result.
        It handles caching, error handling, and result formatting.

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
            ClassifierError: If classification fails due to implementation errors,
                            invalid input, or other issues
        """
        if not self._config.cache_enabled:
            return super().classify(text)

        cache_key = text
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = super().classify(text)
        if len(self._cache) >= self._config.cache_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result


class ConfigurableClassifier(BaseClassifier[L, M]):
    """
    Base class for classifiers with configuration.

    This class extends BaseClassifier with additional configuration functionality,
    allowing for more complex configuration management and validation.

    ## Architecture
    The ConfigurableClassifier class:
    - Extends BaseClassifier with configuration functionality
    - Provides configuration validation and management
    - Handles configuration updates and changes
    - Maintains configuration state

    ## Lifecycle
    1. **Initialization**: Set up classifier with configuration
    2. **Configuration**: Validate and manage configuration
    3. **Classification**: Process text through implementation
    4. **Error Handling**: Handle and track errors
    5. **Result Formatting**: Format results consistently

    ## Examples
    ```python
    class MyConfigurableClassifier(ConfigurableClassifier):
        def _classify_impl(self, text: str) -> ClassificationResult:
            # Implementation-specific classification logic
            return ClassificationResult(
                label="positive",
                confidence=0.8,
                metadata={"source": "my_classifier"}
            )

        def _validate_config(self, config: ClassifierConfig) -> None:
            # Custom configuration validation
            if config.threshold < 0.0 or config.threshold > 1.0:
                raise ValueError("threshold must be between 0.0 and 1.0")
    ```
    """

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        """
        Initialize the configurable classifier.

        This method sets up the classifier with the provided configuration.
        It initializes the internal state and validates the configuration.

        Args:
            config: Classifier configuration with settings for caching,
                   confidence thresholds, etc.
        """
        super().__init__(config)
        self._validate_config(self._config)

    def _validate_config(self, config: ClassifierConfig) -> None:
        """
        Validate the classifier configuration.

        This method validates the provided configuration, ensuring that all
        required settings are present and valid. It can be overridden by
        subclasses to provide additional validation.

        Args:
            config: The configuration to validate

        Raises:
            ValueError: If the configuration is invalid
        """
        if config.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if config.threshold < 0.0 or config.threshold > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
