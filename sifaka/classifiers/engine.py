"""
Classifier Engine Module

This module provides the core classification engine for the Sifaka classifiers system.
It coordinates the flow between components, handles caching, and manages state.

## Overview
The Engine class is the central component of the classification system, responsible
for coordinating the classification process, managing caching, tracking statistics,
and handling errors. It serves as an intermediary between the user-facing Classifier
class and the implementation-specific classification logic.

## Components
1. **Engine**: Core classification engine that coordinates the flow
2. **StateManager**: Manages engine state and statistics
3. **ClassifierConfig**: Configuration for the engine
4. **ClassifierImplementation**: Interface for classifier implementations

## Architecture
The engine follows a layered architecture:
1. **Interface Layer**: Public methods for classification
2. **Caching Layer**: Optional caching of classification results
3. **Execution Layer**: Delegation to classifier implementations
4. **State Management Layer**: Tracking of statistics and state
5. **Error Handling Layer**: Standardized error handling and reporting

## Usage Examples
```python
from sifaka.classifiers.engine import Engine
from sifaka.classifiers.config import ClassifierConfig
from sifaka.utils.state import StateManager

# Create engine
engine = Engine(
    state_manager=StateManager(),
    config=ClassifierConfig(cache_enabled=True, cache_size=100)
)

# Classify text
result = engine.classify(
    text="This is a friendly message.",
    implementation=implementation
)

# Access result
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")

# Classify text asynchronously
import asyncio

async def classify_async():
    result = await engine.classify_async(
        text="This is a friendly message.",
        implementation=implementation
    )
    return result

result = asyncio.run(classify_async())
```

## Error Handling
The Engine class provides robust error handling:
- ClassifierError: Raised when classification fails
- ImplementationError: Raised when implementation fails
- Automatic error tracking and statistics
- Detailed error messages and context

## Configuration
The Engine class supports configuration through the ClassifierConfig class:
- cache_enabled: Whether to enable result caching
- cache_size: Maximum number of cached results
- min_confidence: Minimum confidence threshold
- async_enabled: Whether to enable asynchronous classification
"""

from typing import Any, Dict, List, Optional
import time

from .interfaces import ClassifierImplementation
from ..utils.state import StateManager
from ..utils.logging import get_logger
from .result import ClassificationResult
from .config import ClassifierConfig
from ..utils.errors import ClassifierError, safely_execute_component_operation as safely_execute
from .adapters import ImplementationError

# Configure logger
logger = get_logger(__name__)


class Engine:
    """
    Core classification engine for the Sifaka classifiers system.

    This class provides the central coordination logic for the classification process,
    managing the flow between components, handling caching, tracking statistics,
    and standardizing error handling.

    ## Architecture
    The Engine class follows a component-based architecture:
    - Uses StateManager for state tracking and statistics
    - Implements caching with configurable cache size
    - Delegates to classifier implementations for actual classification
    - Provides both synchronous and asynchronous interfaces
    - Handles errors with standardized error classes

    ## Lifecycle
    1. **Initialization**: Set up engine with state manager and configuration
    2. **Classification**: Process text through classifier implementations
    3. **Caching**: Optionally cache results for improved performance
    4. **Statistics**: Track execution statistics and performance metrics
    5. **Error Handling**: Handle and track errors with detailed context

    ## Examples
    ```python
    # Create engine with configuration
    engine = Engine(
        state_manager=StateManager(),
        config=ClassifierConfig(
            cache_enabled=True,
            cache_size=100,
            min_confidence=0.7
        )
    )

    # Classify text
    result = engine.classify(
        text="This is a friendly message.",
        implementation=implementation
    )
    ```
    """

    def __init__(
        self,
        state_manager: StateManager,
        config: Optional[ClassifierConfig] = None,
    ):
        """
        Initialize the engine.

        This method sets up the engine with the provided state manager and configuration.
        It initializes the internal state, including execution counters, result cache,
        and metadata.

        Args:
            state_manager: State manager for tracking state and statistics
            config: Engine configuration with settings for caching, confidence thresholds, etc.
        """
        self._state_manager = state_manager
        self._config = config or ClassifierConfig()

        # Initialize state
        self._state_manager.update("config", self._config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("result_cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "engine")
        self._state_manager.set_metadata("creation_time", time.time())

    def classify(
        self,
        text: str,
        implementation: ClassifierImplementation,
    ) -> ClassificationResult:
        """
        Classify the given text.

        This method processes the input text through the provided classifier implementation
        and returns a standardized classification result. It handles caching, state tracking,
        error handling, and statistics updates.

        Args:
            text: The text to classify, which can be any string content
                 that the implementation can process
            implementation: The classifier implementation to use for classification,
                           which must implement the ClassifierImplementation interface

        Returns:
            The classification result containing:
            - label: The classification label (e.g., "positive", "toxic")
            - confidence: A confidence score between 0.0 and 1.0
            - metadata: Optional additional information about the classification
            - issues: Any issues encountered during classification
            - suggestions: Suggestions for improving the input

        Raises:
            ClassifierError: If classification fails due to engine errors
            ImplementationError: If the implementation fails to classify the text
        """
        # Track execution count
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)

        # Check cache if enabled
        if self._config.cache_enabled:
            cache = self._state_manager.get("result_cache", {})
            cache_key = f"{text}_{implementation.__class__.__name__}"
            if cache_key in cache:
                self._state_manager.set_metadata("cache_hit", True)
                return cache[cache_key]

        # Record start time
        start_time = time.time()

        try:
            # Store components in state
            self._state_manager.update("implementation", implementation)
            self._state_manager.update("text", text)

            # Classify text
            result = self._classify_text(text, implementation)

            # Cache result if caching is enabled
            if self._config.cache_enabled:
                cache = self._state_manager.get("result_cache", {})
                cache_size = self._config.cache_size
                cache_key = f"{text}_{implementation.__class__.__name__}"

                # If cache is full, remove oldest entry
                if len(cache) >= cache_size:
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]

                # Add result to cache
                cache[cache_key] = result
                self._state_manager.update("result_cache", cache)

            return result

        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Log error
            logger.error(f"Engine execution error: {str(e)}")

            # Raise as classifier error
            raise ClassifierError(f"Engine execution failed: {str(e)}")

        finally:
            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self._state_manager.set_metadata("last_execution_time", execution_time)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            count = execution_count + 1
            new_avg = (avg_time * (count - 1) + execution_time) / count
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

    def _classify_text(
        self,
        text: str,
        implementation: ClassifierImplementation,
    ) -> ClassificationResult:
        """
        Classify text using the implementation.

        This internal method delegates the actual classification work to the
        implementation and handles post-processing of the result, including
        confidence threshold checking and statistics tracking.

        Args:
            text: The text to classify, which can be any string content
                 that the implementation can process
            implementation: The classifier implementation to use for classification,
                           which must implement the ClassifierImplementation interface

        Returns:
            The classification result from the implementation, potentially
            enhanced with additional issues and suggestions

        Raises:
            ImplementationError: If the implementation fails to classify the text
                                or returns an invalid result
        """

        def classify_operation():
            return implementation.classify(text)

        result = safely_execute(
            operation=classify_operation,
            component_name="implementation",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )

        # Check if result meets minimum confidence threshold
        if result.confidence < self._config.min_confidence:
            # Add issue and suggestion
            result = result.with_issues(
                [
                    f"Confidence ({result.confidence:.2f}) below threshold ({self._config.min_confidence:.2f})"
                ]
            )
            result = result.with_suggestions(
                ["Consider using a different classifier or improving the input text"]
            )

        # Track statistics
        stats = self._state_manager.get_metadata("label_stats", {})
        stats[result.label] = stats.get(result.label, 0) + 1
        self._state_manager.set_metadata("label_stats", stats)

        return result

    async def classify_async(
        self,
        text: str,
        implementation: ClassifierImplementation,
    ) -> ClassificationResult:
        """
        Classify the given text asynchronously.

        This method provides the same functionality as classify() but operates
        asynchronously, allowing for non-blocking classification operations.
        It is particularly useful for implementations that use external APIs
        or perform I/O operations.

        Args:
            text: The text to classify, which can be any string content
                 that the implementation can process
            implementation: The classifier implementation to use for classification,
                           which must implement the ClassifierImplementation interface

        Returns:
            The classification result containing:
            - label: The classification label (e.g., "positive", "toxic")
            - confidence: A confidence score between 0.0 and 1.0
            - metadata: Optional additional information about the classification
            - issues: Any issues encountered during classification
            - suggestions: Suggestions for improving the input

        Raises:
            ClassifierError: If classification fails due to engine errors or
                            if async execution is not enabled in the configuration
            ImplementationError: If the implementation fails to classify the text
        """
        # Check if async is enabled
        if not self._config.async_enabled:
            raise ClassifierError("Async execution is not enabled in the configuration")

        # Track execution count
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)

        # Check cache if enabled
        if self._config.cache_enabled:
            cache = self._state_manager.get("result_cache", {})
            cache_key = f"{text}_{implementation.__class__.__name__}"
            if cache_key in cache:
                self._state_manager.set_metadata("cache_hit", True)
                return cache[cache_key]

        # Record start time
        start_time = time.time()

        try:
            # Store components in state
            self._state_manager.update("implementation", implementation)
            self._state_manager.update("text", text)

            # Classify text asynchronously
            result = await implementation.classify_async(text)

            # Cache result if caching is enabled
            if self._config.cache_enabled:
                cache = self._state_manager.get("result_cache", {})
                cache_size = self._config.cache_size
                cache_key = f"{text}_{implementation.__class__.__name__}"

                # If cache is full, remove oldest entry
                if len(cache) >= cache_size:
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]

                # Add result to cache
                cache[cache_key] = result
                self._state_manager.update("result_cache", cache)

            return result

        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Log error
            logger.error(f"Engine execution error: {str(e)}")

            # Raise as classifier error
            raise ClassifierError(f"Engine execution failed: {str(e)}")

        finally:
            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self._state_manager.set_metadata("last_execution_time", execution_time)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            count = execution_count + 1
            new_avg = (avg_time * (count - 1) + execution_time) / count
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)
