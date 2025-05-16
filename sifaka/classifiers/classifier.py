"""
Classifier Module

This module provides the main user-facing interface for the Sifaka classifier system.
It defines the `Classifier` class, which wraps classifier implementations with
standardized error handling, state management, and result formatting.

## Overview
The Classifier class is the primary entry point for classification operations,
providing a consistent interface regardless of the underlying implementation.
It abstracts away implementation details and provides additional features like
caching, error handling, and statistics tracking.

## Core Classes
1. **Classifier**: Main class for classification operations
   - Wraps a classifier implementation
   - Provides consistent error handling
   - Tracks statistics and state
   - Supports result caching

## Architecture
The classifier system follows a layered architecture:
1. **User Interface Layer**: Classifier class
2. **Engine Layer**: Manages classification workflow
3. **Implementation Layer**: Provides actual classification logic
4. **State Management Layer**: Tracks state and statistics
5. **Error Handling Layer**: Handles and reports errors

## Usage Example
```python
from sifaka.classifiers import create_toxicity_classifier

# Create a toxicity classifier
classifier = create_toxicity_classifier(
    name="content_moderation",
    description="Checks for toxic content in user inputs",
    min_confidence=0.7
)

# Classify text
result = classifier.classify("This is a friendly message.")

# Check result
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Metadata: {result.metadata}")
```

## Error Handling
The Classifier class provides robust error handling:
- ClassifierError: Base class for classifier errors
- ImplementationError: Raised when classification fails
- ConfigurationError: Raised for invalid configuration
- TimeoutError: Raised when classification times out
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, cast
import time

from sifaka.interfaces.classifier import (
    ClassifierImplementationProtocol as ClassifierImplementation,
)
from .engine import Engine
from ..utils.state import StateManager, create_classifier_state
from ..utils.common import update_statistics, record_error
from ..utils.logging import get_logger
from ..core.results import ClassificationResult
from ..utils.config import ClassifierConfig
from ..utils.errors import ClassifierError

# Define type variables for label and metadata types
L = TypeVar("L")
M = TypeVar("M")

# Configure logger
logger = get_logger(__name__)


class Classifier(Generic[L, M]):
    """
    Main user-facing class for classification.

    This class provides a standardized interface for text classification operations.
    It wraps classifier implementations with consistent state management, caching,
    error handling, and statistics tracking.

    ## Architecture
    The Classifier class follows a facade pattern:
    - Provides a simple, unified interface for classification
    - Delegates to an Engine instance for core classification logic
    - Uses StateManager for state tracking and statistics
    - Wraps implementation-specific errors in standardized ClassifierError

    ## Lifecycle
    1. **Initialization**: Set up classifier with implementation and configuration
    2. **Classification**: Process text through the implementation
    3. **Result Handling**: Return standardized ClassificationResult
    4. **State Management**: Track statistics and cache results
    5. **Error Handling**: Handle and track errors

    ## Examples
    ```python
    # Create classifier with implementation
    classifier = Classifier(
        implementation=ToxicityClassifier(),
        name="toxicity_classifier",
        description="Detects toxic content in text"
    )

    # Classify text
    result = classifier.classify("This is a friendly message.") if classifier else ""

    # Check result
    if result.confidence > 0.8:
        print(f"High confidence classification: {result.label}")
    ```
    """

    def __init__(
        self,
        implementation: ClassifierImplementation,
        config: Optional[ClassifierConfig] = None,
        name: str = "classifier",
        description: str = "Sifaka classifier for text classification",
    ):
        """
        Initialize the classifier.

        Args:
            implementation: The classifier implementation to use
            config: Classifier configuration
            name: Classifier name
            description: Classifier description
        """
        self._name = name
        self._description = description
        self._implementation = implementation
        self._config = config or ClassifierConfig()

        # Create state manager using the standardized state management
        self._state_manager = create_classifier_state()

        # Create engine
        self._engine = Engine(
            state_manager=self._state_manager,
            config=self._config,
        )

        # Initialize state
        if self._state_manager:
            self._state_manager.update("name", name)
            self._state_manager.update("description", description)
            self._state_manager.update("implementation", implementation)
            self._state_manager.update("config", self._config)
            self._state_manager.update("initialized", True)
            self._state_manager.update("execution_count", 0)
            self._state_manager.update("result_cache", {})

            # Set metadata
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

    def classify(self, text: str) -> ClassificationResult[L, M]:
        """
        Classify the given text.

        This method processes the input text through the classifier implementation
        and returns a standardized classification result. It handles state tracking,
        error handling, and statistics updates.

        Args:
            text: The text to classify

        Returns:
            The classification result with label, confidence, and metadata

        Raises:
            ClassifierError: If classification fails due to implementation errors,
                             configuration issues, or other exceptions
        """
        try:
            # Track execution count
            execution_count = (
                self._state_manager.get("execution_count", 0) if self._state_manager else 0
            )
            if self._state_manager:
                self._state_manager.update("execution_count", execution_count + 1)

            # Record start time
            start_time = time.time()
            if self._state_manager:
                self._state_manager.update("last_execution_time", start_time)

            # Check cache if enabled
            if self._config.cache_enabled and self._state_manager:
                cache_key = text
                cached_result = self._state_manager.get("result_cache", {}).get(cache_key)
                if cached_result:
                    return cast(ClassificationResult[L, M], cached_result)

            # Perform classification
            result = self._engine.classify(text, implementation=self._implementation)

            # Cache result if enabled
            if self._config.cache_enabled and self._state_manager:
                cache = self._state_manager.get("result_cache", {})
                cache[cache_key] = result
                if len(cache) > self._config.cache_size:
                    # Remove oldest entry
                    cache.pop(next(iter(cache)))
                self._state_manager.update("result_cache", cache)

            # Update statistics
            execution_time = time.time() - start_time
            self._update_statistics(execution_time, True)

            return cast(ClassificationResult[L, M], result)

        except Exception as e:
            # Record error
            if self._state_manager:
                self._state_manager.update("last_error", str(e))
                self._state_manager.update(
                    "error_count", self._state_manager.get("error_count", 0) + 1
                )

            # Update statistics
            execution_time = time.time() - start_time
            self._update_statistics(execution_time, False, e)

            # Raise standardized error
            raise ClassifierError(f"Classification failed: {str(e)}") from e

    def classify_batch(self, texts: List[str]) -> List[ClassificationResult[L, M]]:
        """
        Classify a batch of texts.

        This method processes multiple texts through the classifier implementation
        and returns a list of standardized classification results. It handles state
        tracking, error handling, and statistics updates for each text.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results, one for each input text

        Raises:
            ClassifierError: If classification fails for any text
        """
        results = []
        for text in texts:
            try:
                result = self.classify(text)
                results.append(result)
            except Exception as e:
                # Log error and continue with next text
                logger.error(f"Failed to classify text: {str(e)}")
                # Create a default result with empty string that will be cast internally
                default_result: ClassificationResult[Any, Any] = ClassificationResult(
                    label="",  # The constructor will handle type conversion
                    confidence=0.0,
                    passed=False,
                    message="Classification failed",
                )
                results.append(cast(ClassificationResult[L, M], default_result))
        return results

    def _update_statistics(
        self,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Update classifier statistics.

        This method updates the classifier's statistics with the latest execution
        information, including execution time, success status, and any errors.

        Args:
            execution_time: Time taken for the execution
            success: Whether the execution was successful
            error: Any error that occurred during execution
        """
        if not self._state_manager:
            return

        # Get current statistics
        stats = self._state_manager.get("statistics", {})
        execution_count = self._state_manager.get("execution_count", 0)

        # Update statistics
        update_statistics(
            self._state_manager,
            execution_time=execution_time,
            success=success,
            error=error,
        )

        # Update state
        self._state_manager.update("statistics", stats)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier statistics.

        This method returns the current statistics for the classifier, including
        execution count, success rate, average execution time, and error information.

        Returns:
            Dictionary containing classifier statistics
        """
        if not self._state_manager:
            return {}

        stats = self._state_manager.get("statistics", {})
        execution_count = self._state_manager.get("execution_count", 0)
        error_count = self._state_manager.get("error_count", 0)
        last_error = self._state_manager.get("last_error")

        return {
            "execution_count": execution_count,
            "error_count": error_count,
            "last_error": last_error,
            "success_rate": (
                (execution_count - error_count) / execution_count if execution_count > 0 else 0
            ),
            "avg_execution_time": stats.get("avg_execution_time", 0),
            "min_execution_time": stats.get("min_execution_time", 0),
            "max_execution_time": stats.get("max_execution_time", 0),
            "total_execution_time": stats.get("total_execution_time", 0),
        }

    def clear_cache(self) -> None:
        """
        Clear the classifier's result cache.

        This method clears all cached classification results from the state manager.
        """
        if self._state_manager:
            self._state_manager.update("result_cache", {})

    def reset_state(self) -> None:
        """
        Reset the classifier's state.

        This method resets all state information, including statistics, cache,
        and error tracking. It maintains the configuration and implementation.
        """
        if self._state_manager:
            self._state_manager.update("execution_count", 0)
            self._state_manager.update("error_count", 0)
            self._state_manager.update("last_error", None)
            self._state_manager.update("statistics", {})
            self._state_manager.update("result_cache", {})
