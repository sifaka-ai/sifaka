"""
Classifier Module

This module provides the main Classifier class for the Sifaka classifiers system.
It serves as the primary user-facing interface for text classification operations.

## Overview
The Classifier class is the main entry point for classification operations in Sifaka.
It provides a clean, consistent interface for classifying text, managing state, and
handling errors. The class delegates the actual classification work to implementation
classes while providing standardized result handling, caching, and statistics tracking.

## Components
1. **Classifier**: Main user-facing class for classification
2. **Engine**: Core classification engine (imported from engine.py)
3. **StateManager**: Manages classifier state (imported from utils/state.py)

## Architecture
The classifier component follows a layered architecture:
1. **User Interface Layer**: Classifier class with public methods
2. **Engine Layer**: Engine class that coordinates the classification flow
3. **Implementation Layer**: ClassifierImplementation interface implementations
4. **State Management Layer**: StateManager for tracking state and statistics

## Usage Examples
```python
from sifaka.classifiers import Classifier
from sifaka.classifiers.implementations.content import ToxicityClassifier

# Create classifier implementation
implementation = ToxicityClassifier()

# Create classifier
classifier = Classifier(
    implementation=implementation,
    name="toxicity_classifier",
    description="Detects toxic content in text"
)

# Classify text
result = classifier.classify("This is a friendly message.") if classifier else ""
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")

# Classify batch of texts
results = classifier.classify_batch([
    "This is a friendly message.",
    "This is a toxic message!",
    "This is another friendly message."
]) if classifier else ""
for i, result in enumerate(results):
    print(f"Text {i+1}: {result.label} ({result.confidence:.2f})")

# Get classifier statistics
stats = classifier.get_statistics() if classifier else ""
print(f"Execution count: {stats['execution_count']}")
print(f"Average execution time: {stats['avg_execution_time']:.2f}s")

# Clear cache
classifier.clear_cache() if classifier else ""
```

## Error Handling
The Classifier class provides robust error handling:
- ClassifierError: Raised when classification fails
- Automatic error tracking and statistics
- Fallback results for batch operations

## Configuration
The Classifier class supports configuration through the ClassifierConfig class:
- cache_enabled: Whether to enable result caching
- cache_size: Maximum number of cached results
- min_confidence: Minimum confidence threshold
"""

from typing import Any, Dict, List, Optional
import time

from .interfaces import ClassifierImplementation
from .engine import Engine
from ..utils.state import StateManager, create_classifier_state
from ..utils.common import update_statistics, record_error
from ..utils.logging import get_logger
from ..core.results import ClassificationResult
from ..utils.config import ClassifierConfig
from ..utils.errors import ClassifierError

# Configure logger
logger = get_logger(__name__)


class Classifier:
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
        config: Optional[Optional[ClassifierConfig]] = None,
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

    def classify(self, text: str) -> ClassificationResult:
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
                    return cached_result

            # Perform classification
            result = self._engine.classify(text)

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

            return result

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

    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
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
                results.append(ClassificationResult(label="", confidence=0.0))
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
        stats = update_statistics(
            stats,
            execution_time=execution_time,
            success=success,
            error=error,
            execution_count=execution_count,
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
