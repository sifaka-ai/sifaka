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
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")

# Classify batch of texts
results = classifier.classify_batch([
    "This is a friendly message.",
    "This is a toxic message!",
    "This is another friendly message."
])
for i, result in enumerate(results):
    print(f"Text {i+1}: {result.label} ({result.confidence:.2f})")

# Get classifier statistics
stats = classifier.get_statistics()
print(f"Execution count: {stats['execution_count']}")
print(f"Average execution time: {stats['avg_execution_time']:.2f}s")

# Clear cache
classifier.clear_cache()
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
- async_enabled: Whether to enable asynchronous classification
"""

from typing import Any, Dict, List, Optional
import time
import asyncio

from .interfaces import ClassifierImplementation
from .engine import Engine
from ..utils.state import StateManager, create_classifier_state
from ..utils.common import update_statistics, record_error
from ..utils.logging import get_logger
from .result import ClassificationResult
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
    result = classifier.classify("This is a friendly message.")

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
            execution_count = self._state_manager.get("execution_count", 0)
            self._state_manager.update("execution_count", execution_count + 1)

            # Record start time
            start_time = time.time()
            self._state_manager.set_metadata("execution_start_time", start_time)

            # Run engine
            result = self._engine.classify(
                text=text,
                implementation=self._implementation,
            )

            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update statistics
            self._update_statistics(execution_time, success=True)

            return result

        except Exception as e:
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update statistics
            self._update_statistics(execution_time, success=False, error=e)

            # Raise as classifier error
            if isinstance(e, ClassifierError):
                raise e
            raise ClassifierError(f"Classification failed: {str(e)}")

    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        This method processes multiple input texts through the classifier implementation
        and returns a list of standardized classification results. It handles errors for
        individual texts without failing the entire batch operation.

        Args:
            texts: The texts to classify

        Returns:
            List of classification results, one for each input text

        Raises:
            ClassifierError: If the entire batch operation fails
                             (individual text failures are handled gracefully)
        """
        results = []
        for text in texts:
            try:
                result = self.classify(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify text: {str(e)}")
                # Create fallback result
                results.append(
                    ClassificationResult(
                        label=self._config.params.get("fallback_label", "unknown"),
                        confidence=0.0,
                        metadata={"error": str(e), "error_type": type(e).__name__},
                        issues=[f"Classification failed: {str(e)}"],
                        suggestions=["Try with a different classifier or improve the input text"],
                    )
                )
        return results

    async def classify_async(self, text: str) -> ClassificationResult:
        """
        Classify the given text asynchronously.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ClassifierError: If classification fails
        """
        # Check if async is enabled
        if not self._config.async_enabled:
            raise ClassifierError("Async execution is not enabled in the configuration")

        try:
            # Track execution count
            execution_count = self._state_manager.get("execution_count", 0)
            self._state_manager.update("execution_count", execution_count + 1)

            # Record start time
            start_time = time.time()
            self._state_manager.set_metadata("execution_start_time", start_time)

            # Run engine
            result = await self._engine.classify_async(
                text=text,
                implementation=self._implementation,
            )

            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update statistics
            self._update_statistics(execution_time, success=True)

            return result

        except Exception as e:
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update statistics
            self._update_statistics(execution_time, success=False, error=e)

            # Raise as classifier error
            if isinstance(e, ClassifierError):
                raise e
            raise ClassifierError(f"Classification failed: {str(e)}")

    async def classify_batch_async(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts asynchronously.

        Args:
            texts: The texts to classify

        Returns:
            List of classification results

        Raises:
            ClassifierError: If classification fails
        """
        # Check if async is enabled
        if not self._config.async_enabled:
            raise ClassifierError("Async execution is not enabled in the configuration")

        # Create tasks for all texts
        tasks = [self.classify_async(text) for text in texts]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to classify text: {str(result)}")
                # Create fallback result
                processed_results.append(
                    ClassificationResult(
                        label=self._config.params.get("fallback_label", "unknown"),
                        confidence=0.0,
                        metadata={"error": str(result), "error_type": type(result).__name__},
                        issues=[f"Classification failed: {str(result)}"],
                        suggestions=["Try with a different classifier or improve the input text"],
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _update_statistics(
        self,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Update classifier statistics.

        This internal method updates the classifier's execution statistics,
        including success/failure counts, execution times, and error tracking.
        It uses the standardized utility function from utils.common and adds
        classifier-specific statistics.

        Args:
            execution_time: Execution time in seconds
            success: Whether execution was successful
            error: Optional error that occurred
        """
        # Use the standardized utility function
        update_statistics(
            state_manager=self._state_manager,
            execution_time=execution_time,
            success=success,
            error=error,
        )

        # Update additional classifier-specific statistics
        self._state_manager.set_metadata("last_execution_time", execution_time)

        max_time = self._state_manager.get_metadata("max_execution_time", 0)
        if execution_time > max_time:
            self._state_manager.set_metadata("max_execution_time", execution_time)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier statistics.

        This method returns a comprehensive dictionary of classifier statistics,
        including execution counts, success/failure rates, timing information,
        error details, and label distribution statistics.

        Returns:
            Dictionary with classifier statistics including:
            - name: Classifier name
            - execution_count: Total number of classification operations
            - success_count: Number of successful classifications
            - failure_count: Number of failed classifications
            - error_count: Number of errors encountered
            - avg_execution_time: Average execution time in seconds
            - max_execution_time: Maximum execution time in seconds
            - last_execution_time: Most recent execution time in seconds
            - last_error: Most recent error message
            - last_error_time: Timestamp of most recent error
            - cache_size: Current size of the result cache
            - label_stats: Distribution of classification labels
        """
        return {
            "name": self._name,
            "execution_count": self._state_manager.get("execution_count", 0),
            "success_count": self._state_manager.get_metadata("success_count", 0),
            "failure_count": self._state_manager.get_metadata("failure_count", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
            "avg_execution_time": self._state_manager.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state_manager.get_metadata("max_execution_time", 0),
            "last_execution_time": self._state_manager.get_metadata("last_execution_time", 0),
            "last_error": self._state_manager.get_metadata("last_error", None),
            "last_error_time": self._state_manager.get_metadata("last_error_time", None),
            "cache_size": len(self._state_manager.get("result_cache", {})),
            "label_stats": self._state_manager.get_metadata("label_stats", {}),
        }

    def clear_cache(self) -> None:
        """
        Clear the classifier result cache.

        This method removes all cached classification results, which can be
        useful when changing configuration or when memory usage needs to be reduced.
        """
        self._state_manager.update("result_cache", {})
        logger.debug("Classifier cache cleared")

    def reset_state(self) -> None:
        """
        Reset classifier state.

        This method resets all state information, including execution counts,
        statistics, and the result cache. It then re-initializes the state with
        the current classifier configuration.

        This is useful when you want to start fresh with the same classifier
        configuration, for example when running a new batch of classifications
        that should not be influenced by previous runs.
        """
        self._state_manager.reset()

        # Re-initialize state
        self._state_manager.update("name", self._name)
        self._state_manager.update("description", self._description)
        self._state_manager.update("implementation", self._implementation)
        self._state_manager.update("config", self._config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("result_cache", {})

        logger.debug("Classifier state reset")
