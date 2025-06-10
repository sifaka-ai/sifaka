"""Base classifier classes for Sifaka.

This module provides the base classifier interface and common functionality
for all classifiers

Key features:
- Async-first design compatible with PydanticAI
- Rich classification results with confidence scores
- Integration with Sifaka's logging and error handling
- Support for caching and performance optimization
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Result of a classification operation.

    This class encapsulates the result of classifying text. It provides
    detailed information about the predicted class, confidence score,
    and additional metadata about the classification process.

    Attributes:
        label: Predicted class label
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata about the classification
        processing_time_ms: Time taken to perform classification in milliseconds
    """

    label: str
    confidence: float
    metadata: Dict[str, Any] = None
    processing_time_ms: float = 0.0

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.metadata is None:
            self.metadata = {}

        # Validate confidence score
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


class BaseClassifier(ABC):
    """Base class for all classifiers in Sifaka.

    This abstract base class defines the interface that all classifiers must implement.
    It provides common functionality for classification operations and integrates with
    Sifaka's logging and error handling systems.

    Classifiers should be async-first but also support sync operations for backward
    compatibility. The base class handles the sync/async coordination.

    Attributes:
        name: Human-readable name of the classifier
        description: Description of what the classifier does
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize the base classifier.

        Args:
            name: Human-readable name of the classifier
            description: Description of what the classifier does
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text asynchronously.

        This is the main classification method that all classifiers must implement.
        It should be async-first and return a detailed ClassificationResult.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with detailed classification information

        Raises:
            ValidationError: If classification cannot be performed
        """

    def classify(self, text: str) -> ClassificationResult:
        """Classify text synchronously.

        This method provides sync compatibility by running the async classification
        in an event loop. Use classify_async() when possible for better performance.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with detailed classification information

        Raises:
            ValidationError: If classification cannot be performed
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a new task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.classify_async(text))
                    return future.result()
            else:
                # No running loop, we can use asyncio.run
                return asyncio.run(self.classify_async(text))
        except Exception as e:
            logger.error(
                f"Sync classification failed for {self.name}",
                extra={
                    "classifier": self.name,
                    "text_length": len(text) if text else 0,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise ValidationError(
                f"Classification failed for {self.name}: {str(e)}",
                error_code="classification_execution_error",
                context={
                    "classifier": self.name,
                    "text_length": len(text) if text else 0,
                    "error_type": type(e).__name__,
                },
                suggestions=[
                    "Check classifier configuration",
                    "Verify input text is valid",
                    "Check system resources",
                ],
            ) from e

    @abstractmethod
    def get_classes(self) -> List[str]:
        """Get the list of possible class labels.

        Returns:
            List of possible class labels that this classifier can predict
        """

    def create_classification_result(
        self,
        label: str,
        confidence: float,
        metadata: Dict[str, Any] = None,
        processing_time_ms: float = 0.0,
    ) -> ClassificationResult:
        """Create a ClassificationResult with consistent formatting.

        This helper method creates a ClassificationResult with the classifier's
        metadata and ensures consistent formatting across all classifiers.

        Args:
            label: Predicted class label
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata
            processing_time_ms: Processing time in milliseconds

        Returns:
            Properly formatted ClassificationResult
        """
        if metadata is None:
            metadata = {}

        # Add classifier information to metadata
        metadata.update(
            {
                "classifier_name": self.name,
                "classifier_description": self.description,
            }
        )

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata=metadata,
            processing_time_ms=processing_time_ms,
        )

    def create_empty_text_result(self, default_label: str = "unknown") -> ClassificationResult:
        """Create a result for empty or None text.

        Args:
            default_label: Default label to use for empty text

        Returns:
            ClassificationResult indicating empty text
        """
        return self.create_classification_result(
            label=default_label,
            confidence=0.0,
            metadata={
                "reason": "empty_text",
                "input_length": 0,
            },
        )

    def __str__(self) -> str:
        """String representation of the classifier."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the classifier."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"


class CachedClassifier(BaseClassifier):
    """Base class for classifiers with LRU caching.

    This class extends BaseClassifier to provide automatic caching of classification
    results using an LRU cache. This can significantly improve performance for
    classifiers that are called repeatedly with the same text.

    Attributes:
        cache_size: Maximum number of results to cache
    """

    def __init__(self, name: str, description: str = "", cache_size: int = 128):
        """Initialize the cached classifier.

        Args:
            name: Human-readable name of the classifier
            description: Description of what the classifier does
            cache_size: Maximum number of results to cache
        """
        super().__init__(name=name, description=description)
        self.cache_size = cache_size

        # Create a cached version of the classification method
        self._cached_classify = lru_cache(maxsize=cache_size)(self._classify_uncached)

        logger.debug(
            "Created cached classifier",
            extra={
                "classifier": self.name,
                "cache_size": cache_size,
            },
        )

    def classify(self, text: str) -> ClassificationResult:
        """Classify text synchronously with caching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult (potentially from cache)
        """
        if not text or not text.strip():
            return self.create_empty_text_result()

        start_time = time.time()

        # Use cached classification
        result = self._cached_classify(text.strip())

        # Update processing time
        processing_time = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time

        # Add cache information to metadata
        result.metadata["cached"] = True
        result.metadata["cache_size"] = self.cache_size
        result.metadata["cache_info"] = self._cached_classify.cache_info()._asdict()

        return result

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text asynchronously with caching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult (potentially from cache)
        """
        # For cached classifiers, we can just call the synchronous method
        # since _classify_uncached is already synchronous
        return self.classify(text)

    @abstractmethod
    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform classification without caching.

        This method should implement the actual classification logic.
        It will be automatically cached by the classify_async method.

        Args:
            text: The text to classify (guaranteed to be non-empty and stripped)

        Returns:
            ClassificationResult with classification information
        """

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cached_classify.cache_clear()
        logger.debug("Cleared cache for classifier", extra={"classifier": self.name})

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache state.

        Returns:
            Dictionary with cache statistics
        """
        cache_info = self._cached_classify.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": (
                cache_info.hits / (cache_info.hits + cache_info.misses)
                if (cache_info.hits + cache_info.misses) > 0
                else 0.0
            ),
        }


class TimingMixin:
    """Mixin for classifiers that need performance timing."""

    def time_operation(self, operation_name: str = "classification"):
        """Context manager for timing operations.

        Args:
            operation_name: Name of the operation being timed

        Returns:
            Context manager that tracks timing
        """
        return logger.performance_timer(operation_name, classifier=getattr(self, "name", "unknown"))
