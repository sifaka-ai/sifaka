"""
Base classes for Sifaka classifiers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# No longer using lru_cache directly
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

# No longer using TypeGuard


@runtime_checkable
class TextProcessor(Protocol):
    """Protocol for text processing components."""

    def process(self, text: str) -> Dict[str, Any]: ...


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Protocol defining the interface for classifiers."""

    def classify(self, text: str) -> "ClassificationResult": ...
    def batch_classify(self, texts: List[str]) -> List["ClassificationResult"]: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def min_confidence(self) -> float: ...


@dataclass(frozen=True)
class ClassifierConfig:
    """
    Immutable configuration for classifiers.

    This class follows the same pattern as RuleConfig.
    All classifier-specific configuration options should be placed in the params dictionary:

    ```python
    config = ClassifierConfig(
        labels=["label1", "label2"],
        cost=1,
        params={
            "option1": "value1",
            "option2": "value2",
        }
    )
    ```
    """

    labels: List[str]
    cache_size: int = 0
    cost: int = 1
    min_confidence: float = 0.5
    params: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, list) or not all(isinstance(l, str) for l in self.labels):
            raise ValueError("labels must be a list of strings")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    def with_options(self, **kwargs: Any) -> "ClassifierConfig":
        """Create a new config with updated options."""
        return ClassifierConfig(**{**self.__dict__, **kwargs})

    def with_params(self, **kwargs: Any) -> "ClassifierConfig":
        """Create a new config with updated parameters."""
        new_params = {**self.params, **kwargs}
        return ClassifierConfig(
            labels=self.labels,
            cache_size=self.cache_size,
            cost=self.cost,
            min_confidence=self.min_confidence,
            params=new_params,
        )


class ClassificationResult(BaseModel):
    """
    Result of a classification.

    Attributes:
        label: The predicted label/class
        confidence: Confidence score for the prediction (0-1)
        metadata: Additional metadata about the classification
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)  # Immutable model

    label: str = Field(description="The classification label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def with_metadata(self, **kwargs: Any) -> "ClassificationResult":
        """Create a new result with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return ClassificationResult(
            label=self.label, confidence=self.confidence, metadata=new_metadata
        )


T = TypeVar("T", bound="BaseClassifier")


class BaseClassifier(ABC, BaseModel):
    """
    Base class for all Sifaka classifiers.

    A classifier provides predictions that can be used by rules.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    name: str = Field(description="Name of the classifier", min_length=1)
    description: str = Field(description="Description of the classifier", min_length=1)
    config: ClassifierConfig

    def model_post_init(self, _: Any) -> None:
        """Initialize cache if enabled."""
        # We don't need to do anything special for caching
        # The _classify_impl method will handle caching internally

    @property
    def min_confidence(self) -> float:
        """Get the minimum confidence threshold."""
        return self.config.min_confidence

    def validate_input(self, text: Any) -> bool:
        """Validate input text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        return True

    def validate_batch_input(self, texts: Any) -> bool:
        """Validate batch input texts."""
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings")
        return True

    @abstractmethod
    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """
        Implement classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _classify_impl_uncached")

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Wrapper around _classify_impl_uncached that handles caching.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence
        """
        # If caching is enabled, use a function-local cache
        if self.config.cache_size > 0:
            # Create a cache key from the text
            cache_key = text

            # Check if we have a cached result
            if not hasattr(self, "_result_cache"):
                # Initialize the cache as a dict mapping strings to ClassificationResults
                self._result_cache: Dict[str, ClassificationResult] = {}

            if cache_key in self._result_cache:
                return self._result_cache[cache_key]

            # Get the result
            result = self._classify_impl_uncached(text)

            # Cache the result
            if len(self._result_cache) >= self.config.cache_size:
                # Simple LRU: just clear the cache when it gets full
                # A more sophisticated implementation would use an OrderedDict
                self._result_cache.clear()
            self._result_cache[cache_key] = result

            return result
        else:
            # No caching, just call the implementation directly
            return self._classify_impl_uncached(text)

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the input text.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with prediction details

        Raises:
            ValueError: If input validation fails
        """
        self.validate_input(text)
        if not text.strip():
            return ClassificationResult(
                label="unknown", confidence=0.0, metadata={"reason": "empty_input"}
            )
        return self._classify_impl(text)

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults

        Raises:
            ValueError: If input validation fails
        """
        self.validate_batch_input(texts)
        return [self.classify(text) for text in texts]

    def warm_up(self) -> None:
        """
        Optional warm-up method for classifiers that need initialization.
        """
        pass

    @classmethod
    def create(
        cls: Type[T], name: str, description: str, labels: List[str], **config_kwargs: Any
    ) -> T:
        """
        Factory method to create a classifier instance.

        Args:
            name: Name of the classifier
            description: Description of what this classifier does
            labels: List of valid labels for classification
            **config_kwargs: Additional configuration parameters

        Returns:
            New classifier instance
        """
        # Extract params from config_kwargs if present
        params = config_kwargs.pop("params", {})

        # Create config with remaining kwargs
        config = ClassifierConfig(labels=labels, params=params, **config_kwargs)

        # Create instance
        return cls(name=name, description=description, config=config)


# Type alias for external usage
Classifier = BaseClassifier

# Export these types
__all__ = [
    "Classifier",
    "ClassificationResult",
    "ClassifierConfig",
    "ClassifierProtocol",
    "TextProcessor",
]
