"""
Base classes for Sifaka classifiers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeGuard


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
    """Immutable configuration for classifiers."""

    labels: List[str]
    cache_size: int = 0
    cost: int = 1
    min_confidence: float = 0.5
    params: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.labels, list) or not all(isinstance(l, str) for l in self.labels):
            raise ValueError("labels must be a list of strings")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

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

    label: str = Field(description="The classification label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_metadata(self) -> "ClassificationResult":
        """Ensure metadata is immutable."""
        self.metadata = dict(self.metadata)  # Create a new dict to prevent mutations
        return self

    def with_metadata(self, **kwargs) -> "ClassificationResult":
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

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[ClassifierConfig] = None,
    ) -> None:
        """
        Initialize a classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
        """
        super().__init__(
            name=name,
            description=description,
            config=config or ClassifierConfig(labels=[]),
        )

        # Initialize cache if enabled
        if self.config.cache_size > 0:
            self._classify_impl_original = self._classify_impl
            self._classify_impl = lru_cache(maxsize=self.config.cache_size)(
                self._classify_impl_original
            )

    name: str = Field(description="Name of the classifier")
    description: str = Field(description="Description of the classifier")
    config: ClassifierConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def min_confidence(self) -> float:
        """Get the minimum confidence threshold."""
        return self.config.min_confidence

    def validate_input(self, text: str) -> TypeGuard[str]:
        """Validate input text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        return True

    def validate_batch_input(self, texts: List[str]) -> TypeGuard[List[str]]:
        """Validate batch input texts."""
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings")
        return True

    @abstractmethod
    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _classify_impl")

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
    def create(cls: Type[T], name: str, description: str, labels: List[str], **config_kwargs) -> T:
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
