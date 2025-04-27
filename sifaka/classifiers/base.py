"""
Base classes for Sifaka classifiers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClassificationResult(BaseModel):
    """
    Result of a classification.

    Attributes:
        label: The predicted label/class
        confidence: Confidence score for the prediction (0-1)
        metadata: Additional metadata about the classification
    """

    label: Any = Field(description="The classification label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Classifier(ABC, BaseModel):
    """
    Base class for all Sifaka classifiers.

    A classifier provides predictions that can be used by rules.

    Attributes:
        name: The name of the classifier
        description: Description of the classifier
        config: Configuration for the classifier
        labels: List of possible labels/classes
        cache_size: Size of the LRU cache (0 to disable)
        cost: Estimated computational cost (higher numbers are more expensive)
        min_confidence: Minimum confidence threshold for classification
    """

    name: str = Field(description="Name of the classifier")
    description: str = Field(description="Description of the classifier")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    labels: List[str] = Field(description="List of possible classification labels")
    cache_size: int = Field(default=0, ge=0, description="Size of the classification cache")
    cost: int = Field(default=1, ge=0, description="Cost of running the classifier")
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the input text.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with prediction details
        """
        pass

    @abstractmethod
    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        pass

    def warm_up(self) -> None:
        """
        Optional warm-up method for classifiers that need initialization.
        """
        pass

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
