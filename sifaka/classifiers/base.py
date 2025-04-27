"""
Base classes for Sifaka classifiers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, ConfigDict


class ClassificationResult(BaseModel):
    """
    Result of a classification.

    Attributes:
        label: The predicted label/class
        confidence: Confidence score for the prediction (0-1)
        metadata: Additional metadata about the classification
    """

    label: Union[str, int, float, bool]
    confidence: float
    metadata: Dict[str, Any] = {}

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
    """

    name: str
    description: str
    config: Dict[str, Any] = {}
    labels: List[Union[str, int, float, bool]]
    cache_size: int = 0
    cost: int = 1

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
