"""
Classifiers for Sifaka.

This module provides classifier classes that categorize text into specific classes or labels.
Classifiers can be used directly or adapted into validators using the classifier validator.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ClassificationResult:
    """
    Result of a classification operation.

    This class represents the result of classifying text, including the class label,
    confidence score, and additional metadata.

    Attributes:
        label: The class label assigned to the text.
        confidence: The confidence score for the classification (0.0 to 1.0).
        metadata: Optional additional information about the classification.
    """

    label: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class Classifier(Protocol):
    """
    Protocol for classifiers that categorize text into specific classes.

    Classifiers implement this protocol to categorize text into specific classes
    with confidence scores.
    """

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text into a specific category.

        Args:
            text: The text to classify

        Returns:
            A ClassificationResult with the class label and confidence score
        """
        ...

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        This method provides a way to classify multiple texts at once,
        which may be more efficient than calling classify() multiple times.

        Args:
            texts: The list of texts to classify

        Returns:
            A list of ClassificationResults
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the classifier name.

        Returns:
            The name of the classifier
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the classifier description.

        Returns:
            The description of the classifier
        """
        ...


# Import classifier implementations
from sifaka.classifiers.bias import BiasClassifier
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.classifiers.spam import SpamClassifier
from sifaka.classifiers.toxicity import ToxicityClassifier

__all__ = [
    "Classifier",
    "ClassificationResult",
    "BiasClassifier",
    "SentimentClassifier",
    "ToxicityClassifier",
    "SpamClassifier",
    "ProfanityClassifier",
    "LanguageClassifier",
]
