"""
Classifiers for the Sifaka library.

This package provides classifier classes that categorize text into specific classes or labels.
"""

from typing import Dict, Any, List, Optional, Protocol, Union, runtime_checkable


class ClassificationResult:
    """
    Result of a classification operation.

    This class represents the result of classifying text, including the class label,
    confidence score, and additional metadata.
    """

    def __init__(
        self,
        label: str,
        confidence: float,
        passed: bool = True,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a classification result.

        Args:
            label: The classification label assigned to the text
            confidence: Confidence score between 0.0 and 1.0
            passed: Whether the classification operation succeeded
            message: Optional message describing the classification
            metadata: Additional information about the classification
        """
        self.label = label
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0.0, 1.0]
        self.passed = passed
        self.message = message
        self.metadata = metadata or {}


@runtime_checkable
class ClassifierProtocol(Protocol):
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


# Import classifier implementations
from .toxicity import ToxicityClassifier
from .sentiment import SentimentClassifier
from .spam import SpamClassifier
from .bias import BiasClassifier

# Try to import optional classifiers
try:
    from .language import LanguageClassifier

    LANGUAGE_AVAILABLE = True
except ImportError:
    LANGUAGE_AVAILABLE = False

# Export classifiers
__all__ = [
    "ClassifierProtocol",
    "ClassificationResult",
    "ToxicityClassifier",
    "SentimentClassifier",
    "SpamClassifier",
    "BiasClassifier",
]

# Add optional classifiers if available
if LANGUAGE_AVAILABLE:
    __all__.append("LanguageClassifier")
