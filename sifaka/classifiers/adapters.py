"""
Classifier Adapters Module

This module provides adapter classes for integrating existing Sifaka classifiers
with the new classifiers system. These adapters implement the new interfaces while
delegating to the existing components.

## Adapter Classes
1. **ImplementationAdapter**: Adapts existing classifiers to the ClassifierImplementation interface

## Usage Examples
```python
from sifaka.classifiers.adapters import ImplementationAdapter
from sifaka.classifiers.implementations.content.toxicity import ToxicityClassifier

# Create classifier
classifier = ToxicityClassifier.create(
    name="toxicity_classifier",
    description="Detects toxic content in text",
    labels=["toxic", "non-toxic"],
    cache_size=100
)

# Create adapter
implementation = ImplementationAdapter(classifier)

# Use adapter
result = implementation.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```
"""

from typing import Any, TypeVar, Generic, cast
from sifaka.interfaces.classifier import (
    ClassifierImplementationProtocol as ClassifierImplementation,
)
from ..core.results import ClassificationResult
from ..utils.errors import ClassifierError
from ..utils.errors import safely_execute_component_operation as safely_execute

# Define type variable for label type
L = TypeVar("L")


class ImplementationError(ClassifierError):
    """Error raised when classifier implementation fails."""

    pass


class ImplementationAdapter(ClassifierImplementation, Generic[L]):
    """Adapter for existing classifiers."""

    def __init__(self, classifier: Any) -> None:
        """
        Initialize the implementation adapter.

        Args:
            classifier: The classifier to adapt
        """
        self._classifier = classifier

    def classify(self, text: str) -> ClassificationResult[Any, L]:
        """
        Classify the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """

        def classify_operation() -> Any:
            if hasattr(self._classifier, "classify"):
                return self._classifier.classify(text)
            elif hasattr(self._classifier, "process"):
                return self._classifier.process(text)
            elif hasattr(self._classifier, "run"):
                return self._classifier.run(text)
            else:
                raise ImplementationError(
                    f"Unsupported classifier: {type(self._classifier).__name__}"
                )

        result = safely_execute(
            operation=classify_operation,
            component_name="implementation_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )
        return self._convert_result(result)

    def _convert_result(self, result: Any) -> ClassificationResult[Any, L]:
        """
        Convert a classifier result to a ClassificationResult.

        Args:
            result: The classifier result to convert

        Returns:
            The converted ClassificationResult
        """
        if isinstance(result, ClassificationResult):
            return result
        label = getattr(result, "label", "unknown")
        confidence = getattr(result, "confidence", 0.0)
        metadata = getattr(result, "metadata", {})
        issues = getattr(result, "issues", [])
        suggestions = getattr(result, "suggestions", [])
        # Cast the label to the expected type to satisfy mypy
        typed_label = cast(L, label)
        return ClassificationResult(
            label=typed_label,
            confidence=confidence,
            metadata=metadata,  # No need to cast metadata, it's already Dict[str, Any]
            issues=issues,
            suggestions=suggestions,
            passed=True,
            message="Classification completed",
        )
