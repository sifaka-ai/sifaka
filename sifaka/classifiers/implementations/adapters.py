"""
Classifier Implementation Adapters Module

This module provides adapter classes for integrating existing classifier implementations
with the Sifaka classifiers system. These adapters implement the ClassifierImplementation
interface while delegating to the existing components.

## Adapter Classes
1. **BaseAdapter**: Base class for classifier implementation adapters
2. **ModelAdapter**: Adapts model-based classifiers
3. **RuleAdapter**: Adapts rule-based classifiers
4. **HybridAdapter**: Adapts hybrid classifiers

## Usage Examples
```python
from sifaka.classifiers.implementations.adapters import ModelAdapter
from sifaka.models import Model

# Create model
model = Model.create(
    name="sentiment_model",
    description="Detects sentiment in text",
    labels=["positive", "negative", "neutral"]
)

# Create adapter
implementation = ModelAdapter(model)

# Use adapter
result = implementation.classify("This is a friendly message.")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, cast
from ...core.results import ClassificationResult
from ...utils.errors import ClassifierError
from ...utils.errors import safely_execute_component_operation as safely_execute
from ..interfaces import ClassifierImplementation

# Define type variable for label type
L = TypeVar("L")


class ImplementationError(ClassifierError):
    """Error raised when classifier implementation fails."""

    pass


class BaseAdapter(ClassifierImplementation, Generic[L]):
    """Base class for classifier implementation adapters."""

    def __init__(self, implementation: Any) -> None:
        """
        Initialize the base adapter.

        Args:
            implementation: The implementation to adapt
        """
        self._implementation = implementation

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
            if hasattr(self._implementation, "classify"):
                return self._implementation.classify(text)
            elif hasattr(self._implementation, "process"):
                return self._implementation.process(text)
            elif hasattr(self._implementation, "run"):
                return self._implementation.run(text)
            else:
                raise ImplementationError(
                    f"Unsupported implementation: {type(self._implementation).__name__}"
                )

        result = safely_execute(
            operation=classify_operation,
            component_name="base_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )
        return self._convert_result(result)

    def _convert_result(self, result: Any) -> ClassificationResult[Any, L]:
        """
        Convert an implementation result to a ClassificationResult.

        Args:
            result: The implementation result to convert

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


class ModelAdapter(BaseAdapter[L]):
    """Adapter for model-based classifiers."""

    def classify(self, text: str) -> ClassificationResult[Any, L]:
        """
        Classify the given text using the model.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """

        def classify_operation() -> Any:
            if hasattr(self._implementation, "predict"):
                return self._implementation.predict(text)
            elif hasattr(self._implementation, "classify"):
                return self._implementation.classify(text)
            elif hasattr(self._implementation, "process"):
                return self._implementation.process(text)
            else:
                raise ImplementationError(
                    f"Unsupported model: {type(self._implementation).__name__}"
                )

        result = safely_execute(
            operation=classify_operation,
            component_name="model_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )
        return self._convert_result(result)


class RuleAdapter(BaseAdapter[L]):
    """Adapter for rule-based classifiers."""

    def classify(self, text: str) -> ClassificationResult[Any, L]:
        """
        Classify the given text using the rules.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """

        def classify_operation() -> Any:
            if hasattr(self._implementation, "evaluate"):
                return self._implementation.evaluate(text)
            elif hasattr(self._implementation, "classify"):
                return self._implementation.classify(text)
            elif hasattr(self._implementation, "process"):
                return self._implementation.process(text)
            else:
                raise ImplementationError(
                    f"Unsupported rule set: {type(self._implementation).__name__}"
                )

        result = safely_execute(
            operation=classify_operation,
            component_name="rule_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )
        return self._convert_result(result)


class HybridAdapter(BaseAdapter[L]):
    """Adapter for hybrid classifiers."""

    def classify(self, text: str) -> ClassificationResult[Any, L]:
        """
        Classify the given text using the hybrid approach.

        Args:
            text: The text to classify

        Returns:
            The classification result

        Raises:
            ImplementationError: If classification fails
        """

        def classify_operation() -> Any:
            if hasattr(self._implementation, "classify"):
                return self._implementation.classify(text)
            elif hasattr(self._implementation, "process"):
                return self._implementation.process(text)
            elif hasattr(self._implementation, "run"):
                return self._implementation.run(text)
            else:
                raise ImplementationError(
                    f"Unsupported hybrid classifier: {type(self._implementation).__name__}"
                )

        result = safely_execute(
            operation=classify_operation,
            component_name="hybrid_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )
        return self._convert_result(result)
