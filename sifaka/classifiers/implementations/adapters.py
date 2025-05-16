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
from sifaka.interfaces.classifier import (
    ClassifierImplementationProtocol as ClassifierImplementation,
)

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
            if hasattr(self._implementation, "analyze"):
                return self._implementation.analyze(text)
            elif hasattr(self._implementation, "classify"):
                return self._implementation.classify(text)
            elif hasattr(self._implementation, "process"):
                return self._implementation.process(text)
            else:
                raise ImplementationError(
                    f"Unsupported hybrid implementation: {type(self._implementation).__name__}"
                )

        result = safely_execute(
            operation=classify_operation,
            component_name="hybrid_adapter",
            component_type="ClassifierImplementation",
            error_class=ImplementationError,
        )
        return self._convert_result(result)


class ToxicityClassifierAdapter(BaseAdapter[str]):
    """Adapter for toxicity classifiers."""

    def __init__(
        self,
        name: str = "toxicity_classifier",
        description: str = "Detects toxic content",
        general_threshold: float = 0.5,
        severe_toxic_threshold: float = 0.7,
        threat_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the toxicity classifier adapter.

        Args:
            name: The classifier name
            description: The classifier description
            general_threshold: Threshold for general toxicity
            severe_toxic_threshold: Threshold for severe toxicity
            threat_threshold: Threshold for threats
            **kwargs: Additional arguments
        """
        super().__init__(None)  # We don't have a real implementation yet
        self.name = name
        self.description = description
        self.general_threshold = general_threshold
        self.severe_toxic_threshold = severe_toxic_threshold
        self.threat_threshold = threat_threshold
        self.labels = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
            "non_toxic",
        ]

    def classify(self, text: str) -> ClassificationResult[Any, str]:
        """
        Classify the toxicity of the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """
        # This is a mock implementation
        # In a real implementation, we would use a toxicity detection model
        if any(bad_word in text.lower() for bad_word in ["hate", "stupid", "idiot", "kill"]):
            return ClassificationResult(
                label="toxic",
                confidence=0.8,
                metadata={"analysis": "Contains toxic language"},
                passed=False,
                message="Text contains toxic content",
            )
        return ClassificationResult(
            label="non_toxic",
            confidence=0.9,
            metadata={"analysis": "No toxic language detected"},
            passed=True,
            message="Text is non-toxic",
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult[Any, str]]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results
        """
        return [self.classify(text) for text in texts]


class SentimentClassifierAdapter(BaseAdapter[str]):
    """Adapter for sentiment classifiers."""

    def __init__(
        self,
        name: str = "sentiment_classifier",
        description: str = "Classifies text sentiment",
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the sentiment classifier adapter.

        Args:
            name: The classifier name
            description: The classifier description
            positive_threshold: Threshold for positive sentiment
            negative_threshold: Threshold for negative sentiment
            **kwargs: Additional arguments
        """
        super().__init__(None)  # We don't have a real implementation yet
        self.name = name
        self.description = description
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.labels = ["positive", "negative", "neutral", "unknown"]

    def classify(self, text: str) -> ClassificationResult[Any, str]:
        """
        Classify the sentiment of the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """
        # This is a mock implementation
        # In a real implementation, we would use a sentiment analysis model
        if any(word in text.lower() for word in ["happy", "good", "great", "excellent"]):
            return ClassificationResult(
                label="positive",
                confidence=0.8,
                metadata={"analysis": "Positive sentiment detected"},
                passed=True,
                message="Text has positive sentiment",
            )
        elif any(word in text.lower() for word in ["sad", "bad", "awful", "terrible"]):
            return ClassificationResult(
                label="negative",
                confidence=0.8,
                metadata={"analysis": "Negative sentiment detected"},
                passed=True,
                message="Text has negative sentiment",
            )
        return ClassificationResult(
            label="neutral",
            confidence=0.6,
            metadata={"analysis": "Neutral sentiment"},
            passed=True,
            message="Text has neutral sentiment",
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult[Any, str]]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results
        """
        return [self.classify(text) for text in texts]


class ProfanityClassifierAdapter(BaseAdapter[str]):
    """Adapter for profanity classifiers."""

    def __init__(
        self,
        name: str = "profanity_classifier",
        description: str = "Detects profanity in text",
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the profanity classifier adapter.

        Args:
            name: The classifier name
            description: The classifier description
            threshold: Threshold for profanity detection
            **kwargs: Additional arguments
        """
        super().__init__(None)  # We don't have a real implementation yet
        self.name = name
        self.description = description
        self.threshold = threshold
        self.labels = ["profane", "clean"]

    def classify(self, text: str) -> ClassificationResult[Any, str]:
        """
        Classify the profanity of the given text.

        Args:
            text: The text to classify

        Returns:
            The classification result
        """
        # This is a mock implementation
        # In a real implementation, we would use a profanity detection model
        if any(word in text.lower() for word in ["damn", "hell", "crap"]):
            return ClassificationResult(
                label="profane",
                confidence=0.7,
                metadata={"analysis": "Contains profanity"},
                passed=False,
                message="Text contains profanity",
            )
        return ClassificationResult(
            label="clean",
            confidence=0.9,
            metadata={"analysis": "No profanity detected"},
            passed=True,
            message="Text is clean",
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult[Any, str]]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results
        """
        return [self.classify(text) for text in texts]
