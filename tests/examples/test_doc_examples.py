"""
Tests for examples in docstrings.

This module tests the examples provided in docstrings to ensure they stay
current with API changes.
"""

import pytest
from typing import Dict, Any, List

from sifaka.adapters.rules.base import BaseAdapter, Adaptable, create_adapter
from sifaka.adapters.rules.classifier import (
    ClassifierAdapter,
    create_classifier_adapter,
    create_classifier_rule,
    ClassifierRule
)
from sifaka.rules.base import Rule, RuleResult
from sifaka.classifiers.base import ClassificationResult


class TestBaseAdapterExamples:
    """Tests for examples in the BaseAdapter docstrings."""

    def test_basic_adapter_example(self):
        """Test the basic adapter example from the module docstring."""
        # Define a simple component that implements Adaptable
        class MyComponent:
            @property
            def name(self) -> str:
                return "my_component"

            @property
            def description(self) -> str:
                return "A component that can classify text"

            def process(self, text: str) -> bool:
                # For this example, we'll consider text valid if it has > 10 chars
                return len(text) > 10

        # Define an adapter for the component
        class MyAdapter(BaseAdapter[str, MyComponent]):
            def validate(self, input_text: str, **kwargs) -> RuleResult:
                # Handle empty text first
                empty_result = self.handle_empty_text(input_text)
                if empty_result:
                    return empty_result

                # Use the adaptee to process the text with error handling
                try:
                    result = self.adaptee.process(input_text)

                    # Convert the result to a RuleResult
                    return RuleResult(
                        passed=result,
                        message="Text validation successful" if result else "Text validation failed",
                        metadata={"component_name": self.adaptee.name}
                    )
                except Exception as e:
                    # Handle adaptee errors gracefully
                    return RuleResult(
                        passed=False,
                        message=f"Validation error: {str(e)}",
                        metadata={"error_type": type(e).__name__}
                    )

        # Create the component and adapter
        component = MyComponent()
        adapter = MyAdapter(component)

        # Test valid input
        result = adapter.validate("This text is long enough")
        assert result.passed
        assert "successful" in result.message
        assert result.metadata["component_name"] == "my_component"

        # Test invalid input
        result = adapter.validate("Too short")
        assert not result.passed
        assert "failed" in result.message

    def test_sentiment_adapter_example(self):
        """Test the sentiment adapter example from the BaseAdapter docstring."""
        # Define a simple sentiment classifier
        class SentimentClassifier:
            @property
            def name(self) -> str:
                return "sentiment_classifier"

            @property
            def description(self) -> str:
                return "Classifies sentiment of text"

            def classify(self, text: str) -> dict:
                # Simplified implementation
                positive_words = ["good", "great", "excellent"]
                negative_words = ["bad", "terrible", "awful"]

                text_lower = text.lower()
                has_positive = any(word in text_lower for word in positive_words)
                has_negative = any(word in text_lower for word in negative_words)

                if has_positive and not has_negative:
                    return {"label": "positive", "confidence": 0.9}
                elif has_negative and not has_positive:
                    return {"label": "negative", "confidence": 0.9}
                else:
                    return {"label": "neutral", "confidence": 0.7}

        # Define an adapter for the classifier
        class SentimentAdapter(BaseAdapter[str, SentimentClassifier]):
            def __init__(self, adaptee: SentimentClassifier, valid_labels: list[str]):
                super().__init__(adaptee)
                self.valid_labels = valid_labels

            def validate(self, text: str, **kwargs) -> RuleResult:
                # Handle empty text
                empty_result = self.handle_empty_text(text)
                if empty_result:
                    return empty_result

                try:
                    # Use the adaptee to classify the text
                    classification = self.adaptee.classify(text)

                    # Determine if the classification is valid
                    label = classification.get("label", "")
                    confidence = classification.get("confidence", 0.0)
                    is_valid = label in self.valid_labels

                    # Create appropriate message
                    if is_valid:
                        message = f"Text classified as '{label}' which is valid"
                    else:
                        message = f"Text classified as '{label}' which is not in valid labels: {self.valid_labels}"

                    # Convert the classification to a RuleResult
                    return RuleResult(
                        passed=is_valid,
                        message=message,
                        metadata={
                            "confidence": confidence,
                            "label": label,
                            "classifier": self.adaptee.name
                        }
                    )
                except Exception as e:
                    # Handle any errors from the adaptee
                    return RuleResult(
                        passed=False,
                        message=f"Classifier error: {str(e)}",
                        metadata={"error_type": type(e).__name__}
                    )

        # Create the classifier and adapter
        classifier = SentimentClassifier()
        adapter = SentimentAdapter(adaptee=classifier, valid_labels=["positive", "neutral"])

        # Test positive text
        result = adapter.validate("This is a great example!")
        assert result.passed
        assert result.metadata["label"] == "positive"

        # Test negative text
        result = adapter.validate("This is a terrible implementation.")
        assert not result.passed
        assert result.metadata["label"] == "negative"

    def test_create_adapter_example(self):
        """Test the create_adapter example from the docstring."""
        # Define a simple classifier (just enough to match the example)
        class SentimentClassifier:
            @property
            def name(self) -> str:
                return "sentiment_classifier"

            @property
            def description(self) -> str:
                return "Classifies sentiment of text"

            def classify(self, text: str) -> dict:
                # Just a placeholder
                return {"label": "positive", "confidence": 0.8}

        # Define a simple adapter
        class SentimentAdapter(BaseAdapter[str, SentimentClassifier]):
            def __init__(self, adaptee: SentimentClassifier, threshold: float, valid_labels: list[str]):
                super().__init__(adaptee)
                self.threshold = threshold
                self.valid_labels = valid_labels

            def validate(self, text: str, **kwargs) -> RuleResult:
                # Simple implementation for testing
                classification = self.adaptee.classify(text)
                valid = (classification["label"] in self.valid_labels and
                         classification["confidence"] >= self.threshold)
                return RuleResult(passed=valid, message="Test result")

        # Create the classifier
        classifier = SentimentClassifier()

        # Use create_adapter as shown in the example
        adapter = create_adapter(
            adapter_type=SentimentAdapter,
            adaptee=classifier,
            threshold=0.8,
            valid_labels=["positive", "neutral"]
        )

        # Verify the adapter was created correctly
        assert isinstance(adapter, SentimentAdapter)
        assert adapter.adaptee == classifier
        assert adapter.threshold == 0.8
        assert adapter.valid_labels == ["positive", "neutral"]


class TestClassifierAdapterExamples:
    """Tests for examples in the ClassifierAdapter docstrings."""

    def test_basic_classifier_adapter_example(self):
        """Test the basic classifier adapter example."""
        # Create a simplified SentimentClassifier
        class SentimentClassifier:
            @property
            def name(self) -> str:
                return "sentiment_classifier"

            @property
            def description(self) -> str:
                return "Classifies text sentiment as positive, negative, or neutral"

            @property
            def config(self) -> Any:
                return {"labels": ["positive", "negative", "neutral"]}

            def classify(self, text: str) -> ClassificationResult:
                # Simplified classification
                positive_words = ["good", "great", "excellent"]
                if any(word in text.lower() for word in positive_words):
                    return ClassificationResult(
                        label="positive",
                        confidence=0.9,
                        metadata={}
                    )
                return ClassificationResult(
                    label="neutral",
                    confidence=0.7,
                    metadata={}
                )

            def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
                return [self.classify(text) for text in texts]

        # Create classifier instance
        classifier = SentimentClassifier()

        # Create adapter as shown in example
        adapter = ClassifierAdapter(classifier, valid_labels=["positive"])

        # Test with positive text
        result = adapter.validate("This is great!")
        assert result.passed
        assert result.metadata["label"] == "positive"

        # Test with neutral text
        result = adapter.validate("This is a test.")
        assert not result.passed
        assert result.metadata["label"] == "neutral"

    def test_create_classifier_rule_example(self):
        """Test the create_classifier_rule example."""
        # Create a simplified ToxicityClassifier
        class ToxicityClassifier:
            @property
            def name(self) -> str:
                return "toxicity_classifier"

            @property
            def description(self) -> str:
                return "Classifies text toxicity"

            @property
            def config(self) -> Any:
                return {"labels": ["toxic", "safe"]}

            def classify(self, text: str) -> ClassificationResult:
                # Simplified classification - toxic if contains "toxic" word
                is_toxic = "toxic" in text.lower()
                return ClassificationResult(
                    label="toxic" if is_toxic else "safe",
                    confidence=0.9,
                    metadata={}
                )

            def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
                return [self.classify(text) for text in texts]

        # Create a classifier
        classifier = ToxicityClassifier()

        # Create a rule as shown in example
        rule = create_classifier_rule(
            classifier=classifier,
            valid_labels=["safe"],
            name="safety_rule",
            description="Ensures text is safe and non-toxic"
        )

        # Test safe text
        result = rule.validate("This is safe text")
        assert result.passed
        assert result.metadata["label"] == "safe"

        # Test toxic text
        result = rule.validate("This is toxic text")
        assert not result.passed
        assert result.metadata["label"] == "toxic"