"""
Tests for the rules adapters module of Sifaka.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.rules.adapters import ClassifierAdapter, ClassifierRule, create_classifier_rule
from sifaka.rules.base import RuleConfig, RuleResult
from sifaka.classifiers.base import ClassificationResult


class MockClassifier:
    """Mock classifier for testing."""

    def __init__(self, threshold=0.5, model_name=None):
        self.threshold = threshold
        self.model_name = model_name
        self.name = "mock_classifier" if model_name is None else model_name
        self.description = "Mock classifier for testing"

    def classify(self, text):
        """Mock classify method."""
        # Return different results based on text content
        if "positive" in text.lower():
            return ClassificationResult(
                label="positive", confidence=0.9, metadata={"negative": 0.1}
            )
        elif "negative" in text.lower():
            return ClassificationResult(
                label="negative", confidence=0.8, metadata={"positive": 0.2}
            )
        else:
            return ClassificationResult(
                label="neutral", confidence=0.5, metadata={"positive": 0.5, "negative": 0.5}
            )


class TestClassifierAdapter:
    """Test suite for ClassifierAdapter class."""

    def test_classifier_adapter_initialization(self):
        """Test ClassifierAdapter initialization."""
        classifier = MockClassifier()
        # Update to use valid_labels instead of positive_class
        adapter = ClassifierAdapter(classifier=classifier, threshold=0.7, valid_labels=["positive"])

        assert adapter.classifier == classifier
        assert "positive" in adapter.valid_labels
        assert adapter.threshold == 0.7

    def test_classifier_adapter_validate_positive(self):
        """Test ClassifierAdapter validation with positive result."""
        classifier = MockClassifier()
        # Update to use valid_labels instead of positive_class
        adapter = ClassifierAdapter(classifier=classifier, threshold=0.7, valid_labels=["positive"])

        result = adapter.validate("This is a positive text")

        assert result.passed is True
        assert "Classification passed: positive" in result.message
        assert result.metadata["result"].confidence == 0.9
        assert result.metadata["result"].label == "positive"
        assert result.score == 0.9  # Score is now set to confidence

    def test_classifier_adapter_validate_negative(self):
        """Test ClassifierAdapter validation with negative result."""
        classifier = MockClassifier()
        # Update to use valid_labels instead of positive_class
        adapter = ClassifierAdapter(classifier=classifier, threshold=0.7, valid_labels=["positive"])

        result = adapter.validate("This is a negative text")

        assert result.passed is False  # Label is "negative", not in valid_labels
        assert "Classification failed: negative" in result.message
        assert result.metadata["result"].confidence == 0.8
        assert result.metadata["result"].label == "negative"
        assert result.score == 0.8

    def test_classifier_adapter_validate_neutral(self):
        """Test ClassifierAdapter validation with neutral result."""
        classifier = MockClassifier()
        # Update to use valid_labels instead of positive_class
        adapter = ClassifierAdapter(classifier=classifier, threshold=0.7, valid_labels=["positive"])

        result = adapter.validate("This is a neutral text")

        assert result.passed is False  # Below threshold and not in valid_labels
        assert "Classification failed: neutral" in result.message
        assert result.metadata["result"].confidence == 0.5
        assert result.metadata["result"].label == "neutral"
        assert result.score == 0.5

    def test_classifier_adapter_with_negative_class(self):
        """Test ClassifierAdapter with negative class."""
        classifier = MockClassifier()
        # Update to use valid_labels with "negative" instead of positive_class
        adapter = ClassifierAdapter(classifier=classifier, threshold=0.7, valid_labels=["negative"])

        result = adapter.validate("This is a negative text")

        assert result.passed is True
        assert "Classification passed: negative" in result.message
        assert result.metadata["result"].confidence == 0.8
        assert result.metadata["result"].label == "negative"
        assert result.score == 0.8

    def test_classifier_adapter_with_custom_threshold(self):
        """Test ClassifierAdapter with custom threshold."""
        classifier = MockClassifier()
        # Very high threshold
        adapter_high = ClassifierAdapter(
            classifier=classifier, threshold=0.95, valid_labels=["positive"]
        )

        result = adapter_high.validate("This is a positive text")
        assert result.passed is False  # 0.9 is below 0.95

        # Very low threshold
        adapter_low = ClassifierAdapter(
            classifier=classifier, threshold=0.1, valid_labels=["positive"]
        )

        result = adapter_low.validate("This is a negative text")
        assert result.passed is False  # Label is not in valid_labels

        # Test with threshold only, no label constraint
        adapter_threshold_only = ClassifierAdapter(classifier=classifier, threshold=0.7)

        result = adapter_threshold_only.validate("This is a negative text")
        assert result.passed is True  # Above threshold with no label constraint

        # Custom validation function
        def custom_validation(result):
            return result.label == "negative" and result.confidence > 0.6

        adapter_custom = ClassifierAdapter(classifier=classifier, validation_fn=custom_validation)

        result = adapter_custom.validate("This is a negative text")
        assert result.passed is True  # Matches custom validation


class TestClassifierRule:
    """Test suite for ClassifierRule class."""

    def test_classifier_rule_initialization(self):
        """Test ClassifierRule initialization."""
        classifier = MockClassifier()

        rule = ClassifierRule(
            name="test_classifier_rule",
            classifier=classifier,
            threshold=0.6,
            valid_labels=["positive"],
            rule_config=RuleConfig(params={"custom_param": "value"}),
        )

        assert rule.name == "test_classifier_rule"
        assert rule.classifier == classifier
        assert rule._threshold == 0.6
        assert "positive" in rule._valid_labels

    def test_classifier_rule_validate(self):
        """Test ClassifierRule validation."""
        classifier = MockClassifier()

        rule = ClassifierRule(
            name="test_classifier_rule",
            classifier=classifier,
            threshold=0.6,
            valid_labels=["positive"],
        )

        # Test with positive text
        result = rule.validate("This is a positive text")
        assert result.passed is True
        assert result.score == 0.9

        # Test with negative text
        result = rule.validate("This is a negative text")
        assert result.passed is False
        assert result.score == 0.8

    def test_create_classifier_rule_factory(self):
        """Test create_classifier_rule factory function."""
        classifier = MockClassifier()

        rule = create_classifier_rule(
            name="factory_rule",
            classifier=classifier,
            threshold=0.8,
            valid_labels=["positive"],
            description="A test rule created by factory function",
        )

        assert rule.name == "factory_rule"
        assert "A test rule created by factory function" in rule.description

        # Validate with threshold of 0.8
        result = rule.validate("This is a positive text")
        assert result.passed is True  # 0.9 above 0.8

        result = rule.validate("This is a neutral text")
        assert result.passed is False  # 0.5 below 0.8
