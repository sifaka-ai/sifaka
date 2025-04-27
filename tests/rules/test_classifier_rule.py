"""Tests for the ClassifierRule."""

import pytest
from typing import Dict, Any, List, Union, Optional
from pydantic import BaseModel, Field, ConfigDict, ValidationError

from sifaka.rules.classifier_rule import ClassifierRule
from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.rules.base import RuleResult


class MockClassifier(Classifier):
    """Mock classifier for testing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="mock_classifier")
    description: str = Field(default="A mock classifier for testing")
    labels: List[str] = Field(default=["positive", "negative", "neutral"])
    fixed_label: str = Field(default="positive")
    fixed_confidence: float = Field(default=0.8)
    warm_up_called: bool = Field(default=False)

    def __init__(
        self,
        name: str = "mock",
        description: str = "Mock classifier",
        labels: List[str] = None,
        confidence: float = 0.9,
        **kwargs,
    ):
        """Initialize mock classifier."""
        super().__init__(name=name, description=description, **kwargs)
        self.labels = labels or ["positive", "negative"]
        self.fixed_confidence = confidence
        self.fixed_label = self.labels[0]
        self.warm_up_called = False

    def warm_up(self) -> None:
        """Record that warm_up was called."""
        self.warm_up_called = True

    def classify(self, text: str) -> ClassificationResult:
        """Return fixed classification result."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        return ClassificationResult(
            label=self.fixed_label,
            confidence=self.fixed_confidence,
            metadata={"mock": True, "text_length": len(text)},
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Return fixed classification results for each text."""
        return [self.classify(text) for text in texts]


@pytest.fixture
def mock_classifier():
    """Create a mock classifier instance."""
    return MockClassifier(
        name="mock_classifier",
        description="A mock classifier for testing",
        labels=["positive", "negative", "neutral"],
    )


@pytest.fixture
def rule(mock_classifier):
    """Create a ClassifierRule instance with default configuration."""
    return ClassifierRule(
        name="test_rule",
        description="Test classifier rule",
        classifier=mock_classifier,
        threshold=0.5,
        valid_labels=["positive", "neutral"],
    )


def test_classifier_rule_validation():
    """Test Pydantic validation for ClassifierRule fields"""
    # Test valid initialization
    classifier = MockClassifier()
    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=classifier,
        threshold=0.7,
        valid_labels=["positive"],
    )
    assert rule.threshold == 0.7
    assert rule.valid_labels == ["positive"]

    # Test invalid threshold
    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(name="test", description="test rule", classifier=classifier, threshold=1.5)
    assert "threshold" in str(exc_info.value)

    # Test invalid classifier type
    with pytest.raises(ValidationError):
        ClassifierRule(name="test", description="test rule", classifier="not a classifier")

    # Test missing required fields
    with pytest.raises(ValidationError):
        ClassifierRule(name="test", description="test rule")


def test_classifier_rule_initialization(mock_classifier):
    """Test basic initialization of ClassifierRule"""
    rule = ClassifierRule(name="test", description="test rule", classifier=mock_classifier)
    assert rule.name == "test"
    assert rule.description == "test rule"
    assert rule.classifier == mock_classifier
    assert rule.threshold == 0.5
    assert rule.valid_labels == []
    assert mock_classifier.warm_up_called


def test_classifier_rule_validation_with_default_logic(mock_classifier):
    """Test validation using default logic"""
    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=mock_classifier,
        threshold=0.8,
        valid_labels=["positive"],
    )

    # Test valid output
    result = rule.validate("test text")
    assert result.passed
    assert result.message
    assert "confidence" in result.metadata

    # Test None input
    with pytest.raises(ValueError):
        rule.validate(None)

    # Test invalid label
    mock_classifier.fixed_label = "negative"
    result = rule.validate("test text")
    assert not result.passed


def test_classifier_rule_with_custom_validation(mock_classifier):
    """Test validation using custom validation function"""

    def custom_validation(result: ClassificationResult) -> RuleResult:
        return RuleResult(
            passed=result.confidence > 0.95, message="Custom validation", metadata={"custom": True}
        )

    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=mock_classifier,
        validation_fn=custom_validation,
    )

    # Test with high confidence
    mock_classifier.fixed_confidence = 0.96
    result = rule.validate("test text")
    assert result.passed
    assert result.metadata.get("custom")

    # Test with low confidence
    mock_classifier.fixed_confidence = 0.94
    result = rule.validate("test text")
    assert not result.passed


def test_classifier_rule_metadata(mock_classifier):
    """Test metadata handling"""
    rule = ClassifierRule(name="test", description="test rule", classifier=mock_classifier)

    result = rule.validate("test text")
    assert "confidence" in result.metadata
    assert "label" in result.metadata
    assert result.metadata.get("mock")


def test_classifier_rule_edge_cases(mock_classifier):
    """Test edge cases"""
    rule = ClassifierRule(name="test", description="test rule", classifier=mock_classifier)

    # Test empty string
    result = rule.validate("")
    assert isinstance(result, RuleResult)

    # Test very long text
    result = rule.validate("a" * 1000)
    assert isinstance(result, RuleResult)

    # Test special characters
    result = rule.validate("!@#$%^&*()")
    assert isinstance(result, RuleResult)


def test_initialization(mock_classifier):
    """Test initialization of ClassifierRule."""
    # Test basic initialization
    rule = ClassifierRule(
        name="test_rule",
        description="Test classifier rule",
        classifier=mock_classifier,
    )
    assert rule.name == "test_rule"
    assert rule.description == "Test classifier rule"
    assert rule.classifier == mock_classifier
    assert rule.threshold == 0.5
    assert rule.valid_labels == []

    # Test with custom parameters
    rule = ClassifierRule(
        name="custom_rule",
        description="Custom test rule",
        classifier=mock_classifier,
        threshold=0.7,
        valid_labels=["positive"],
        cost=5,
    )
    assert rule.threshold == 0.7
    assert rule.valid_labels == ["positive"]
    assert rule.cost == 5

    # Test that classifier is warmed up during initialization
    assert mock_classifier.warm_up_called


def test_validation_with_default_logic(rule):
    """Test validation using default logic."""
    # Test passing case (high confidence, valid label)
    result = rule.validate("This is a test")
    assert result.passed
    assert "confidence: 0.80" in result.message
    assert result.metadata["confidence"] == 0.8
    assert result.metadata["label"] == "positive"

    # Test failing case (invalid label)
    rule.classifier.fixed_label = "negative"
    result = rule.validate("This should fail")
    assert not result.passed
    assert result.metadata["label"] == "negative"

    # Test failing case (low confidence)
    rule.classifier.fixed_label = "positive"
    rule.classifier.fixed_confidence = 0.3
    result = rule.validate("Low confidence test")
    assert not result.passed
    assert result.metadata["confidence"] == 0.3


def test_validation_with_custom_function(mock_classifier):
    """Test validation using custom validation function."""

    def custom_validator(result: ClassificationResult) -> RuleResult:
        return RuleResult(
            passed=result.label == "positive" and result.confidence > 0.6,
            message="Custom validation",
            metadata={"custom_key": "custom_value"},
        )

    rule = ClassifierRule(
        name="custom_rule",
        description="Rule with custom validation",
        classifier=mock_classifier,
        validation_fn=custom_validator,
    )

    # Test passing case
    result = rule.validate("Test with custom validator")
    assert result.passed
    assert result.message == "Custom validation"
    assert result.metadata["custom_key"] == "custom_value"

    # Test failing case
    mock_classifier.fixed_confidence = 0.5
    result = rule.validate("Should fail custom validation")
    assert not result.passed


def test_error_handling(rule):
    """Test error handling in validation."""
    # Test with None input
    with pytest.raises(ValueError, match="Output cannot be None"):
        rule.validate(None)

    # Test with non-string input
    with pytest.raises(ValueError, match="Output must be a string"):
        rule.validate(123)


def test_metadata_handling(rule):
    """Test that metadata is properly handled and passed through."""
    result = rule.validate("Test metadata")
    assert "classifier_metadata" in result.metadata
    assert result.metadata["classifier_metadata"]["text_length"] == len("Test metadata")


def test_threshold_edge_cases(mock_classifier):
    """Test validation with various threshold values."""
    # Test with threshold = 0
    rule = ClassifierRule(
        name="zero_threshold",
        description="Test with zero threshold",
        classifier=mock_classifier,
        threshold=0.0,
    )
    result = rule.validate("Should pass with any confidence")
    assert result.passed

    # Test with threshold = 1
    rule.threshold = 1.0
    mock_classifier.fixed_confidence = 0.99
    result = rule.validate("Should fail with less than 100% confidence")
    assert not result.passed

    # Test exact threshold match
    rule.threshold = 0.8
    mock_classifier.fixed_confidence = 0.8
    result = rule.validate("Should pass with exact threshold match")
    assert result.passed


def test_valid_labels_handling(mock_classifier):
    """Test handling of valid labels list."""
    rule = ClassifierRule(
        name="label_test",
        description="Test valid labels handling",
        classifier=mock_classifier,
        valid_labels=["positive", "neutral"],
    )

    # Test with valid label
    result = rule.validate("Should pass with valid label")
    assert result.passed

    # Test with invalid label
    mock_classifier.fixed_label = "negative"
    result = rule.validate("Should fail with invalid label")
    assert not result.passed

    # Test with empty valid_labels list
    rule.valid_labels = []
    result = rule.validate("Should pass with any label")
    assert result.passed


def test_consistent_results(rule):
    """Test that validation results are consistent for the same input."""
    text = "Test consistency"
    result1 = rule.validate(text)
    result2 = rule.validate(text)

    assert result1.passed == result2.passed
    assert result1.message == result2.message
    assert result1.metadata == result2.metadata


def test_classifier_rule_pydantic_validation():
    # Test valid initialization
    mock_classifier = MockClassifier()
    rule = ClassifierRule(
        name="test",
        description="Test rule",
        classifier=mock_classifier,
        threshold=0.5,
        valid_labels=["positive"],
    )
    assert rule.name == "test"
    assert rule.description == "Test rule"
    assert rule.threshold == 0.5
    assert rule.valid_labels == ["positive"]

    # Test invalid threshold (less than 0)
    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(
            name="test",
            description="Test rule",
            classifier=mock_classifier,
            threshold=-0.1,
            valid_labels=["positive"],
        )
    assert "threshold" in str(exc_info.value)

    # Test invalid threshold (greater than 1)
    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(
            name="test",
            description="Test rule",
            classifier=mock_classifier,
            threshold=1.1,
            valid_labels=["positive"],
        )
    assert "threshold" in str(exc_info.value)

    # Test missing required fields
    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule()
    assert "name" in str(exc_info.value)
    assert "description" in str(exc_info.value)
    assert "classifier" in str(exc_info.value)

    # Test invalid classifier type
    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(
            name="test",
            description="Test rule",
            classifier="not a classifier",
            threshold=0.5,
            valid_labels=["positive"],
        )
    assert "classifier" in str(exc_info.value)

    # Test valid_labels with mixed types
    rule = ClassifierRule(
        name="test",
        description="Test rule",
        classifier=mock_classifier,
        valid_labels=["positive", 1, 0.5, True],
    )
    assert rule.valid_labels == ["positive", 1, 0.5, True]

    # Test validation_fn type checking
    def invalid_validation_fn(result: str) -> bool:
        return True

    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(
            name="test",
            description="Test rule",
            classifier=mock_classifier,
            validation_fn=invalid_validation_fn,
        )
    assert "validation_fn" in str(exc_info.value)

    # Test validation_fn return type
    def invalid_return_validation_fn(result: ClassificationResult) -> bool:
        return True

    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(
            name="test",
            description="Test rule",
            classifier=mock_classifier,
            validation_fn=invalid_return_validation_fn,
        )
    assert "validation_fn" in str(exc_info.value)

    # Test empty string name/description
    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(
            name="",
            description="Test rule",
            classifier=mock_classifier,
        )
    assert "name" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        ClassifierRule(
            name="test",
            description="",
            classifier=mock_classifier,
        )
    assert "description" in str(exc_info.value)


def test_classifier_rule_validation_fn():
    mock_classifier = MockClassifier()

    # Test with custom validation function
    def custom_validation(result: ClassificationResult) -> bool:
        return result.confidence > 0.8

    rule = ClassifierRule(
        name="test",
        description="Test rule",
        classifier=mock_classifier,
        validation_fn=custom_validation,
    )

    # Should pass with high confidence
    result = rule.validate("test text")
    assert result.passed

    # Should fail with low confidence
    mock_classifier.fixed_confidence = 0.7
    result = rule.validate("test text")
    assert not result.passed


def test_classifier_rule_metadata():
    mock_classifier = MockClassifier()
    rule = ClassifierRule(name="test", description="Test rule", classifier=mock_classifier)

    result = rule.validate("test text")
    assert result.metadata["classification_result"].label == mock_classifier.fixed_label
    assert result.metadata["classification_result"].confidence == mock_classifier.fixed_confidence
    assert result.metadata["classification_result"].metadata["mock"] is True


def test_classifier_rule_edge_cases():
    mock_classifier = MockClassifier()
    rule = ClassifierRule(name="test", description="Test rule", classifier=mock_classifier)

    # Test empty string
    result = rule.validate("")
    assert isinstance(result, RuleResult)

    # Test None input
    with pytest.raises(ValueError):
        rule.validate(None)

    # Test very long input
    result = rule.validate("a" * 10000)
    assert isinstance(result, RuleResult)
