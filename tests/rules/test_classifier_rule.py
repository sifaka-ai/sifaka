"""Tests for the ClassifierRule."""

import pytest
from typing import Dict, Any, List, Union, Optional, Callable
from pydantic import BaseModel, Field, ConfigDict, ValidationError

from sifaka.rules.classifier_rule import ClassifierRule
from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.rules.base import RuleResult


class MockClassifier(Classifier):
    """Mock classifier for testing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="mock_classifier")
    description: str = Field(default="A mock classifier for testing")
    labels: List[str] = Field(default_factory=lambda: ["positive", "negative", "neutral"])
    fixed_label: str = Field(default="positive")
    fixed_confidence: float = Field(default=0.8)
    warm_up_called: bool = Field(default=False)

    def warm_up(self) -> None:
        """Mock warm up method."""
        self.warm_up_called = True

    def classify(self, text: str) -> ClassificationResult:
        """Mock classify method."""
        if not text:
            raise ValueError("Text cannot be empty")
        return ClassificationResult(
            label=self.fixed_label, confidence=self.fixed_confidence, metadata={"mock": True}
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Mock batch classify method."""
        return [self.classify(text) for text in texts]


@pytest.fixture
def mock_classifier() -> MockClassifier:
    """Fixture for creating a mock classifier."""
    return MockClassifier()


@pytest.fixture
def rule(mock_classifier: MockClassifier) -> ClassifierRule:
    """Fixture for creating a classifier rule with mock classifier."""
    return ClassifierRule(
        name="test",
        description="test rule",
        classifier=mock_classifier,
        config={},
    )


def test_classifier_rule_validation():
    """Test validation and initialization of ClassifierRule."""
    classifier = MockClassifier()

    # Test valid initialization
    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=classifier,
        config={
            "threshold": 0.7,
            "valid_labels": ["positive"],
        },
    )
    assert rule.threshold == 0.7
    assert rule.valid_labels == ["positive"]

    # Test invalid threshold values
    with pytest.raises(ValueError):
        ClassifierRule(
            name="test",
            description="test rule",
            classifier=classifier,
            config={"threshold": 1.5},
        )

    with pytest.raises(ValueError):
        ClassifierRule(
            name="test",
            description="test rule",
            classifier=classifier,
            config={"threshold": -0.1},
        )

    # Test invalid classifier type
    with pytest.raises(ValueError):
        ClassifierRule(
            name="test",
            description="test rule",
            classifier="not a classifier",
            config={},
        )

    # Test missing required fields
    with pytest.raises(TypeError):
        ClassifierRule(name="test", description="test rule")

    # Test invalid validation function signature
    def invalid_validation_fn(result: str) -> bool:
        return True

    with pytest.raises(ValueError):
        ClassifierRule(
            name="test",
            description="test rule",
            classifier=classifier,
            config={"validation_fn": invalid_validation_fn},
        )

    # Test invalid validation function return type
    def invalid_return_validation_fn(result: ClassificationResult) -> str:
        return "Invalid"

    with pytest.raises(ValueError):
        ClassifierRule(
            name="test",
            description="test rule",
            classifier=classifier,
            config={"validation_fn": invalid_return_validation_fn},
        )


def test_classifier_rule_initialization(mock_classifier):
    """Test basic initialization of ClassifierRule"""
    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=mock_classifier,
        config={},
    )
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
        config={
            "threshold": 0.8,
            "valid_labels": ["positive"],
        },
    )

    # Test valid output
    result = rule.validate("test text")
    assert result.passed
    assert result.message
    assert "classification" in result.metadata

    # Test None input
    with pytest.raises(ValueError):
        rule.validate(None)

    # Test invalid label
    mock_classifier.fixed_label = "negative"
    result = rule.validate("test text")
    assert not result.passed


def test_classifier_rule_with_custom_validation(mock_classifier):
    """Test validation using custom validation function"""

    def validation_fn(result: ClassificationResult) -> bool:
        return result.confidence > 0.95

    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=mock_classifier,
        config={
            "validation_fn": validation_fn,
        },
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
    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=mock_classifier,
        config={},
    )

    result = rule.validate("test text")
    assert "classification" in result.metadata
    assert result.metadata["classification"]["label"] == "positive"
    assert result.metadata["classification"]["confidence"] == 0.8
    assert result.metadata["classification"]["metadata"]["mock"]


def test_classifier_rule_edge_cases(mock_classifier):
    """Test edge cases"""
    rule = ClassifierRule(
        name="test",
        description="test rule",
        classifier=mock_classifier,
        config={},
    )

    # Test empty string
    result = rule.validate("")
    assert not result.passed
    assert "Text cannot be empty" in result.message

    # Test very long text
    result = rule.validate("a" * 1000)
    assert isinstance(result, RuleResult)

    # Test special characters
    result = rule.validate("!@#$%^&*()")
    assert isinstance(result, RuleResult)


def test_consistent_results(rule):
    """Test that validation results are consistent."""
    text = "test text"
    result1 = rule.validate(text)
    result2 = rule.validate(text)

    assert result1.passed == result2.passed
    assert result1.message == result2.message
    assert result1.metadata == result2.metadata
