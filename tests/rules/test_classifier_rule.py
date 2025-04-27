"""Tests for the ClassifierRule."""

import pytest
from typing import Dict, Any, List, Protocol, runtime_checkable
from dataclasses import dataclass

from sifaka.rules.classifier_rule import (
    ClassifierRule,
    ClassifierRuleConfig,
    ClassifierValidator,
    ClassifierProtocol,
    RuleResult,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.rules.base import (
    RuleConfig,
    RuleResultHandler,
    RulePriority,
    ValidationError,
    ConfigurationError,
)


class MockClassifier:
    """Mock classifier for testing."""

    def __init__(
        self,
        name: str = "mock_classifier",
        description: str = "A mock classifier for testing",
        fixed_label: str = "positive",
        fixed_confidence: float = 0.8,
    ) -> None:
        self._name = name
        self._description = description
        self.fixed_label = fixed_label
        self.fixed_confidence = fixed_confidence

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def classify(self, text: str) -> ClassificationResult:
        """Mock classify method."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        if not text:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={"error": "empty_input"},
            )
        return ClassificationResult(
            label=self.fixed_label,
            confidence=self.fixed_confidence,
            metadata={"mock": True},
        )


class MockHandler(RuleResultHandler[RuleResult]):
    """Mock handler for testing."""

    def __init__(self) -> None:
        self.handled_results: list[RuleResult] = []
        self.continue_validation = True

    def handle_result(self, result: RuleResult) -> None:
        self.handled_results.append(result)

    def should_continue(self, result: RuleResult) -> bool:
        return self.continue_validation

    def can_handle(self, result: RuleResult) -> bool:
        return isinstance(result, RuleResult)


@pytest.fixture
def mock_classifier() -> MockClassifier:
    """Fixture for creating a mock classifier."""
    return MockClassifier()


@pytest.fixture
def mock_handler() -> MockHandler:
    """Fixture for creating a mock handler."""
    return MockHandler()


@pytest.fixture
def config() -> ClassifierRuleConfig:
    """Fixture for creating a classifier rule config."""
    return ClassifierRuleConfig(
        threshold=0.7,
        valid_labels=["positive"],
        priority=RulePriority.HIGH,
        cache_size=10,
        cost=2,
        metadata={"test": True},
    )


@pytest.fixture
def validator(mock_classifier: MockClassifier, config: ClassifierRuleConfig) -> ClassifierValidator:
    """Fixture for creating a classifier validator."""
    return ClassifierValidator(
        classifier=mock_classifier,
        validation_fn=lambda r: r.confidence >= config.threshold and r.label in config.valid_labels,
        config=config,
    )


@pytest.fixture
def rule(
    mock_classifier: MockClassifier,
    config: ClassifierRuleConfig,
    mock_handler: MockHandler,
) -> ClassifierRule:
    """Fixture for creating a classifier rule."""
    return ClassifierRule(
        name="test_rule",
        description="Test classifier rule",
        classifier=mock_classifier,
        threshold=config.threshold,
        valid_labels=config.valid_labels,
        config=config,
        result_handler=mock_handler,
    )


def test_classifier_protocol():
    """Test ClassifierProtocol implementation."""
    classifier = MockClassifier()
    assert isinstance(classifier, ClassifierProtocol)

    # Test non-compliant object
    class BadClassifier:
        def classify(self, text: str) -> str:
            return text

    bad_classifier = BadClassifier()
    assert not isinstance(bad_classifier, ClassifierProtocol)


def test_classifier_rule_config():
    """Test ClassifierRuleConfig validation and behavior."""
    # Test valid config
    config = ClassifierRuleConfig(threshold=0.7, valid_labels=["positive"])
    assert config.threshold == 0.7
    assert config.valid_labels == ["positive"]
    assert config.priority == RulePriority.MEDIUM

    # Test invalid threshold
    with pytest.raises(ConfigurationError):
        ClassifierRuleConfig(threshold=1.5)

    with pytest.raises(ConfigurationError):
        ClassifierRuleConfig(threshold=-0.1)

    # Test with_threshold
    config2 = config.with_threshold(0.8)
    assert config2.threshold == 0.8
    assert config2.valid_labels == config.valid_labels

    # Test with_labels
    config3 = config.with_labels(["negative"])
    assert config3.threshold == config.threshold
    assert config3.valid_labels == ["negative"]


def test_classifier_validator(mock_classifier: MockClassifier, config: ClassifierRuleConfig):
    """Test ClassifierValidator initialization and validation."""
    # Test valid initialization
    validator = ClassifierValidator(
        classifier=mock_classifier,
        validation_fn=lambda r: True,
        config=config,
    )
    assert validator.classifier == mock_classifier
    assert validator.config == config

    # Test invalid classifier type
    with pytest.raises(ConfigurationError):
        ClassifierValidator(
            classifier="not a classifier",  # type: ignore
            validation_fn=lambda r: True,
            config=config,
        )

    # Test validation
    result = validator.validate("test text")
    assert isinstance(result, RuleResult)
    assert "classifier_name" in result.metadata
    assert "classifier_result" in result.metadata
    assert "threshold" in result.metadata
    assert "valid_labels" in result.metadata

    # Test validation error
    validator = ClassifierValidator(
        classifier=mock_classifier,
        validation_fn=lambda r: 1 / 0,  # Force an error
        config=config,
    )
    with pytest.raises(ValidationError):
        validator.validate("test")


def test_classifier_rule_validation(rule: ClassifierRule):
    """Test ClassifierRule validation."""
    # Test valid text
    result = rule.validate("test text")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert result.message
    assert result.score == 0.8
    assert result.metadata

    # Test empty text
    result = rule.validate("")
    assert not result.passed
    assert result.score == 0.0
    assert "error" in result.metadata

    # Test invalid input type
    with pytest.raises(TypeError):
        rule.validate(123)  # type: ignore

    # Test validation with handler
    assert len(rule._result_handler.handled_results) == 2  # From previous tests
    rule._result_handler.continue_validation = False
    result = rule.validate("test text")
    assert len(rule._result_handler.handled_results) == 3
    assert not rule._result_handler.should_continue(result)


def test_classifier_rule_properties(rule: ClassifierRule):
    """Test ClassifierRule property access."""
    assert rule.classifier.name == "mock_classifier"
    assert rule.threshold == 0.7
    assert rule.valid_labels == ["positive"]


def test_classifier_rule_with_custom_validation(
    mock_classifier: MockClassifier,
    config: ClassifierRuleConfig,
):
    """Test ClassifierRule with custom validation function."""
    # Create rule with custom validation
    rule = ClassifierRule(
        name="test_rule",
        description="Test classifier rule",
        classifier=mock_classifier,
        validation_fn=lambda r: r.confidence > 0.95,
        threshold=config.threshold,
        valid_labels=config.valid_labels,
    )

    # Test with high confidence
    mock_classifier.fixed_confidence = 0.96
    result = rule.validate("test text")
    assert result.passed
    assert result.score == 0.96

    # Test with low confidence
    mock_classifier.fixed_confidence = 0.94
    result = rule.validate("test text")
    assert not result.passed
    assert result.score == 0.94


def test_classifier_rule_edge_cases(rule: ClassifierRule):
    """Test ClassifierRule edge cases."""
    # Test whitespace
    result = rule.validate("   \n\t   ")
    assert not result.passed
    assert result.score == 0.0

    # Test very long text
    long_text = "test " * 1000
    result = rule.validate(long_text)
    assert isinstance(result, RuleResult)

    # Test special characters
    special_chars = "!@#$%^&*()"
    result = rule.validate(special_chars)
    assert isinstance(result, RuleResult)


def test_consistent_results(rule: ClassifierRule):
    """Test consistency of validation results."""
    text = "test text"

    # Run validation multiple times
    results = [rule.validate(text) for _ in range(3)]

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result.passed == first_result.passed
        assert result.message == first_result.message
        assert result.score == first_result.score
        assert result.metadata == first_result.metadata
