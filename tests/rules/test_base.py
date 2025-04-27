"""Tests for base rule functionality."""

import pytest
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from dataclasses import dataclass

from sifaka.rules.base import (
    Rule,
    RuleResult,
    RuleConfig,
    RuleValidator,
    RuleResultHandler,
    RulePriority,
    ValidationError,
    ConfigurationError,
    Validatable,
)


@runtime_checkable
class MockValidator(Protocol):
    """Protocol for mock validator testing."""

    def validate(self, output: str) -> RuleResult: ...
    def can_validate(self, output: str) -> bool: ...
    @property
    def validation_type(self) -> type[str]: ...


class SimpleValidator(RuleValidator[str]):
    """Simple validator for testing."""

    def validate(self, output: str, **kwargs) -> RuleResult:
        return RuleResult(
            passed=bool(output),
            message="Valid" if output else "Invalid",
            metadata={"length": len(output)},
        )

    def can_validate(self, output: str) -> bool:
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        return str


class SimpleHandler(RuleResultHandler[RuleResult]):
    """Simple handler for testing."""

    def __init__(self) -> None:
        self.handled_results: list[RuleResult] = []
        self.continue_validation = True

    def handle_result(self, result: RuleResult) -> None:
        self.handled_results.append(result)

    def should_continue(self, result: RuleResult) -> bool:
        return self.continue_validation

    def can_handle(self, result: RuleResult) -> bool:
        return isinstance(result, RuleResult)


class SimpleRule(Rule[str, RuleResult, RuleValidator[str], RuleResultHandler[RuleResult]]):
    """Simple rule implementation for testing."""

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        return self._validator.validate(output, **kwargs)


@pytest.fixture
def validator() -> SimpleValidator:
    """Fixture for creating a simple validator."""
    return SimpleValidator()


@pytest.fixture
def handler() -> SimpleHandler:
    """Fixture for creating a simple handler."""
    return SimpleHandler()


@pytest.fixture
def config() -> RuleConfig:
    """Fixture for creating a rule config."""
    return RuleConfig(
        priority=RulePriority.HIGH,
        cache_size=10,
        cost=2,
        metadata={"test": True},
    )


@pytest.fixture
def rule(validator: SimpleValidator, handler: SimpleHandler, config: RuleConfig) -> SimpleRule:
    """Fixture for creating a simple rule."""
    return SimpleRule(
        name="test_rule",
        description="Test rule",
        validator=validator,
        config=config,
        result_handler=handler,
    )


def test_rule_result():
    """Test RuleResult initialization and behavior."""
    # Test valid initialization
    result = RuleResult(passed=True, message="test")
    assert result.passed
    assert result.message == "test"
    assert not result.metadata
    assert result.score is None

    # Test with metadata and score
    result = RuleResult(
        passed=True,
        message="test",
        metadata={"key": "value"},
        score=0.8,
    )
    assert result.metadata["key"] == "value"
    assert result.score == 0.8

    # Test invalid score
    with pytest.raises(ValueError):
        RuleResult(passed=True, message="test", score=1.5)

    # Test with_metadata
    result2 = result.with_metadata(new_key="new_value")
    assert result2.metadata["key"] == "value"
    assert result2.metadata["new_key"] == "new_value"
    assert result2.passed == result.passed
    assert result2.message == result.message
    assert result2.score == result.score


def test_rule_config():
    """Test RuleConfig initialization and behavior."""
    # Test default initialization
    config = RuleConfig()
    assert config.priority == RulePriority.MEDIUM
    assert config.cache_size == 0
    assert config.cost == 1
    assert not config.metadata

    # Test custom initialization
    config = RuleConfig(
        priority=RulePriority.HIGH,
        cache_size=10,
        cost=2,
        metadata={"test": True},
    )
    assert config.priority == RulePriority.HIGH
    assert config.cache_size == 10
    assert config.cost == 2
    assert config.metadata["test"]

    # Test invalid values
    with pytest.raises(ConfigurationError):
        RuleConfig(cache_size=-1)
    with pytest.raises(ConfigurationError):
        RuleConfig(cost=-1)

    # Test with_options
    config2 = config.with_options(cache_size=20)
    assert config2.cache_size == 20
    assert config2.priority == config.priority
    assert config2.cost == config.cost
    assert config2.metadata == config.metadata


def test_rule_initialization(
    validator: SimpleValidator, handler: SimpleHandler, config: RuleConfig
):
    """Test Rule initialization."""
    # Test valid initialization
    rule = SimpleRule(
        name="test",
        description="test rule",
        validator=validator,
        config=config,
        result_handler=handler,
    )
    assert rule.name == "test"
    assert rule.description == "test rule"
    assert rule.config == config

    # Test invalid validator
    with pytest.raises(ConfigurationError):
        SimpleRule(
            name="test",
            description="test",
            validator="not a validator",  # type: ignore
            config=config,
        )

    # Test invalid handler
    with pytest.raises(ConfigurationError):
        SimpleRule(
            name="test",
            description="test",
            validator=validator,
            config=config,
            result_handler="not a handler",  # type: ignore
        )


def test_rule_validation(rule: SimpleRule):
    """Test Rule validation."""
    # Test valid input
    result = rule.validate("test")
    assert result.passed
    assert result.message == "Valid"
    assert result.metadata["length"] == 4

    # Test empty input
    result = rule.validate("")
    assert not result.passed
    assert result.message == "Invalid"
    assert result.metadata["length"] == 0

    # Test invalid input type
    with pytest.raises(TypeError):
        rule.validate(123)  # type: ignore

    # Test validation with handler
    assert len(rule._result_handler.handled_results) == 2  # From previous tests
    rule._result_handler.continue_validation = False
    result = rule.validate("test")
    assert len(rule._result_handler.handled_results) == 3
    assert not rule._result_handler.should_continue(result)


def test_rule_caching(config: RuleConfig):
    """Test Rule result caching."""
    # Create rule with caching
    rule = SimpleRule(
        name="test",
        description="test",
        validator=SimpleValidator(),
        config=config,
    )

    # Test cache hit
    result1 = rule.validate("test")
    result2 = rule.validate("test")
    assert result1 is result2  # Should be the same object due to caching

    # Test cache miss
    result3 = rule.validate("different")
    assert result3 is not result1

    # Test cache with kwargs
    result4 = rule.validate("test", extra="param")
    assert result4 is not result1  # Different kwargs should bypass cache


def test_function_rule():
    """Test FunctionRule behavior."""

    # Test with boolean function
    def bool_func(text: str) -> bool:
        return bool(text)

    rule = FunctionRule(
        func=bool_func,
        name="bool_rule",
        description="Boolean function rule",
    )
    assert rule.validate("test").passed
    assert not rule.validate("").passed

    # Test with result function
    def result_func(text: str) -> RuleResult:
        return RuleResult(passed=bool(text), message="test")

    rule = FunctionRule(
        func=result_func,
        name="result_rule",
    )
    assert rule.validate("test").passed
    assert not rule.validate("").passed

    # Test with tuple function
    def tuple_func(text: str) -> tuple[bool, str, dict]:
        return (bool(text), "test", {"length": len(text)})

    rule = FunctionRule(
        func=tuple_func,
        name="tuple_rule",
    )
    result = rule.validate("test")
    assert result.passed
    assert result.message == "test"
    assert result.metadata["length"] == 4

    # Test invalid function
    def invalid_func(text: str) -> str:
        return text

    rule = FunctionRule(
        func=invalid_func,  # type: ignore
        name="invalid_rule",
    )
    with pytest.raises(ValidationError):
        rule.validate("test")
