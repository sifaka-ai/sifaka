"""
Tests for the base rule components of Sifaka.
"""

import pytest
from unittest.mock import MagicMock

from sifaka.rules.base import (
    Rule,
    RuleConfig,
    RuleResult,
    FunctionRule,
    RulePriority,
    ValidationError,
    ConfigurationError,
)


class TestRuleResult:
    """Test suite for RuleResult class."""

    def test_rule_result_initialization(self):
        """Test RuleResult initialization with valid values."""
        result = RuleResult(passed=True, message="Validation passed")
        assert result.passed is True
        assert result.message == "Validation passed"
        assert result.metadata == {}
        assert result.score is None

    def test_rule_result_with_metadata(self):
        """Test RuleResult initialization with metadata."""
        metadata = {"key": "value"}
        result = RuleResult(passed=False, message="Failed", metadata=metadata)
        assert result.passed is False
        assert result.message == "Failed"
        assert result.metadata == metadata

    def test_rule_result_with_score(self):
        """Test RuleResult initialization with a score."""
        result = RuleResult(passed=True, message="Passed", score=0.8)
        assert result.passed is True
        assert result.score == 0.8

    def test_rule_result_invalid_score(self):
        """Test RuleResult with invalid score raises ValueError."""
        with pytest.raises(ValueError):
            RuleResult(passed=True, message="Test", score=1.5)

    def test_rule_result_bool_conversion(self):
        """Test RuleResult bool conversion."""
        assert bool(RuleResult(passed=True, message="Test"))
        assert not bool(RuleResult(passed=False, message="Test"))

    def test_rule_result_failed_property(self):
        """Test RuleResult failed property."""
        assert RuleResult(passed=False, message="Test").failed
        assert not RuleResult(passed=True, message="Test").failed

    def test_rule_result_with_metadata_method(self):
        """Test with_metadata method."""
        result = RuleResult(passed=True, message="Test", metadata={"a": 1})
        new_result = result.with_metadata(b=2)
        assert new_result.metadata == {"a": 1, "b": 2}
        assert new_result.passed == result.passed
        assert new_result.message == result.message


class TestRuleConfig:
    """Test suite for RuleConfig class."""

    def test_rule_config_initialization(self):
        """Test RuleConfig initialization with default values."""
        config = RuleConfig()
        assert config.priority == RulePriority.MEDIUM
        assert config.cache_size == 0
        assert config.cost == 1
        assert config.params == {}
        assert config.metadata == {}

    def test_rule_config_with_custom_values(self):
        """Test RuleConfig initialization with custom values."""
        config = RuleConfig(
            priority=RulePriority.HIGH,
            cache_size=10,
            cost=5,
            params={"param1": "value1"},
            metadata={"meta1": "value1"},
        )
        assert config.priority == RulePriority.HIGH
        assert config.cache_size == 10
        assert config.cost == 5
        assert config.params == {"param1": "value1"}
        assert config.metadata == {"meta1": "value1"}

    def test_rule_config_negative_cache_size(self):
        """Test RuleConfig with negative cache size raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            RuleConfig(cache_size=-1)

    def test_rule_config_negative_cost(self):
        """Test RuleConfig with negative cost raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            RuleConfig(cost=-1)

    def test_rule_config_with_options(self):
        """Test with_options method."""
        config = RuleConfig(priority=RulePriority.LOW, cost=1)
        new_config = config.with_options(priority=RulePriority.HIGH, cost=5)
        assert new_config.priority == RulePriority.HIGH
        assert new_config.cost == 5

    def test_rule_config_with_params(self):
        """Test with_params method."""
        config = RuleConfig(params={"a": 1})
        new_config = config.with_params(b=2)
        assert new_config.params == {"a": 1, "b": 2}


class TestFunctionRule:
    """Test suite for FunctionRule class."""

    def test_function_rule_with_bool_function(self):
        """Test FunctionRule with a function returning a boolean."""

        def validate_func(text):
            return len(text) > 5

        rule = FunctionRule(func=validate_func, name="length_check")

        result = rule.validate("Hello, world!")
        assert result.passed is True

        result = rule.validate("Hi")
        assert result.passed is False

    def test_function_rule_with_tuple_function(self):
        """Test FunctionRule with a function returning a tuple."""

        def validate_func(text):
            passed = len(text) > 5
            message = "Text is long enough" if passed else "Text is too short"
            return passed, message

        rule = FunctionRule(
            func=validate_func,
            name="length_check",
            description="Checks if text is longer than 5 characters",
        )

        result = rule.validate("Hello, world!")
        assert result.passed is True
        assert result.message == "Text is long enough"

        result = rule.validate("Hi")
        assert result.passed is False
        assert result.message == "Text is too short"

    def test_function_rule_with_rule_result_function(self):
        """Test FunctionRule with a function returning a RuleResult."""

        def validate_func(text):
            passed = len(text) > 5
            message = "Text is long enough" if passed else "Text is too short"
            return RuleResult(passed=passed, message=message, metadata={"length": len(text)})

        rule = FunctionRule(func=validate_func, name="length_check")

        result = rule.validate("Hello, world!")
        assert result.passed is True
        assert result.message == "Text is long enough"
        assert result.metadata == {"length": 13}

        result = rule.validate("Hi")
        assert result.passed is False
        assert result.metadata == {"length": 2}

    def test_function_rule_with_tuple_metadata_function(self):
        """Test FunctionRule with a function returning a tuple with metadata."""

        def validate_func(text):
            passed = len(text) > 5
            message = "Text is long enough" if passed else "Text is too short"
            metadata = {"length": len(text)}
            return passed, message, metadata

        rule = FunctionRule(func=validate_func, name="length_check")

        result = rule.validate("Hello, world!")
        assert result.passed is True
        assert result.message == "Text is long enough"
        assert result.metadata == {"length": 13}
