"""
Unit tests for the Guardrails adapter.

These tests verify that Guardrails validators can be properly adapted
and used within the Sifaka rule system.
"""

from unittest.mock import patch, MagicMock

import pytest

from sifaka.rules.base import RuleResult
from sifaka.adapters.rules.guardrails_adapter import (
    GuardrailsValidatorAdapter,
    GuardrailsRule,
    create_guardrails_rule,
    GUARDRAILS_AVAILABLE,
)


# Skip all tests if Guardrails is not installed
pytestmark = pytest.mark.skipif(
    not GUARDRAILS_AVAILABLE, reason="Guardrails is not installed"
)


class MockPassResult:
    """Mock for a Guardrails PassResult."""

    def __init__(self):
        self.__dict__ = {"passed": True, "metadata": {}}
        # Not all PassResult objects have the same interface, so we'll add common methods

    def __bool__(self):
        return True

    def __dict__(self):
        return self.__dict__


class MockFailResult:
    """Mock for a Guardrails FailResult."""

    def __init__(self, error_message="Validation failed"):
        self.__dict__ = {"passed": False, "metadata": {}, "error_message": error_message}
        self._error_message = error_message

    def __bool__(self):
        return False

    def get_failure_reason(self):
        return self._error_message


class MockGuardrailsValidator:
    """Mock Guardrails validator for testing."""

    def __init__(self, should_pass=True, error_message="Validation failed"):
        self.should_pass = should_pass
        self.error_message = error_message
        self.called_with = None
        self.metadata = None
        self.__class__.__name__ = "MockValidator"

    def validate(self, text, metadata=None):
        """Mock validate method that returns pass or fail based on configuration."""
        self.called_with = text
        self.metadata = metadata
        if self.should_pass:
            return MockPassResult()
        return MockFailResult(error_message=self.error_message)


class TestGuardrailsValidatorAdapter:
    """Tests for GuardrailsValidatorAdapter."""

    def test_initialization(self):
        """Test that the adapter initializes correctly."""
        validator = MockGuardrailsValidator()
        adapter = GuardrailsValidatorAdapter(validator)
        assert adapter._guardrails_validator == validator

    @patch('sifaka.adapters.rules.guardrails_adapter.PassResult', MockPassResult)
    def test_validate_pass(self):
        """Test validation that passes."""
        validator = MockGuardrailsValidator(should_pass=True)
        adapter = GuardrailsValidatorAdapter(validator)

        # Directly patch the _convert_guardrails_result method
        original_method = adapter._convert_guardrails_result

        def patched_convert(gr_result):
            return RuleResult(
                passed=True,
                message="Validation passed",
                metadata={"guardrails_metadata": gr_result.__dict__},
            )

        adapter._convert_guardrails_result = patched_convert

        result = adapter.validate("test text")

        # Restore the original method
        adapter._convert_guardrails_result = original_method

        assert result.passed is True
        assert result.message == "Validation passed"
        assert "guardrails_metadata" in result.metadata
        assert validator.called_with == "test text"

    @patch('sifaka.adapters.rules.guardrails_adapter.FailResult', MockFailResult)
    def test_validate_fail(self):
        """Test validation that fails."""
        validator = MockGuardrailsValidator(should_pass=False, error_message="Custom error")
        adapter = GuardrailsValidatorAdapter(validator)

        # Directly patch the _convert_guardrails_result method
        original_method = adapter._convert_guardrails_result

        def patched_convert(gr_result):
            return RuleResult(
                passed=False,
                message="Custom error",
                metadata={
                    "guardrails_metadata": gr_result.__dict__,
                    "errors": ["Custom error"]
                },
            )

        adapter._convert_guardrails_result = patched_convert

        result = adapter.validate("test text")

        # Restore the original method
        adapter._convert_guardrails_result = original_method

        assert result.passed is False
        assert result.message == "Custom error"
        assert "guardrails_metadata" in result.metadata
        assert "errors" in result.metadata
        assert result.metadata["errors"] == ["Custom error"]
        assert validator.called_with == "test text"

    def test_validate_with_empty_text(self):
        """Test validation with empty text."""
        validator = MockGuardrailsValidator()
        adapter = GuardrailsValidatorAdapter(validator)
        result = adapter.validate("")

        assert result.passed is True
        assert "empty text" in result.message.lower()
        assert result.metadata.get("reason") == "empty_input"
        # The validator should not be called with empty text
        assert validator.called_with is None

    def test_validate_with_exception(self):
        """Test validation when Guardrails validator raises an exception."""
        validator = MockGuardrailsValidator()

        # Make the validate method raise an exception
        def raise_exception(*args, **kwargs):
            raise ValueError("Test error")

        validator.validate = raise_exception
        adapter = GuardrailsValidatorAdapter(validator)
        result = adapter.validate("test text")

        assert result.passed is False
        assert "guardrails validation error" in result.message.lower()
        assert "test error" in result.message.lower()
        assert result.metadata["error_type"] == "ValueError"

    def test_metadata_passing(self):
        """Test that metadata is correctly passed to the Guardrails validator."""
        validator = MockGuardrailsValidator()
        adapter = GuardrailsValidatorAdapter(validator)
        adapter.validate("test text", test_key="test_value")

        assert validator.metadata == {"test_key": "test_value"}


class TestGuardrailsRule:
    """Tests for GuardrailsRule."""

    def test_initialization(self):
        """Test that the rule initializes correctly."""
        validator = MockGuardrailsValidator()
        rule = GuardrailsRule(validator)

        assert rule._guardrails_validator == validator
        assert isinstance(rule._adapter, GuardrailsValidatorAdapter)
        assert rule.name == f"guardrails_{validator.__class__.__name__}"
        assert "guardrails validator" in rule.description.lower()

    def test_initialization_with_custom_name(self):
        """Test initialization with custom name and description."""
        validator = MockGuardrailsValidator()
        rule = GuardrailsRule(
            validator,
            name="custom_rule",
            description="Custom description",
            rule_id="custom_id"
        )

        assert rule.name == "custom_rule"
        assert rule.description == "Custom description"
        assert rule.rule_id == "custom_id"

    @patch('sifaka.adapters.rules.guardrails_adapter.PassResult', MockPassResult)
    def test_validate(self):
        """Test validation through the rule."""
        validator = MockGuardrailsValidator(should_pass=True)
        rule = GuardrailsRule(validator, rule_id="test_rule")

        # Patch the adapter's validation method
        original_method = rule._adapter.validate

        def patched_validate(text, **kwargs):
            return RuleResult(
                passed=True,
                message="Validation passed",
                metadata={"test": "metadata"}
            )

        rule._adapter.validate = patched_validate

        result = rule.validate("test text")

        # Restore the original method
        rule._adapter.validate = original_method

        assert result.passed is True
        assert result.metadata["rule_id"] == "test_rule"

    def test_create_default_validator(self):
        """Test that _create_default_validator returns the correct adapter."""
        validator = MockGuardrailsValidator()
        rule = GuardrailsRule(validator)
        default_validator = rule._create_default_validator()

        assert isinstance(default_validator, GuardrailsValidatorAdapter)
        assert default_validator._guardrails_validator == validator


class TestCreateGuardrailsRule:
    """Tests for create_guardrails_rule factory function."""

    def test_create_guardrails_rule(self):
        """Test creating a rule using the factory function."""
        validator = MockGuardrailsValidator()
        rule = create_guardrails_rule(
            guardrails_validator=validator,
            rule_id="test_rule",
            name="test_name",
            description="test description"
        )

        assert isinstance(rule, GuardrailsRule)
        assert rule.rule_id == "test_rule"
        assert rule.name == "test_name"
        assert rule.description == "test description"

    def test_create_guardrails_rule_with_minimal_args(self):
        """Test creating a rule with minimal arguments."""
        validator = MockGuardrailsValidator()
        rule = create_guardrails_rule(guardrails_validator=validator)

        assert isinstance(rule, GuardrailsRule)
        assert rule.rule_id == rule.name  # Should default to the same
        assert "guardrails" in rule.name.lower()