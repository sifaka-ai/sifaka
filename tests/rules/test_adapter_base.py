"""
Unit tests for the BaseAdapter class and Adaptable protocol.

These tests cover the core functionality of the BaseAdapter class
and the Adaptable protocol, which are key components of Sifaka's adapter system.
"""

import pytest
from typing import Dict, Any, Optional

from sifaka.adapters.rules.base import BaseAdapter, Adaptable, create_adapter
from sifaka.rules.base import RuleResult, ConfigurationError, ValidationError


class MockAdaptee:
    """Mock adaptee that implements the Adaptable protocol."""

    @property
    def name(self) -> str:
        return "mock_adaptee"

    @property
    def description(self) -> str:
        return "Mock adaptee for testing"

    def process(self, text: str) -> bool:
        """Process text and return a boolean result."""
        return len(text) > 10


class InvalidAdaptee:
    """Mock adaptee that does not implement the Adaptable protocol."""

    def process(self, text: str) -> bool:
        """Process text and return a boolean result."""
        return len(text) > 10


class MockAdapter(BaseAdapter[str, MockAdaptee]):
    """Mock adapter for testing BaseAdapter functionality."""

    def validate(self, input_text: str, **kwargs) -> RuleResult:
        # Handle empty text first
        empty_result = self.handle_empty_text(input_text)
        if empty_result:
            return empty_result

        # Validate using the adaptee
        try:
            result = self.adaptee.process(input_text)
            return RuleResult(
                passed=result,
                message="Validation " + ("passed" if result else "failed"),
                metadata={"adaptee_name": self.adaptee.name},
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Validation error: {str(e)}",
                metadata={"error_type": type(e).__name__},
            )


class ErrorAdapter(BaseAdapter[str, MockAdaptee]):
    """Adapter that raises an error during validation."""

    def validate(self, input_text: str, **kwargs) -> RuleResult:
        raise ValidationError("Test error")


class TestAdaptableProtocol:
    """Tests for the Adaptable protocol."""

    def test_valid_adaptee_runtime_checkable(self):
        """Test that a valid adaptee is recognized by the protocol at runtime."""
        adaptee = MockAdaptee()
        assert isinstance(adaptee, Adaptable)

    def test_invalid_adaptee_runtime_checkable(self):
        """Test that an invalid adaptee is not recognized by the protocol at runtime."""
        adaptee = InvalidAdaptee()
        assert not isinstance(adaptee, Adaptable)


class TestBaseAdapter:
    """Tests for the BaseAdapter class."""

    def test_initialization_with_valid_adaptee(self):
        """Test initialization with a valid adaptee."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        assert adapter.adaptee == adaptee

    def test_initialization_with_invalid_adaptee(self):
        """Test initialization with an invalid adaptee raises ConfigurationError."""
        adaptee = InvalidAdaptee()
        with pytest.raises(ConfigurationError) as excinfo:
            MockAdapter(adaptee)
        assert "must implement Adaptable protocol" in str(excinfo.value)

    def test_validation_type_default(self):
        """Test that the default validation_type is str."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        assert adapter.validation_type == str

    def test_handle_empty_text_with_empty_string(self):
        """Test handle_empty_text with an empty string."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        result = adapter.handle_empty_text("")

        assert result is not None
        assert result.passed
        assert "empty text" in result.message.lower()
        assert result.metadata["reason"] == "empty_input"
        assert "input_length" in result.metadata

    def test_handle_empty_text_with_whitespace(self):
        """Test handle_empty_text with whitespace."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        result = adapter.handle_empty_text("   \n\t  ")

        assert result is not None
        assert result.passed
        assert "empty text" in result.message.lower()

    def test_handle_empty_text_with_non_empty_string(self):
        """Test handle_empty_text with a non-empty string."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        result = adapter.handle_empty_text("Hello")

        assert result is None

    def test_handle_empty_text_with_non_string(self):
        """Test handle_empty_text with a non-string value."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        result = adapter.handle_empty_text(123)  # type: ignore

        assert result is None

    def test_validate_not_implemented(self):
        """Test that the base validate method raises NotImplementedError."""
        adaptee = MockAdaptee()
        adapter = BaseAdapter(adaptee)

        with pytest.raises(ValidationError) as excinfo:
            adapter.validate("test")
        assert "must implement" in str(excinfo.value)

    def test_validate_with_empty_text(self):
        """Test validation with empty text."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        result = adapter.validate("")

        assert result.passed
        assert "empty text" in result.message.lower()

    def test_validate_with_valid_input(self):
        """Test validation with valid input."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        result = adapter.validate("This is a long text that should pass validation")

        assert result.passed
        assert "passed" in result.message
        assert result.metadata["adaptee_name"] == adaptee.name

    def test_validate_with_invalid_input(self):
        """Test validation with invalid input."""
        adaptee = MockAdaptee()
        adapter = MockAdapter(adaptee)
        result = adapter.validate("Short")

        assert not result.passed
        assert "failed" in result.message
        assert result.metadata["adaptee_name"] == adaptee.name

    def test_validate_with_error(self):
        """Test validation with an error."""
        adaptee = MockAdaptee()
        adapter = ErrorAdapter(adaptee)

        with pytest.raises(ValidationError) as excinfo:
            adapter.validate("test")
        assert "Test error" in str(excinfo.value)


class TestCreateAdapter:
    """Tests for the create_adapter factory function."""

    def test_create_adapter_with_valid_inputs(self):
        """Test create_adapter with valid inputs."""
        adaptee = MockAdaptee()
        adapter = create_adapter(MockAdapter, adaptee)

        assert isinstance(adapter, MockAdapter)
        assert adapter.adaptee == adaptee

    def test_create_adapter_with_invalid_adapter_type(self):
        """Test create_adapter with an invalid adapter type."""
        adaptee = MockAdaptee()

        with pytest.raises(ConfigurationError) as excinfo:
            create_adapter(type, adaptee)  # type: ignore
        assert "must be a subclass of BaseAdapter" in str(excinfo.value)

    def test_create_adapter_with_invalid_adaptee(self):
        """Test create_adapter with an invalid adaptee."""
        adaptee = InvalidAdaptee()

        with pytest.raises(ConfigurationError) as excinfo:
            create_adapter(MockAdapter, adaptee)
        assert "must implement Adaptable protocol" in str(excinfo.value)

    def test_create_adapter_with_kwargs(self):
        """Test create_adapter with additional kwargs."""

        class ParameterizedAdapter(BaseAdapter[str, MockAdaptee]):
            def __init__(self, adaptee: MockAdaptee, parameter: str):
                super().__init__(adaptee)
                self.parameter = parameter

            def validate(self, input_text: str, **kwargs) -> RuleResult:
                return RuleResult(passed=True, message=self.parameter)

        adaptee = MockAdaptee()
        adapter = create_adapter(ParameterizedAdapter, adaptee, parameter="test_value")

        assert isinstance(adapter, ParameterizedAdapter)
        assert adapter.parameter == "test_value"

        result = adapter.validate("text")
        assert result.passed
        assert result.message == "test_value"
