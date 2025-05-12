"""
Tests for the error handling module.

This module tests the functionality of the error handling module,
including exception classes, error handling functions, and safe execution functions.
"""

import pytest
from pydantic import BaseModel

from sifaka.utils.errors import (
    SifakaError,
    ValidationError,
    ConfigurationError,
    ProcessingError,
    ResourceError,
    TimeoutError,
    InputError,
    StateError,
    DependencyError,
    InitializationError,
    ComponentError,
    ChainError,
    ImproverError,
    FormatterError,
    PluginError,
    ModelError,
    RuleError,
    CriticError,
    ClassifierError,
    RetrievalError,
    handle_error,
    try_operation,
    log_error,
    handle_component_error,
    create_error_handler,
    handle_chain_error,
    handle_model_error,
    handle_rule_error,
    handle_critic_error,
    handle_classifier_error,
    handle_retrieval_error,
    ErrorResult,
    create_error_result,
    create_error_result_factory,
    create_chain_error_result,
    create_model_error_result,
    create_rule_error_result,
    create_critic_error_result,
    create_classifier_error_result,
    create_retrieval_error_result,
    try_component_operation,
    safely_execute_component_operation,
    create_safe_execution_factory,
    safely_execute_chain,
    safely_execute_model,
    safely_execute_rule,
    safely_execute_critic,
    safely_execute_classifier,
    safely_execute_retrieval,
    safely_execute_component,
)


class TestBaseErrorClasses:
    """Test base error classes."""

    def test_sifaka_error(self):
        """Test SifakaError class."""
        # Test basic error
        error = SifakaError("Test error")
        assert error.message == "Test error"
        assert error.metadata == {}
        assert str(error) == "Test error"

        # Test error with metadata
        error = SifakaError("Test error", metadata={"key": "value"})
        assert error.message == "Test error"
        assert error.metadata == {"key": "value"}
        assert str(error) == "Test error (metadata: {'key': 'value'})"

    def test_validation_error(self):
        """Test ValidationError class."""
        error = ValidationError("Validation failed")
        assert error.message == "Validation failed"
        assert isinstance(error, SifakaError)

    def test_component_error(self):
        """Test ComponentError class."""
        error = ComponentError(
            "Component error",
            component_name="TestComponent",
            component_type="test",
            error_type="test_error",
        )
        assert error.message == "Component error"
        assert error.metadata["component_name"] == "TestComponent"
        assert error.metadata["component_type"] == "test"
        assert error.metadata["error_type"] == "test_error"


class TestComponentErrorClasses:
    """Test component-specific error classes."""

    def test_chain_error(self):
        """Test ChainError class."""
        error = ChainError("Chain error")
        assert error.message == "Chain error"
        assert isinstance(error, SifakaError)

    def test_model_error(self):
        """Test ModelError class."""
        error = ModelError("Model error")
        assert error.message == "Model error"
        assert isinstance(error, SifakaError)

    def test_rule_error(self):
        """Test RuleError class."""
        error = RuleError("Rule error")
        assert error.message == "Rule error"
        assert isinstance(error, ValidationError)
        assert isinstance(error, SifakaError)


class TestErrorHandling:
    """Test error handling functions."""

    def test_handle_error(self):
        """Test handle_error function."""
        # Test with basic error
        error = ValueError("Test error")
        metadata = handle_error(error, "TestComponent")
        assert metadata["error_type"] == "ValueError"
        assert metadata["error_message"] == "Test error"
        assert metadata["component"] == "TestComponent"
        assert "traceback" in metadata

        # Test with SifakaError
        error = SifakaError("Test error", metadata={"key": "value"})
        metadata = handle_error(error, "TestComponent")
        assert metadata["error_type"] == "SifakaError"
        # The error message includes the metadata
        assert "Test error" in metadata["error_message"]
        assert "key" in metadata
        assert metadata["component"] == "TestComponent"
        assert metadata["key"] == "value"

    def test_try_operation(self):
        """Test try_operation function."""
        # Test successful operation
        result = try_operation(lambda: "success", "TestComponent")
        assert result == "success"

        # Test failed operation with default value
        result = try_operation(lambda: 1 / 0, "TestComponent", default_value="default")
        assert result == "default"

        # Test failed operation with error handler
        def error_handler(e):
            return "handled"

        result = try_operation(lambda: 1 / 0, "TestComponent", error_handler=error_handler)
        assert result == "handled"


class TestErrorResult:
    """Test ErrorResult class."""

    def test_error_result(self):
        """Test ErrorResult class."""
        result = ErrorResult(
            error_type="TestError",
            error_message="Test error",
            component_name="TestComponent",
            metadata={"key": "value"},
        )
        assert result.error_type == "TestError"
        assert result.error_message == "Test error"
        assert result.component_name == "TestComponent"
        assert result.metadata == {"key": "value"}


class TestSafeExecution:
    """Test safe execution functions."""

    def test_safely_execute_component_operation(self):
        """Test safely_execute_component_operation function."""
        # Test successful operation
        result = safely_execute_component_operation(
            lambda: "success",
            component_name="TestComponent",
            component_type="Test",
            error_class=SifakaError,
        )
        assert result == "success"

        # Test failed operation
        result = safely_execute_component_operation(
            lambda: 1 / 0,
            component_name="TestComponent",
            component_type="Test",
            error_class=SifakaError,
        )
        assert isinstance(result, ErrorResult)
        assert result.error_type == "ZeroDivisionError"
        assert result.component_name == "TestComponent"

    def test_safely_execute_chain(self):
        """Test safely_execute_chain function."""
        # Test successful operation
        result = safely_execute_chain(lambda: "success", "TestChain")
        assert result == "success"

        # Test failed operation
        result = safely_execute_chain(lambda: 1 / 0, "TestChain")
        assert isinstance(result, ErrorResult)
        assert result.error_type == "ZeroDivisionError"
        assert result.component_name == "TestChain"
