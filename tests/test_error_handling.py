"""
Tests for the error handling utilities.

This module contains tests for the error handling utilities in the Sifaka framework.
"""

import logging
import pytest
from typing import Any, Dict, List, Optional

from sifaka.utils.error_handling import (
    format_error_message,
    log_error,
    convert_exception,
    error_context,
    with_error_handling,
    validation_context,
    improvement_context,
    model_context,
    retrieval_context,
    critic_context,
    chain_context,
)
from sifaka.errors import (
    SifakaError,
    ValidationError,
    ImproverError,
    ModelError,
    ChainError,
    RetrieverError,
)


class TestFormatErrorMessage:
    """Tests for the format_error_message function."""

    def test_basic_message(self) -> None:
        """Test formatting a basic error message."""
        message = "An error occurred"
        formatted = format_error_message(message)
        assert formatted == message

    def test_with_component(self) -> None:
        """Test formatting an error message with a component."""
        message = "An error occurred"
        component = "TestComponent"
        formatted = format_error_message(message, component=component)
        assert formatted == f"[{component}] {message}"

    def test_with_operation(self) -> None:
        """Test formatting an error message with an operation."""
        message = "An error occurred"
        operation = "test_operation"
        formatted = format_error_message(message, operation=operation)
        assert formatted == f"{message} (during {operation})"

    def test_with_suggestions(self) -> None:
        """Test formatting an error message with suggestions."""
        message = "An error occurred"
        suggestions = ["Try this", "Try that"]
        formatted = format_error_message(message, suggestions=suggestions)
        assert formatted == f"{message}. Suggestions: Try this; Try that"

    def test_with_all_parameters(self) -> None:
        """Test formatting an error message with all parameters."""
        message = "An error occurred"
        component = "TestComponent"
        operation = "test_operation"
        suggestions = ["Try this", "Try that"]
        formatted = format_error_message(
            message, component=component, operation=operation, suggestions=suggestions
        )
        assert (
            formatted
            == f"[{component}] {message} (during {operation}). Suggestions: Try this; Try that"
        )


class TestConvertException:
    """Tests for the convert_exception function."""

    def test_basic_conversion(self) -> None:
        """Test basic exception conversion."""
        original = ValueError("Original error")
        converted = convert_exception(original, RuntimeError)
        assert isinstance(converted, RuntimeError)
        assert str(converted) == "Original error"

    def test_with_message_prefix(self) -> None:
        """Test exception conversion with a message prefix."""
        original = ValueError("Original error")
        converted = convert_exception(original, RuntimeError, message_prefix="Prefix")
        assert isinstance(converted, RuntimeError)
        assert str(converted) == "Prefix: Original error"

    def test_to_sifaka_error(self) -> None:
        """Test conversion to a SifakaError."""
        original = ValueError("Original error")
        converted = convert_exception(
            original,
            SifakaError,
            component="TestComponent",
            operation="test_operation",
            suggestions=["Try this"],
            metadata={"key": "value"},
        )
        assert isinstance(converted, SifakaError)
        # The string representation includes the formatted message with component, operation, and suggestions
        assert "Original error" in str(converted)
        assert "TestComponent" in str(converted)
        assert "test_operation" in str(converted)
        assert "Try this" in str(converted)
        assert converted.component == "TestComponent"
        assert converted.operation == "test_operation"
        assert converted.suggestions == ["Try this"]
        assert converted.metadata == {"key": "value"}


class TestErrorContext:
    """Tests for the error_context context manager."""

    def test_no_error(self) -> None:
        """Test error_context when no error occurs."""
        with error_context():
            pass  # No error

    def test_with_error(self) -> None:
        """Test error_context when an error occurs."""
        with pytest.raises(SifakaError):
            with error_context():
                raise ValueError("Test error")

    def test_with_component_and_operation(self) -> None:
        """Test error_context with component and operation."""
        with pytest.raises(SifakaError) as excinfo:
            with error_context(component="TestComponent", operation="test_operation"):
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "TestComponent"
        assert error.operation == "test_operation"

    def test_with_custom_error_class(self) -> None:
        """Test error_context with a custom error class."""
        with pytest.raises(ValidationError):
            with error_context(error_class=ValidationError):
                raise ValueError("Test error")

    def test_with_metadata(self) -> None:
        """Test error_context with metadata."""
        with pytest.raises(SifakaError) as excinfo:
            with error_context(metadata={"key": "value"}) as context:
                context.metadata["another_key"] = "another_value"
                raise ValueError("Test error")

        error = excinfo.value
        assert error.metadata == {"key": "value", "another_key": "another_value"}


class TestWithErrorHandling:
    """Tests for the with_error_handling decorator."""

    def test_no_error(self) -> None:
        """Test with_error_handling when no error occurs."""

        @with_error_handling()
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"

    def test_with_error(self) -> None:
        """Test with_error_handling when an error occurs."""

        @with_error_handling()
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(SifakaError):
            test_function()

    def test_with_component_and_operation(self) -> None:
        """Test with_error_handling with component and operation."""

        @with_error_handling(component="TestComponent", operation="test_operation")
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(SifakaError) as excinfo:
            test_function()

        error = excinfo.value
        assert error.component == "TestComponent"
        assert error.operation == "test_operation"


class TestSpecializedContextManagers:
    """Tests for specialized context managers."""

    def test_validation_context(self) -> None:
        """Test validation_context."""
        with pytest.raises(ValidationError) as excinfo:
            with validation_context(validator_name="TestValidator"):
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Validator"
        assert error.metadata["validator_name"] == "TestValidator"

    def test_validation_context_with_suggestions(self) -> None:
        """Test validation_context with suggestions."""
        suggestions = ["Check input format", "Verify data types"]
        with pytest.raises(ValidationError) as excinfo:
            with validation_context(
                validator_name="TestValidator", suggestions=suggestions, operation="validate_text"
            ):
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Validator"
        assert error.metadata["validator_name"] == "TestValidator"
        assert error.suggestions == suggestions
        assert error.operation == "validate_text"
        assert "validate_text" in str(error)
        for suggestion in suggestions:
            assert suggestion in str(error)

    def test_improvement_context(self) -> None:
        """Test improvement_context."""
        with pytest.raises(ImproverError) as excinfo:
            with improvement_context(improver_name="TestImprover"):
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Improver"
        assert error.metadata["improver_name"] == "TestImprover"

    def test_improvement_context_with_metadata(self) -> None:
        """Test improvement_context with additional metadata."""
        with pytest.raises(ImproverError) as excinfo:
            with improvement_context(
                improver_name="TestImprover", metadata={"text_length": 100, "attempt": 2}
            ):
                # The specialized context managers don't yield the context object
                # This is a design issue in the implementation
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Improver"
        assert error.metadata["improver_name"] == "TestImprover"
        assert error.metadata["text_length"] == 100
        assert error.metadata["attempt"] == 2

    def test_model_context(self) -> None:
        """Test model_context."""
        with pytest.raises(ModelError) as excinfo:
            with model_context(model_name="TestModel"):
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Model"
        assert error.metadata["model_name"] == "TestModel"

    def test_model_context_with_api_error(self) -> None:
        """Test model_context with an API error."""
        with pytest.raises(ModelError) as excinfo:
            with model_context(
                model_name="gpt-4",
                operation="generate",
                message_prefix="API request failed",
                suggestions=["Check your API key", "Verify network connection"],
            ):
                raise ConnectionError("Connection refused")

        error = excinfo.value
        assert error.component == "Model"
        assert error.metadata["model_name"] == "gpt-4"
        assert error.operation == "generate"
        assert "API request failed" in str(error)
        assert "Connection refused" in str(error)
        # Check for suggestions we provided
        assert any("API key" in suggestion for suggestion in error.suggestions)

    def test_retrieval_context(self) -> None:
        """Test retrieval_context."""
        with pytest.raises(RetrieverError) as excinfo:
            with retrieval_context(retriever_name="TestRetriever"):
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Retriever"
        assert error.metadata["retriever_name"] == "TestRetriever"

    def test_critic_context(self) -> None:
        """Test critic_context."""
        with pytest.raises(ImproverError) as excinfo:
            with critic_context(critic_name="TestCritic"):
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Critic"
        assert error.metadata["critic_name"] == "TestCritic"

    def test_chain_context(self) -> None:
        """Test chain_context."""
        with pytest.raises(ChainError) as excinfo:
            with chain_context():
                raise ValueError("Test error")

        error = excinfo.value
        assert error.component == "Chain"

    def test_chain_context_with_complex_metadata(self) -> None:
        """Test chain_context with complex metadata."""
        with pytest.raises(ChainError) as excinfo:
            with chain_context(
                operation="run",
                metadata={
                    "model": "gpt-4",
                    "validators": ["length", "prohibited_content"],
                    "critics": ["reflexion"],
                    "config": {"temperature": 0.7, "max_tokens": 500},
                    # Include all metadata upfront since we can't modify it during execution
                    "generation_time": 2.5,
                    "validation_results": [
                        {"name": "length", "passed": True},
                        {"name": "prohibited_content", "passed": False},
                    ],
                },
            ):
                # The specialized context managers don't yield the context object
                # This is a design issue in the implementation
                raise ValueError("Validation failed")

        error = excinfo.value
        assert error.component == "Chain"
        assert error.operation == "run"
        assert error.metadata["model"] == "gpt-4"
        assert error.metadata["validators"] == ["length", "prohibited_content"]
        assert error.metadata["critics"] == ["reflexion"]
        assert error.metadata["config"] == {"temperature": 0.7, "max_tokens": 500}
        assert error.metadata["generation_time"] == 2.5
        assert len(error.metadata["validation_results"]) == 2
        assert error.metadata["validation_results"][1]["passed"] is False
