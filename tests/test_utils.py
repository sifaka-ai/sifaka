#!/usr/bin/env python3
"""Comprehensive tests for Sifaka utility modules.

This test suite covers utility modules including factory_utils and error_handling.
"""


import pytest

from sifaka.utils.error_handling import (
    ModelError,
    SifakaError,
    StorageError,
    ValidationError,
    error_context,
    model_context,
    storage_context,
    validation_context,
)
from sifaka.utils.factory_utils import create_with_error_handling


class TestFactoryUtils:
    """Test factory utility functions."""

    def test_create_with_error_handling_success(self):
        """Test successful model creation with error handling."""

        def mock_factory(model_name, **kwargs):
            return f"MockModel({model_name})"

        result = create_with_error_handling(
            mock_factory, "TestProvider", "test-model", param1="value1"
        )

        assert result == "MockModel(test-model)"

    def test_create_with_error_handling_failure(self):
        """Test model creation failure with error handling."""

        def failing_factory(model_name, **kwargs):
            raise ValueError("Model creation failed")

        with pytest.raises(ValueError):
            create_with_error_handling(failing_factory, "TestProvider", "test-model")


class TestErrorHandling:
    """Test error handling utilities."""

    def test_sifaka_error_basic(self):
        """Test basic SifakaError functionality."""
        error = SifakaError("Test error message")
        assert str(error) == "Test error message"
        assert error.component is None
        assert error.operation is None

    def test_sifaka_error_with_context(self):
        """Test SifakaError with context information."""
        error = SifakaError("Test error", component="TestComponent", operation="test_operation")
        assert error.component == "TestComponent"
        assert error.operation == "test_operation"

    def test_model_error(self):
        """Test ModelError functionality."""
        error = ModelError("Model failed", model_name="test-model")
        assert "Model failed" in str(error)
        assert error.model_name == "test-model"

    def test_validation_error(self):
        """Test ValidationError functionality."""
        error = ValidationError("Validation failed", validator_name="test-validator")
        assert "Validation failed" in str(error)
        assert error.validator_name == "test-validator"

    def test_storage_error(self):
        """Test StorageError functionality."""
        error = StorageError("Storage failed", storage_type="test-storage")
        assert "Storage failed" in str(error)
        assert error.storage_type == "test-storage"

    def test_error_context_manager(self):
        """Test error context manager."""
        with pytest.raises(SifakaError) as exc_info:
            with error_context(
                component="TestComponent",
                operation="test_operation",
                error_class=SifakaError,
                message_prefix="Test failed",
            ):
                raise ValueError("Original error")

        error = exc_info.value
        assert error.component == "TestComponent"
        assert error.operation == "test_operation"
        assert "Test failed" in str(error)

    def test_validation_context_manager(self):
        """Test validation context manager."""
        with pytest.raises(ValidationError) as exc_info:
            with validation_context("TestValidator", "validation", "Validation failed"):
                raise ValueError("Validation error")

        error = exc_info.value
        assert error.validator_name == "TestValidator"

    def test_model_context_manager(self):
        """Test model context manager."""
        with pytest.raises(ModelError) as exc_info:
            with model_context("test-model", "generation", "Generation failed"):
                raise ValueError("Model error")

        error = exc_info.value
        assert error.model_name == "test-model"

    def test_storage_context_manager(self):
        """Test storage context manager."""
        with pytest.raises(StorageError) as exc_info:
            with storage_context("test-storage", "save", "Save failed"):
                raise ValueError("Storage error")

        error = exc_info.value
        assert error.storage_type == "test-storage"

    def test_context_manager_success(self):
        """Test context manager with successful operation."""
        with error_context("TestComponent", "test_operation", SifakaError, "Test failed"):
            # Should not raise any error
            result = "success"

        assert result == "success"


class TestUtilsIntegration:
    """Test utility integration and common functionality."""

    def test_error_handling_with_factory_utils(self):
        """Test error handling integration with factory utils."""

        def failing_factory(model_name, **kwargs):
            raise ValueError("Factory failed")

        with pytest.raises(ValueError):
            create_with_error_handling(failing_factory, "TestProvider", "test-model")

    def test_multiple_error_contexts(self):
        """Test nested error contexts."""
        with pytest.raises(ModelError):
            with model_context("test-model", "generation", "Generation failed"):
                with error_context("SubComponent", "sub_operation", ValueError, "Sub failed"):
                    raise RuntimeError("Inner error")

    def test_error_context_with_metadata(self):
        """Test error context with additional metadata."""
        with pytest.raises(SifakaError) as exc_info:
            with error_context(
                component="TestComponent", operation="test_operation", metadata={"custom": "data"}
            ) as context:
                context.metadata["runtime"] = "info"
                raise ValueError("Test error")

        error = exc_info.value
        assert error.metadata["custom"] == "data"
        assert error.metadata["runtime"] == "info"

    def test_comprehensive_error_workflow(self):
        """Test a comprehensive error handling workflow."""

        def complex_operation():
            with model_context("gpt-4", "generation", "Model generation failed"):
                with validation_context("LengthValidator", "validation", "Validation failed"):
                    with storage_context("redis", "save", "Storage failed"):
                        # Simulate a complex failure
                        raise ConnectionError("Database connection lost")

        with pytest.raises(StorageError) as exc_info:
            complex_operation()

        error = exc_info.value
        assert "Storage failed" in str(error)
        assert "Database connection lost" in str(error)

    def test_error_formatting(self):
        """Test error message formatting."""
        error = SifakaError(
            "Base error message",
            component="TestComponent",
            operation="test_operation",
            suggestions=["Suggestion 1", "Suggestion 2"],
        )

        error_str = str(error)
        assert "[TestComponent]" in error_str
        assert "(during test_operation)" in error_str
        assert "Suggestion 1" in error_str
        assert "Suggestion 2" in error_str

    def test_error_inheritance(self):
        """Test error class inheritance."""
        # All custom errors should inherit from SifakaError
        assert issubclass(ModelError, SifakaError)
        assert issubclass(ValidationError, SifakaError)
        assert issubclass(StorageError, SifakaError)

        # Test that they can be caught as SifakaError
        with pytest.raises(SifakaError):
            raise ModelError("Model error")

        with pytest.raises(SifakaError):
            raise ValidationError("Validation error")

        with pytest.raises(SifakaError):
            raise StorageError("Storage error")

    def test_factory_utils_logging(self):
        """Test factory utils logging functionality."""

        def mock_factory(model_name, **kwargs):
            return f"Model({model_name})"

        # Should not raise any errors
        result = create_with_error_handling(mock_factory, "TestProvider", "test-model")

        assert result == "Model(test-model)"

    def test_error_context_cleanup(self):
        """Test error context cleanup and resource management."""
        cleanup_called = False

        def cleanup_function():
            nonlocal cleanup_called
            cleanup_called = True

        try:
            with error_context("TestComponent", "test_operation"):
                try:
                    raise ValueError("Test error")
                finally:
                    cleanup_function()
        except SifakaError:
            pass

        assert cleanup_called
