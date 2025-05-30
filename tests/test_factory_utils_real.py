#!/usr/bin/env python3
"""Tests for Sifaka factory utilities.

This test suite covers the factory utility functions used
throughout the Sifaka framework.
"""

from unittest.mock import Mock, patch

import pytest

from sifaka.utils.factory_utils import create_with_error_handling


class TestFactoryUtils:
    """Test factory utility functions."""

    def test_create_with_error_handling_success(self):
        """Test successful model creation with error handling."""

        # Mock factory function
        def mock_factory(model_name, **kwargs):
            return f"Model({model_name}, {kwargs})"

        # Test successful creation
        result = create_with_error_handling(
            mock_factory, "TestProvider", "test-model", param1="value1", param2="value2"
        )

        expected = "Model(test-model, {'param1': 'value1', 'param2': 'value2'})"
        assert result == expected

    def test_create_with_error_handling_with_logging(self):
        """Test that logging works correctly."""

        def mock_factory(model_name, **kwargs):
            return f"Model({model_name})"

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            create_with_error_handling(mock_factory, "TestProvider", "test-model")

            # Check that debug logging was called
            assert mock_logger.debug.call_count == 2

            # Check log messages
            calls = mock_logger.debug.call_args_list
            assert "Creating TestProvider model with name 'test-model'" in calls[0][0][0]
            assert (
                "Successfully created TestProvider model with name 'test-model'" in calls[1][0][0]
            )

    def test_create_with_error_handling_failure(self):
        """Test error handling when factory function fails."""

        def failing_factory(model_name, **kwargs):
            raise ValueError("Factory failed")

        with patch("sifaka.utils.factory_utils.log_error") as mock_log_error:
            with pytest.raises(ValueError, match="Factory failed"):
                create_with_error_handling(failing_factory, "TestProvider", "test-model")

            # Check that error was logged
            mock_log_error.assert_called_once()

            # Check log_error call arguments
            call_args = mock_log_error.call_args
            assert isinstance(call_args[0][0], ValueError)  # The exception
            assert call_args[1]["component"] == "TestProviderModel"
            assert call_args[1]["operation"] == "creation"

    def test_create_with_error_handling_different_providers(self):
        """Test with different provider names."""

        def mock_factory(model_name, **kwargs):
            return f"Created {model_name}"

        providers = ["OpenAI", "Anthropic", "HuggingFace", "Custom"]

        for provider in providers:
            result = create_with_error_handling(mock_factory, provider, "test-model")
            assert result == "Created test-model"

    def test_create_with_error_handling_no_kwargs(self):
        """Test factory creation without additional kwargs."""

        def mock_factory(model_name):
            return f"Simple model: {model_name}"

        result = create_with_error_handling(mock_factory, "SimpleProvider", "simple-model")

        assert result == "Simple model: simple-model"

    def test_create_with_error_handling_complex_kwargs(self):
        """Test factory creation with complex kwargs."""

        def mock_factory(model_name, **kwargs):
            return {"name": model_name, "config": kwargs}

        result = create_with_error_handling(
            mock_factory,
            "ComplexProvider",
            "complex-model",
            api_key="secret",
            temperature=0.7,
            max_tokens=100,
            options={"stream": True, "debug": False},
        )

        assert result["name"] == "complex-model"
        assert result["config"]["api_key"] == "secret"
        assert result["config"]["temperature"] == 0.7
        assert result["config"]["max_tokens"] == 100
        assert result["config"]["options"]["stream"] is True

    def test_create_with_error_handling_exception_types(self):
        """Test handling of different exception types."""
        exception_types = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            Exception("Generic exception"),
        ]

        for exc in exception_types:

            def failing_factory(model_name, **kwargs):
                raise exc

            with patch("sifaka.utils.factory_utils.log_error"):
                with pytest.raises(type(exc)):
                    create_with_error_handling(failing_factory, "TestProvider", "test-model")

    def test_create_with_error_handling_logger_name(self):
        """Test that the correct logger is used."""

        def mock_factory(model_name, **kwargs):
            return "test"

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            create_with_error_handling(mock_factory, "TestProvider", "test-model")

            # Check that getLogger was called with the correct module name
            mock_get_logger.assert_called_with("sifaka.utils.factory_utils")

    def test_create_with_error_handling_return_value_preservation(self):
        """Test that return values are preserved correctly."""
        # Test with different return types
        test_cases = ["string_result", 42, [1, 2, 3], {"key": "value"}, None, True, False]

        for expected_result in test_cases:

            def mock_factory(model_name, **kwargs):
                return expected_result

            result = create_with_error_handling(mock_factory, "TestProvider", "test-model")

            assert result == expected_result
            assert type(result) == type(expected_result)

    def test_create_with_error_handling_callable_validation(self):
        """Test that the factory function must be callable."""
        # This test ensures the function expects a callable
        # The actual validation might be implicit in Python

        def valid_factory(model_name, **kwargs):
            return "valid"

        # This should work
        result = create_with_error_handling(valid_factory, "TestProvider", "test-model")
        assert result == "valid"

    def test_create_with_error_handling_model_name_parameter(self):
        """Test that model_name is passed correctly."""

        def capture_factory(model_name, **kwargs):
            return {"received_model_name": model_name, "received_kwargs": kwargs}

        test_model_names = ["gpt-4", "claude-3", "llama-2-7b", "custom-model-v1.0"]

        for model_name in test_model_names:
            result = create_with_error_handling(
                capture_factory, "TestProvider", model_name, extra_param="test"
            )

            assert result["received_model_name"] == model_name
            assert result["received_kwargs"]["extra_param"] == "test"

    def test_create_with_error_handling_integration(self):
        """Test integration with actual model-like creation."""

        class MockModel:
            def __init__(self, model_name, api_key=None, temperature=0.7):
                self.model_name = model_name
                self.api_key = api_key
                self.temperature = temperature

            def __eq__(self, other):
                return (
                    isinstance(other, MockModel)
                    and self.model_name == other.model_name
                    and self.api_key == other.api_key
                    and self.temperature == other.temperature
                )

        def model_factory(model_name, **kwargs):
            return MockModel(model_name, **kwargs)

        result = create_with_error_handling(
            model_factory, "MockProvider", "test-model", api_key="secret123", temperature=0.9
        )

        assert isinstance(result, MockModel)
        assert result.model_name == "test-model"
        assert result.api_key == "secret123"
        assert result.temperature == 0.9
