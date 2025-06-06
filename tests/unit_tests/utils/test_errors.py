"""Comprehensive unit tests for Sifaka error handling.

This module tests the custom exception hierarchy:
- SifakaError base exception
- ValidationError for validation failures
- CritiqueError for critique failures
- GraphExecutionError for graph workflow failures
- ConfigurationError for configuration issues

Tests cover:
- Exception hierarchy and inheritance
- Error message handling and formatting
- Additional context and metadata
- Error chaining and cause tracking
- Serialization and string representation
"""

import pytest
from typing import Dict, Any, Optional

from sifaka.utils.errors import (
    SifakaError,
    ValidationError,
    CritiqueError,
    GraphExecutionError,
    ConfigurationError,
)


class TestSifakaError:
    """Test the base SifakaError exception."""

    def test_sifaka_error_basic_creation(self):
        """Test creating a basic SifakaError."""
        error = SifakaError("Test error message")

        assert str(error) == "Test error message"
        assert error.args == ("Test error message",)
        assert isinstance(error, Exception)

    def test_sifaka_error_with_context(self):
        """Test SifakaError with additional context."""
        context = {"component": "test", "operation": "validation"}
        error = SifakaError("Test error", context=context)

        assert str(error) == "Test error"
        assert error.context == context
        assert error.context["component"] == "test"
        assert error.context["operation"] == "validation"

    def test_sifaka_error_without_context(self):
        """Test SifakaError without context."""
        error = SifakaError("Test error")

        assert error.context == {}

    def test_sifaka_error_inheritance(self):
        """Test that SifakaError properly inherits from Exception."""
        error = SifakaError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, SifakaError)

    def test_sifaka_error_chaining(self):
        """Test error chaining with SifakaError."""
        original_error = ValueError("Original error")

        try:
            raise original_error
        except ValueError as e:
            try:
                raise SifakaError("Wrapped error") from e
            except SifakaError as chained_error:
                assert str(chained_error) == "Wrapped error"
                assert chained_error.__cause__ is original_error

    def test_sifaka_error_repr(self):
        """Test string representation of SifakaError."""
        error = SifakaError("Test error", context={"key": "value"})

        repr_str = repr(error)
        assert "SifakaError" in repr_str
        assert "Test error" in repr_str


class TestValidationError:
    """Test the ValidationError exception."""

    def test_validation_error_basic_creation(self):
        """Test creating a basic ValidationError."""
        error = ValidationError("Validation failed")

        assert str(error) == "Validation failed"
        assert isinstance(error, SifakaError)
        assert isinstance(error, ValidationError)

    def test_validation_error_with_validator_info(self):
        """Test ValidationError with validator-specific information."""
        context = {
            "validator_name": "length-validator",
            "expected_length": 100,
            "actual_length": 50,
            "thought_id": "test-thought-123",
        }
        error = ValidationError("Text too short", context=context)

        assert str(error) == "Text too short"
        assert error.context["validator_name"] == "length-validator"
        assert error.context["expected_length"] == 100
        assert error.context["actual_length"] == 50
        assert error.context["thought_id"] == "test-thought-123"

    def test_validation_error_inheritance(self):
        """Test ValidationError inheritance hierarchy."""
        error = ValidationError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, SifakaError)
        assert isinstance(error, ValidationError)

    def test_validation_error_from_validator_result(self):
        """Test creating ValidationError from validation context."""
        # Simulate creating error from failed validation
        validation_context = {
            "validator": "content-validator",
            "score": 0.3,
            "threshold": 0.7,
            "details": {"missing_elements": ["examples", "citations"]},
        }

        error = ValidationError(
            "Content validation failed: score 0.3 below threshold 0.7", context=validation_context
        )

        assert "Content validation failed" in str(error)
        assert error.context["validator"] == "content-validator"
        assert error.context["score"] == 0.3
        assert error.context["threshold"] == 0.7
        assert "examples" in error.context["details"]["missing_elements"]


class TestCritiqueError:
    """Test the CritiqueError exception."""

    def test_critique_error_basic_creation(self):
        """Test creating a basic CritiqueError."""
        error = CritiqueError("Critique failed")

        assert str(error) == "Critique failed"
        assert isinstance(error, SifakaError)
        assert isinstance(error, CritiqueError)

    def test_critique_error_with_critic_info(self):
        """Test CritiqueError with critic-specific information."""
        context = {
            "critic_name": "constitutional-critic",
            "thought_id": "test-thought-456",
            "iteration": 2,
            "model_name": "gpt-4",
            "error_type": "api_timeout",
        }
        error = CritiqueError("Critic API timeout", context=context)

        assert str(error) == "Critic API timeout"
        assert error.context["critic_name"] == "constitutional-critic"
        assert error.context["thought_id"] == "test-thought-456"
        assert error.context["iteration"] == 2
        assert error.context["model_name"] == "gpt-4"
        assert error.context["error_type"] == "api_timeout"

    def test_critique_error_inheritance(self):
        """Test CritiqueError inheritance hierarchy."""
        error = CritiqueError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, SifakaError)
        assert isinstance(error, CritiqueError)

    def test_critique_error_from_critic_failure(self):
        """Test creating CritiqueError from critic execution failure."""
        critic_context = {
            "critic": "reflexion-critic",
            "operation": "improve_async",
            "input_length": 500,
            "max_retries": 3,
            "attempt": 3,
        }

        error = CritiqueError("Reflexion critic failed after 3 attempts", context=critic_context)

        assert "Reflexion critic failed" in str(error)
        assert error.context["critic"] == "reflexion-critic"
        assert error.context["operation"] == "improve_async"
        assert error.context["max_retries"] == 3
        assert error.context["attempt"] == 3


class TestGraphExecutionError:
    """Test the GraphExecutionError exception."""

    def test_graph_execution_error_basic_creation(self):
        """Test creating a basic GraphExecutionError."""
        error = GraphExecutionError("Graph execution failed")

        assert str(error) == "Graph execution failed"
        assert isinstance(error, SifakaError)
        assert isinstance(error, GraphExecutionError)

    def test_graph_execution_error_with_graph_info(self):
        """Test GraphExecutionError with graph execution context."""
        context = {
            "graph_name": "SifakaWorkflow",
            "node_name": "ValidateNode",
            "thought_id": "test-thought-789",
            "iteration": 1,
            "execution_time_ms": 1500.5,
            "node_index": 2,
        }
        error = GraphExecutionError("Node execution failed", context=context)

        assert str(error) == "Node execution failed"
        assert error.context["graph_name"] == "SifakaWorkflow"
        assert error.context["node_name"] == "ValidateNode"
        assert error.context["thought_id"] == "test-thought-789"
        assert error.context["iteration"] == 1
        assert error.context["execution_time_ms"] == 1500.5
        assert error.context["node_index"] == 2

    def test_graph_execution_error_inheritance(self):
        """Test GraphExecutionError inheritance hierarchy."""
        error = GraphExecutionError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, SifakaError)
        assert isinstance(error, GraphExecutionError)

    def test_graph_execution_error_with_thought_id(self):
        """Test GraphExecutionError with thought_id property."""
        error = GraphExecutionError("Test", context={"thought_id": "test-123"})

        assert error.thought_id == "test-123"

    def test_graph_execution_error_without_thought_id(self):
        """Test GraphExecutionError without thought_id in context."""
        error = GraphExecutionError("Test")

        assert error.thought_id is None


class TestConfigurationError:
    """Test the ConfigurationError exception."""

    def test_configuration_error_basic_creation(self):
        """Test creating a basic ConfigurationError."""
        error = ConfigurationError("Invalid configuration")

        assert str(error) == "Invalid configuration"
        assert isinstance(error, SifakaError)
        assert isinstance(error, ConfigurationError)

    def test_configuration_error_with_config_info(self):
        """Test ConfigurationError with configuration context."""
        context = {
            "config_section": "validators",
            "config_key": "length_validator.min_length",
            "provided_value": -10,
            "expected_type": "positive integer",
            "config_file": "sifaka.yaml",
        }
        error = ConfigurationError("Invalid validator configuration", context=context)

        assert str(error) == "Invalid validator configuration"
        assert error.context["config_section"] == "validators"
        assert error.context["config_key"] == "length_validator.min_length"
        assert error.context["provided_value"] == -10
        assert error.context["expected_type"] == "positive integer"
        assert error.context["config_file"] == "sifaka.yaml"

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance hierarchy."""
        error = ConfigurationError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, SifakaError)
        assert isinstance(error, ConfigurationError)

    def test_configuration_error_from_invalid_model(self):
        """Test creating ConfigurationError from invalid model configuration."""
        model_context = {
            "component": "SifakaDependencies",
            "model_type": "generator",
            "model_name": "invalid-model-name",
            "available_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"],
            "provider": "openai",
        }

        error = ConfigurationError(
            "Invalid model name: invalid-model-name not found in provider openai",
            context=model_context,
        )

        assert "Invalid model name" in str(error)
        assert error.context["component"] == "SifakaDependencies"
        assert error.context["model_type"] == "generator"
        assert error.context["model_name"] == "invalid-model-name"
        assert "gpt-4" in error.context["available_models"]


class TestErrorHierarchy:
    """Test the overall error hierarchy and relationships."""

    def test_all_errors_inherit_from_sifaka_error(self):
        """Test that all custom errors inherit from SifakaError."""
        errors = [
            ValidationError("test"),
            CritiqueError("test"),
            GraphExecutionError("test"),
            ConfigurationError("test"),
        ]

        for error in errors:
            assert isinstance(error, SifakaError)
            assert isinstance(error, Exception)

    def test_error_hierarchy_specificity(self):
        """Test that errors can be caught at different levels of specificity."""
        # Test catching specific error type
        try:
            raise ValidationError("Validation failed")
        except ValidationError as e:
            assert isinstance(e, ValidationError)
            assert isinstance(e, SifakaError)

        # Test catching at SifakaError level
        try:
            raise CritiqueError("Critique failed")
        except SifakaError as e:
            assert isinstance(e, CritiqueError)
            assert isinstance(e, SifakaError)

        # Test catching at Exception level
        try:
            raise GraphExecutionError("Graph failed")
        except Exception as e:
            assert isinstance(e, GraphExecutionError)
            assert isinstance(e, SifakaError)

    def test_error_context_preservation(self):
        """Test that error context is preserved through the hierarchy."""
        context = {"test_key": "test_value", "number": 42}

        errors = [
            ValidationError("test", context=context),
            CritiqueError("test", context=context),
            GraphExecutionError("test", context=context),
            ConfigurationError("test", context=context),
        ]

        for error in errors:
            assert error.context == context
            assert error.context["test_key"] == "test_value"
            assert error.context["number"] == 42

    def test_error_chaining_through_hierarchy(self):
        """Test error chaining works through the hierarchy."""
        original = ValueError("Original error")

        try:
            raise original
        except ValueError as e:
            try:
                raise ValidationError("Validation wrapper") from e
            except ValidationError as validation_error:
                try:
                    raise GraphExecutionError("Graph wrapper") from validation_error
                except GraphExecutionError as graph_error:
                    assert graph_error.__cause__ is validation_error
                    assert validation_error.__cause__ is original
                    assert isinstance(graph_error, GraphExecutionError)
                    assert isinstance(graph_error, SifakaError)


class TestErrorUtilities:
    """Test utility functions and methods for error handling."""

    def test_error_string_representation(self):
        """Test string representations of errors."""
        error = ValidationError("Test error", context={"key": "value"})

        error_str = str(error)
        assert error_str == "Test error"

        error_repr = repr(error)
        assert "ValidationError" in error_repr
        assert "Test error" in error_repr

    def test_error_context_access(self):
        """Test accessing error context safely."""
        error_with_context = ValidationError("Test", context={"key": "value"})
        error_without_context = ValidationError("Test")

        # With context
        assert error_with_context.context.get("key") == "value"
        assert error_with_context.context.get("missing_key") is None

        # Without context
        assert error_without_context.context == {}
        assert error_without_context.context.get("any_key") is None

    def test_error_equality(self):
        """Test error equality comparison."""
        error1 = ValidationError("Same message", context={"key": "value"})
        error2 = ValidationError("Same message", context={"key": "value"})
        error3 = ValidationError("Different message", context={"key": "value"})
        error4 = CritiqueError("Same message", context={"key": "value"})

        # Note: Exception equality is based on identity, not content
        # This test documents the behavior
        assert error1 is not error2
        assert error1 is not error3
        assert error1 is not error4
