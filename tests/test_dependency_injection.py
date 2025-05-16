"""
Tests for the dependency injection system.

This module tests the dependency injection system in Sifaka.
"""

import pytest
from unittest.mock import Mock, patch

from sifaka.chain import Chain
from sifaka.interfaces import Model, Validator, Improver
from sifaka.registry import (
    register_model,
    register_validator,
    register_improver,
)
from sifaka.results import ValidationResult, ImprovementResult


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def generate(self, prompt, **options):
        return f"Generated text for: {prompt}"

    def count_tokens(self, text):
        return len(text.split())


class MockValidator:
    """Mock validator for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def validate(self, text):
        return ValidationResult(
            passed=True, message="Validation passed", details={"source": "mock_validator"}
        )


class MockImprover:
    """Mock improver for testing."""

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def improve(self, text):
        improved_text = f"Improved: {text}"
        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=True,
            message="Improved by mock improver",
            details={"source": "mock_improver"},
        )


@pytest.fixture
def register_components():
    """Fixture for registering test components."""

    # Register a model factory
    @register_model("test_di_model")
    def create_test_model(model_name, **options):
        return MockModel(model_name, **options)

    # Register a validator factory
    @register_validator("test_di_validator")
    def create_test_validator(**options):
        return MockValidator(**options)

    # Register an improver factory
    @register_improver("test_di_improver")
    def create_test_improver(model, **options):
        return MockImprover(model, **options)

    yield


def test_chain_with_string_model():
    """Test creating a chain with a string-based model specification."""
    # Create a model
    model = MockModel("test-model")

    # Create a chain with the model
    chain = Chain()
    chain.with_model(model)
    chain.with_prompt("Test prompt")

    # Run the chain
    result = chain.run()

    # Check that the chain used the correct model
    assert result.text == "Generated text for: Test prompt"


def test_chain_with_model_instance():
    """Test creating a chain with a model instance."""
    # Create a model
    model = MockModel("test-model")

    # Create a chain with the model
    chain = Chain()
    chain.with_model(model)
    chain.with_prompt("Test prompt")

    # Run the chain
    result = chain.run()

    # Check that the chain used the correct model
    assert result.text == "Generated text for: Test prompt"


def test_chain_with_custom_model_factory():
    """Test creating a chain with a custom model factory."""

    # Create a custom model factory
    def custom_model_factory(provider, model_name, **options):
        return MockModel(f"custom-{model_name}", **options)

    # Create a chain with the custom model factory
    chain = Chain(model_factory=custom_model_factory)

    # Create a model directly
    model = MockModel("test-model")
    chain.with_model(model)
    chain.with_prompt("Test prompt")

    # Run the chain
    result = chain.run()

    # Check that the chain used the custom model factory
    assert result.text == "Generated text for: Test prompt"


def test_chain_with_validator():
    """Test creating a chain with a validator."""
    # Create a model
    model = MockModel("test-model")

    # Create a chain
    chain = Chain()
    chain.with_model(model)
    chain.with_prompt("Test prompt")

    # Create a validator and add it to the chain
    validator = MockValidator()
    chain.validate_with(validator)

    # Run the chain
    result = chain.run()

    # Check that the validator was used
    assert len(result.validation_results) == 1
    assert result.validation_results[0].passed
    assert result.validation_results[0].message == "Validation passed"
    assert result.validation_results[0].details["source"] == "mock_validator"


def test_chain_with_improver():
    """Test creating a chain with an improver."""
    # Create a model
    model = MockModel("test-model")

    # Create a chain
    chain = Chain()
    chain.with_model(model)
    chain.with_prompt("Test prompt")

    # Create an improver and add it to the chain
    improver = MockImprover(model)
    chain.improve_with(improver)

    # Run the chain
    result = chain.run()

    # Check that the improver was used
    assert len(result.improvement_results) == 1
    assert result.improvement_results[0].changes_made
    assert result.improvement_results[0].message == "Improved by mock improver"
    assert result.improvement_results[0].details["source"] == "mock_improver"
    assert result.improvement_results[0].original_text == "Generated text for: Test prompt"
    assert (
        result.improvement_results[0].improved_text == "Improved: Generated text for: Test prompt"
    )


def test_chain_with_validator_and_improver():
    """Test creating a chain with both a validator and an improver."""
    # Create a model
    model = MockModel("test-model")

    # Create a chain
    chain = Chain()
    chain.with_model(model)
    chain.with_prompt("Test prompt")

    # Create a validator and an improver and add them to the chain
    validator = MockValidator()
    improver = MockImprover(model)
    chain.validate_with(validator)
    chain.improve_with(improver)

    # Run the chain
    result = chain.run()

    # Check that both the validator and improver were used
    assert len(result.validation_results) == 1
    assert result.validation_results[0].passed
    assert result.validation_results[0].message == "Validation passed"

    assert len(result.improvement_results) == 1
    assert result.improvement_results[0].changes_made
    assert result.improvement_results[0].message == "Improved by mock improver"

    # Check that the text was improved
    assert result.text == "Improved: Generated text for: Test prompt"
