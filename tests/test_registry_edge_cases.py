"""
Tests for edge cases and error handling in the registry system.

This module tests edge cases and error handling in the registry system.
"""

import pytest
from unittest.mock import Mock, patch

from sifaka.interfaces import Model, Validator, Improver
from sifaka.registry import (
    register_model,
    register_validator,
    register_improver,
    get_model_factory,
    get_validator_factory,
    get_improver_factory,
)
from sifaka.factories import (
    create_model,
    create_model_from_string,
    create_validator,
    create_improver,
    FactoryError,
)


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
        return {"passed": True, "message": "Validation passed"}


class MockImprover:
    """Mock improver for testing."""

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def improve(self, text):
        return f"Improved: {text}", {"changes_made": True, "message": "Improved"}


def test_register_model_with_invalid_factory():
    """Test registering a model with an invalid factory function."""

    # Register a factory that doesn't return a Model
    @register_model("invalid_model")
    def invalid_factory(model_name, **options):
        return "Not a model"

    # Create a model with the invalid factory
    # Note: The registry doesn't validate the return type at registration time
    model = create_model("invalid_model", "model-name")

    # Check that the model is a string, not a Model instance
    assert isinstance(model, str)
    assert model == "Not a model"


def test_register_validator_with_invalid_factory():
    """Test registering a validator with an invalid factory function."""

    # Register a factory that doesn't return a Validator
    @register_validator("invalid_validator")
    def invalid_factory(**options):
        return "Not a validator"

    # Create a validator with the invalid factory
    # Note: The registry doesn't validate the return type at registration time
    validator = create_validator("invalid_validator")

    # Check that the validator is a string, not a Validator instance
    assert isinstance(validator, str)
    assert validator == "Not a validator"


def test_register_improver_with_invalid_factory():
    """Test registering an improver with an invalid factory function."""

    # Register a factory that doesn't return an Improver
    @register_improver("invalid_improver")
    def invalid_factory(model, **options):
        return "Not an improver"

    # Create an improver with the invalid factory
    # Note: The registry doesn't validate the return type at registration time
    improver = create_improver("invalid_improver", MockModel("model-name"))

    # Check that the improver is a string, not an Improver instance
    assert isinstance(improver, str)
    assert improver == "Not an improver"


def test_create_model_with_invalid_provider():
    """Test creating a model with an invalid provider."""
    # Try to create a model with a provider that doesn't exist
    with pytest.raises(FactoryError):
        create_model("nonexistent", "model-name")


def test_create_validator_with_invalid_provider():
    """Test creating a validator with an invalid provider."""
    # Try to create a validator with a provider that doesn't exist
    with pytest.raises(FactoryError):
        create_validator("nonexistent")


def test_create_improver_with_invalid_provider():
    """Test creating an improver with an invalid provider."""
    # Try to create an improver with a provider that doesn't exist
    with pytest.raises(FactoryError):
        create_improver("nonexistent", MockModel("model-name"))


def test_create_model_from_string_with_invalid_format():
    """Test creating a model from a string with an invalid format."""
    # Try to create a model from a string with an invalid format
    with pytest.raises(ValueError):
        create_model_from_string("invalid-format")


def test_create_model_from_string_with_invalid_provider():
    """Test creating a model from a string with an invalid provider."""
    # Try to create a model from a string with a provider that doesn't exist
    with pytest.raises(FactoryError):
        create_model_from_string("nonexistent:model-name")


def test_create_model_with_factory_error():
    """Test creating a model when the factory raises an error."""

    # Register a factory that raises an error
    @register_model("error")
    def error_factory(model_name, **options):
        raise ValueError("Factory error")

    # Try to create a model with the factory
    with pytest.raises(FactoryError):
        create_model("error", "model-name")


def test_create_validator_with_factory_error():
    """Test creating a validator when the factory raises an error."""

    # Register a factory that raises an error
    @register_validator("error")
    def error_factory(**options):
        raise ValueError("Factory error")

    # Try to create a validator with the factory
    with pytest.raises(FactoryError):
        create_validator("error")


def test_create_improver_with_factory_error():
    """Test creating an improver when the factory raises an error."""

    # Register a factory that raises an error
    @register_improver("error")
    def error_factory(model, **options):
        raise ValueError("Factory error")

    # Try to create an improver with the factory
    with pytest.raises(FactoryError):
        create_improver("error", MockModel("model-name"))


def test_register_model_with_empty_name():
    """Test registering a model with an empty name."""

    # Register a factory with an empty name
    # Note: The registry uses the empty string as the provider name
    @register_model("")
    def empty_name_factory(model_name, **options):
        return MockModel(model_name, **options)

    # The registry should use the empty string as the provider name
    factory = get_model_factory("")
    assert factory is not None

    # Create a model using the factory
    model = factory("model-name")
    assert isinstance(model, MockModel)


def test_register_validator_with_empty_name():
    """Test registering a validator with an empty name."""

    # Register a factory with an empty name
    # Note: The registry uses the empty string as the provider name
    @register_validator("")
    def empty_name_validator(**options):
        return MockValidator(**options)

    # The registry should use the empty string as the provider name
    factory = get_validator_factory("")
    assert factory is not None

    # Create a validator using the factory
    validator = factory()
    assert isinstance(validator, MockValidator)


def test_register_improver_with_empty_name():
    """Test registering an improver with an empty name."""

    # Register a factory with an empty name
    # Note: The registry uses the empty string as the provider name
    @register_improver("")
    def empty_name_improver(model, **options):
        return MockImprover(model, **options)

    # The registry should use the empty string as the provider name
    factory = get_improver_factory("")
    assert factory is not None

    # Create an improver using the factory
    improver = factory(MockModel("model-name"))
    assert isinstance(improver, MockImprover)


def test_get_model_factory_with_nonexistent_provider():
    """Test getting a model factory for a nonexistent provider."""
    # Try to get a factory for a provider that doesn't exist
    factory = get_model_factory("nonexistent")
    assert factory is None


def test_get_validator_factory_with_nonexistent_provider():
    """Test getting a validator factory for a nonexistent provider."""
    # Try to get a factory for a provider that doesn't exist
    factory = get_validator_factory("nonexistent")
    assert factory is None


def test_get_improver_factory_with_nonexistent_provider():
    """Test getting an improver factory for a nonexistent provider."""
    # Try to get a factory for a provider that doesn't exist
    factory = get_improver_factory("nonexistent")
    assert factory is None


def test_create_model_from_string_with_options():
    """Test creating a model from a string with options."""

    # Register a model factory
    @register_model("test")
    def create_test_model(model_name, **options):
        return MockModel(model_name, **options)

    # Create a model from a string with options
    model = create_model_from_string("test:model-name", temperature=0.7, max_tokens=100)

    # Check that the options were passed to the factory
    assert model.kwargs["temperature"] == 0.7
    assert model.kwargs["max_tokens"] == 100
