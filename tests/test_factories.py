"""
Tests for the factory functions.

This module tests the factory functions in sifaka.factories.
"""

import pytest
from unittest.mock import Mock, patch

from sifaka.interfaces import Model, Validator, Improver
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


@pytest.fixture
def mock_registry():
    """Fixture for mocking the registry."""
    with (
        patch("sifaka.factories.get_model_factory") as mock_get_model_factory,
        patch("sifaka.factories.get_validator_factory") as mock_get_validator_factory,
        patch("sifaka.factories.get_improver_factory") as mock_get_improver_factory,
    ):

        # Mock model factory
        def mock_model_factory(model_name, **options):
            return MockModel(model_name, **options)

        # Mock validator factory
        def mock_validator_factory(**options):
            return MockValidator(**options)

        # Mock improver factory
        def mock_improver_factory(model, **options):
            return MockImprover(model, **options)

        # Set up the mocks
        mock_get_model_factory.return_value = mock_model_factory
        mock_get_validator_factory.return_value = mock_validator_factory
        mock_get_improver_factory.return_value = mock_improver_factory

        yield {
            "get_model_factory": mock_get_model_factory,
            "get_validator_factory": mock_get_validator_factory,
            "get_improver_factory": mock_get_improver_factory,
        }


def test_create_model(mock_registry):
    """Test creating a model using create_model."""
    # Create a model
    model = create_model("test", "test-model", temperature=0.7)

    # Check that the model was created correctly
    assert isinstance(model, MockModel)
    assert model.model_name == "test-model"
    assert model.kwargs["temperature"] == 0.7

    # Check that the factory was called correctly
    mock_registry["get_model_factory"].assert_called_once_with("test")


def test_create_model_from_string(mock_registry):
    """Test creating a model using create_model_from_string."""
    # Create a model
    model = create_model_from_string("test:test-model", temperature=0.7)

    # Check that the model was created correctly
    assert isinstance(model, MockModel)
    assert model.model_name == "test-model"
    assert model.kwargs["temperature"] == 0.7

    # Check that the factory was called correctly
    mock_registry["get_model_factory"].assert_called_once_with("test")


def test_create_model_from_string_invalid_format():
    """Test creating a model with an invalid string format."""
    # Try to create a model with an invalid string format
    with pytest.raises(ValueError):
        create_model_from_string("invalid-format")


def test_create_model_factory_not_found(mock_registry):
    """Test creating a model when the factory is not found."""
    # Set up the mock to return None
    mock_registry["get_model_factory"].return_value = None

    # Try to create a model
    with pytest.raises(FactoryError):
        create_model("test", "test-model")


def test_create_validator(mock_registry):
    """Test creating a validator using create_validator."""
    # Create a validator
    validator = create_validator("test", min_length=10, max_length=100)

    # Check that the validator was created correctly
    assert isinstance(validator, MockValidator)
    assert validator.kwargs["min_length"] == 10
    assert validator.kwargs["max_length"] == 100

    # Check that the factory was called correctly
    mock_registry["get_validator_factory"].assert_called_once_with("test")


def test_create_validator_factory_not_found(mock_registry):
    """Test creating a validator when the factory is not found."""
    # Set up the mock to return None
    mock_registry["get_validator_factory"].return_value = None

    # Try to create a validator
    with pytest.raises(FactoryError):
        create_validator("test")


def test_create_improver(mock_registry):
    """Test creating an improver using create_improver."""
    # Create a model
    model = MockModel("test-model")

    # Create an improver
    improver = create_improver("test", model, level="high")

    # Check that the improver was created correctly
    assert isinstance(improver, MockImprover)
    assert improver.model == model
    assert improver.kwargs["level"] == "high"

    # Check that the factory was called correctly
    mock_registry["get_improver_factory"].assert_called_once_with("test")


def test_create_improver_factory_not_found(mock_registry):
    """Test creating an improver when the factory is not found."""
    # Set up the mock to return None
    mock_registry["get_improver_factory"].return_value = None

    # Create a model
    model = MockModel("test-model")

    # Try to create an improver
    with pytest.raises(FactoryError):
        create_improver("test", model)
