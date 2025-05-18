"""
Detailed tests for the factories module.

This module contains more comprehensive tests for the factories module
to improve test coverage.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

from sifaka.factories import (
    FactoryError,
    create_model,
    parse_model_string,
    create_model_from_string,
    create_validator,
    create_improver,
)
from sifaka.config import ModelConfig, ValidatorConfig, CriticConfig
from sifaka.interfaces import Model, Validator, Improver


# Mock classes for testing
class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.options = kwargs
        self.configured = False
    
    def configure(self, **kwargs):
        """Configure the model."""
        self.options.update(kwargs)
        self.configured = True


class MockValidator:
    """Mock validator for testing."""
    
    def __init__(self, **kwargs):
        self.options = kwargs
        self.configured = False
    
    def configure(self, **kwargs):
        """Configure the validator."""
        self.options.update(kwargs)
        self.configured = True


class MockImprover:
    """Mock improver for testing."""
    
    def __init__(self, model, **kwargs):
        self.model = model
        self.options = kwargs
        self.configured = False
    
    def configure(self, **kwargs):
        """Configure the improver."""
        self.options.update(kwargs)
        self.configured = True


# Mock factory functions for testing
def mock_model_factory(model_name: str, **kwargs) -> MockModel:
    """Mock model factory function."""
    return MockModel(model_name, **kwargs)


def mock_validator_factory(**kwargs) -> MockValidator:
    """Mock validator factory function."""
    return MockValidator(**kwargs)


def mock_improver_factory(model, **kwargs) -> MockImprover:
    """Mock improver factory function."""
    return MockImprover(model, **kwargs)


class TestFactoriesDetailed:
    """Detailed tests for the factories module."""

    def test_parse_model_string_valid(self) -> None:
        """Test parsing a valid model string."""
        provider, model_name = parse_model_string("openai:gpt-4")
        
        assert provider == "openai"
        assert model_name == "gpt-4"
        
        # Test with spaces
        provider, model_name = parse_model_string(" anthropic : claude-3 ")
        
        assert provider == "anthropic"
        assert model_name == "claude-3"

    def test_parse_model_string_invalid(self) -> None:
        """Test parsing an invalid model string."""
        with pytest.raises(ValueError) as excinfo:
            parse_model_string("invalid_model_string")
        
        assert "Invalid model string" in str(excinfo.value)
        assert "Expected format: 'provider:model_name'" in str(excinfo.value)

    @patch("sifaka.factories.get_model_factory", return_value=mock_model_factory)
    def test_create_model_success(self, mock_get_factory) -> None:
        """Test successful model creation."""
        model = create_model("openai", "gpt-4", temperature=0.7, max_tokens=100)
        
        assert isinstance(model, MockModel)
        assert model.model_name == "gpt-4"
        assert model.options["temperature"] == 0.7
        assert model.options["max_tokens"] == 100
        assert model.configured is True

    @patch("sifaka.factories.get_model_factory", return_value=None)
    def test_create_model_provider_not_found(self, mock_get_factory) -> None:
        """Test model creation with a non-existent provider."""
        with pytest.raises(FactoryError) as excinfo:
            create_model("nonexistent", "model-name")
        
        assert "Model provider 'nonexistent' not found" in str(excinfo.value)

    @patch("sifaka.factories.get_model_factory", return_value=mock_model_factory)
    def test_create_model_with_config(self, mock_get_factory) -> None:
        """Test model creation with a configuration object."""
        # Create a configuration object
        config = ModelConfig(temperature=0.5, max_tokens=50)
        
        # Create a model with the configuration
        model = create_model("openai", "gpt-4", config=config)
        
        assert isinstance(model, MockModel)
        assert model.model_name == "gpt-4"
        assert model.options["temperature"] == 0.5
        assert model.options["max_tokens"] == 50
        assert model.configured is True

    @patch("sifaka.factories.get_model_factory", return_value=mock_model_factory)
    def test_create_model_with_config_and_options(self, mock_get_factory) -> None:
        """Test model creation with both configuration and explicit options."""
        # Create a configuration object
        config = ModelConfig(temperature=0.5, max_tokens=50)
        
        # Create a model with the configuration and explicit options
        # Explicit options should override configuration
        model = create_model("openai", "gpt-4", config=config, temperature=0.7)
        
        assert isinstance(model, MockModel)
        assert model.model_name == "gpt-4"
        assert model.options["temperature"] == 0.7  # Overridden
        assert model.options["max_tokens"] == 50  # From config
        assert model.configured is True

    @patch("sifaka.factories.get_model_factory", return_value=mock_model_factory)
    def test_create_model_with_custom_config(self, mock_get_factory) -> None:
        """Test model creation with custom configuration options."""
        # Create a configuration object with custom options
        config = ModelConfig(temperature=0.5)
        config.custom = {"custom_option": "value"}
        
        # Create a model with the configuration
        model = create_model("openai", "gpt-4", config=config)
        
        assert isinstance(model, MockModel)
        assert model.options["temperature"] == 0.5
        assert model.options["custom_option"] == "value"
        assert model.configured is True

    @patch("sifaka.factories.get_model_factory")
    def test_create_model_factory_error(self, mock_get_factory) -> None:
        """Test handling of factory errors during model creation."""
        # Set up the factory to raise an exception
        mock_factory = MagicMock(side_effect=RuntimeError("Factory error"))
        mock_get_factory.return_value = mock_factory
        
        with pytest.raises(FactoryError) as excinfo:
            create_model("openai", "gpt-4")
        
        assert "Error creating model" in str(excinfo.value)
        assert "Factory error" in str(excinfo.value)

    @patch("sifaka.factories.create_model")
    def test_create_model_from_string_success(self, mock_create_model) -> None:
        """Test successful model creation from a string."""
        # Set up the mock to return a model
        mock_model = MockModel("gpt-4")
        mock_create_model.return_value = mock_model
        
        # Create a model from a string
        model = create_model_from_string("openai:gpt-4", temperature=0.7)
        
        # Check that create_model was called with the correct arguments
        mock_create_model.assert_called_once_with(
            "openai", "gpt-4", config=None, temperature=0.7
        )
        
        # Check that the model was returned
        assert model == mock_model

    @patch("sifaka.factories.get_validator_factory", return_value=mock_validator_factory)
    def test_create_validator_success(self, mock_get_factory) -> None:
        """Test successful validator creation."""
        validator = create_validator("format", format_type="json")
        
        assert isinstance(validator, MockValidator)
        assert validator.options["format_type"] == "json"
        assert validator.configured is True

    @patch("sifaka.factories.get_validator_factory", return_value=None)
    def test_create_validator_not_found(self, mock_get_factory) -> None:
        """Test validator creation with a non-existent validator."""
        with pytest.raises(FactoryError) as excinfo:
            create_validator("nonexistent")
        
        assert "Validator 'nonexistent' not found" in str(excinfo.value)

    @patch("sifaka.factories.get_validator_factory", return_value=mock_validator_factory)
    def test_create_validator_with_config(self, mock_get_factory) -> None:
        """Test validator creation with a configuration object."""
        # Create a configuration object
        config = ValidatorConfig()
        config.format_type = "json"
        
        # Create a validator with the configuration
        validator = create_validator("format", config=config)
        
        assert isinstance(validator, MockValidator)
        assert validator.options["format_type"] == "json"
        assert validator.configured is True

    @patch("sifaka.factories.get_validator_factory")
    def test_create_validator_factory_error(self, mock_get_factory) -> None:
        """Test handling of factory errors during validator creation."""
        # Set up the factory to raise an exception
        mock_factory = MagicMock(side_effect=RuntimeError("Factory error"))
        mock_get_factory.return_value = mock_factory
        
        with pytest.raises(FactoryError) as excinfo:
            create_validator("format")
        
        assert "Error creating validator" in str(excinfo.value)
        assert "Factory error" in str(excinfo.value)

    @patch("sifaka.factories.get_improver_factory", return_value=mock_improver_factory)
    def test_create_improver_with_model_instance(self, mock_get_factory) -> None:
        """Test improver creation with a model instance."""
        # Create a model instance
        model = MockModel("gpt-4")
        
        # Create an improver with the model instance
        improver = create_improver("reflexion", model, temperature=0.7)
        
        assert isinstance(improver, MockImprover)
        assert improver.model == model
        assert improver.options["temperature"] == 0.7
        assert improver.configured is True

    @patch("sifaka.factories.get_improver_factory", return_value=mock_improver_factory)
    @patch("sifaka.factories.create_model_from_string")
    def test_create_improver_with_model_string(self, mock_create_model, mock_get_factory) -> None:
        """Test improver creation with a model string."""
        # Set up the mock to return a model
        model = MockModel("gpt-4")
        mock_create_model.return_value = model
        
        # Create an improver with a model string
        improver = create_improver("reflexion", "openai:gpt-4", temperature=0.7)
        
        # Check that create_model_from_string was called with the correct arguments
        mock_create_model.assert_called_once_with("openai:gpt-4", config=None)
        
        # Check that the improver was created with the model
        assert isinstance(improver, MockImprover)
        assert improver.model == model
        assert improver.options["temperature"] == 0.7
        assert improver.configured is True

    @patch("sifaka.factories.get_improver_factory", return_value=None)
    def test_create_improver_not_found(self, mock_get_factory) -> None:
        """Test improver creation with a non-existent improver."""
        with pytest.raises(FactoryError) as excinfo:
            create_improver("nonexistent", MockModel("gpt-4"))
        
        assert "Improver 'nonexistent' not found" in str(excinfo.value)

    @patch("sifaka.factories.get_improver_factory", return_value=mock_improver_factory)
    def test_create_improver_with_config(self, mock_get_factory) -> None:
        """Test improver creation with a configuration object."""
        # Create a configuration object
        config = CriticConfig()
        config.temperature = 0.5
        
        # Create an improver with the configuration
        improver = create_improver("reflexion", MockModel("gpt-4"), config=config)
        
        assert isinstance(improver, MockImprover)
        assert improver.options["temperature"] == 0.5
        assert improver.configured is True

    @patch("sifaka.factories.get_improver_factory")
    def test_create_improver_factory_error(self, mock_get_factory) -> None:
        """Test handling of factory errors during improver creation."""
        # Set up the factory to raise an exception
        mock_factory = MagicMock(side_effect=RuntimeError("Factory error"))
        mock_get_factory.return_value = mock_factory
        
        with pytest.raises(FactoryError) as excinfo:
            create_improver("reflexion", MockModel("gpt-4"))
        
        assert "Error creating improver" in str(excinfo.value)
        assert "Factory error" in str(excinfo.value)
