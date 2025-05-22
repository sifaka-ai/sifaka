"""
Factory functions for Sifaka components.

This module provides factory functions for creating Sifaka components.
These factories use the registry to find the appropriate implementation
without creating circular dependencies.
"""

import logging
from typing import Any, Optional, Union

from sifaka.config import CriticConfig, ModelConfig, ValidatorConfig
from sifaka.interfaces import Improver, Model, Validator
from sifaka.registry import get_improver_factory, get_model_factory, get_validator_factory

# Logger
logger = logging.getLogger(__name__)


class FactoryError(Exception):
    """Error raised when a factory function fails."""


def create_model(
    provider: str, model_name: str, config: Optional[ModelConfig] = None, **options: Any
) -> Model:
    """Create a model instance.

    This function uses the registry to find the appropriate model factory
    for the specified provider.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").
        model_name: The model name (e.g., "gpt-4", "claude-3").
        config: Optional configuration for the model.
        **options: Additional options to pass to the model constructor.
            These options will override any options set in the configuration.

    Returns:
        A model instance.

    Raises:
        FactoryError: If the provider is not found or the factory fails.
    """
    provider = provider.lower()

    # Get the factory from the registry
    factory = get_model_factory(provider)
    if factory is None:
        raise FactoryError(f"Model provider '{provider}' not found")

    try:
        # Merge configuration with options
        merged_options = {}

        # Start with configuration if provided
        if config:
            # Convert config to dictionary, filtering out None values
            config_dict = {
                k: v for k, v in config.__dict__.items() if not k.startswith("_") and v is not None
            }
            merged_options.update(config_dict)

            # Add custom options from config
            if hasattr(config, "custom") and config.custom:
                merged_options.update(config.custom)

        # Override with explicit options
        merged_options.update(options)

        # Create the model
        model = factory(model_name, **merged_options)

        # Configure the model with the merged options
        if hasattr(model, "configure"):
            model.configure(**merged_options)

        return model
    except Exception as e:
        raise FactoryError(f"Error creating model: {str(e)}") from e


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse a model string into provider and model name.

    Args:
        model_string: A string in the format "provider:model_name".

    Returns:
        A tuple of (provider, model_name).

    Raises:
        ValueError: If the model string is not in the correct format.
    """
    try:
        provider, model_name = model_string.split(":", 1)
        return provider.strip(), model_name.strip()
    except ValueError:
        raise ValueError(
            f"Invalid model string: '{model_string}'. " f"Expected format: 'provider:model_name'"
        )


def create_model_from_string(
    model_string: str, config: Optional[ModelConfig] = None, **options: Any
) -> Model:
    """Create a model instance from a string.

    Args:
        model_string: A string in the format "provider:model_name".
        config: Optional configuration for the model.
        **options: Additional options to pass to the model constructor.
            These options will override any options set in the configuration.

    Returns:
        A model instance.

    Raises:
        FactoryError: If the provider is not found or the factory fails.
        ValueError: If the model string is not in the correct format.
    """
    provider, model_name = parse_model_string(model_string)
    return create_model(provider, model_name, config=config, **options)


def create_validator(
    name: str, config: Optional[ValidatorConfig] = None, **options: Any
) -> Validator:
    """Create a validator instance.

    This function uses the registry to find the appropriate validator factory
    for the specified name.

    Args:
        name: The name of the validator.
        config: Optional configuration for the validator.
        **options: Options to pass to the validator constructor.
            These options will override any options set in the configuration.

    Returns:
        A validator instance.

    Raises:
        FactoryError: If the validator is not found or the factory fails.
    """
    name = name.lower()

    # Get the factory from the registry
    factory = get_validator_factory(name)
    if factory is None:
        raise FactoryError(f"Validator '{name}' not found")

    try:
        # Merge configuration with options
        merged_options = {}

        # Start with configuration if provided
        if config:
            # Convert config to dictionary, filtering out None values
            config_dict = {
                k: v for k, v in config.__dict__.items() if not k.startswith("_") and v is not None
            }
            merged_options.update(config_dict)

            # Add custom options from config
            if hasattr(config, "custom") and config.custom:
                merged_options.update(config.custom)

        # Override with explicit options
        merged_options.update(options)

        # Create the validator
        validator = factory(**merged_options)

        # Configure the validator with the merged options
        if hasattr(validator, "configure"):
            validator.configure(**merged_options)

        return validator
    except Exception as e:
        raise FactoryError(f"Error creating validator: {str(e)}") from e


def create_improver(
    name: str,
    model: Union[str, Model],
    config: Optional[CriticConfig] = None,
    model_config: Optional[ModelConfig] = None,
    **options: Any,
) -> Improver:
    """Create an improver instance.

    This function uses the registry to find the appropriate improver factory
    for the specified name.

    Args:
        name: The name of the improver.
        model: The model to use for improvement, either a Model instance or a string.
        config: Optional configuration for the improver.
        model_config: Optional configuration for the model, if model is a string.
        **options: Additional options to pass to the improver constructor.
            These options will override any options set in the configuration.

    Returns:
        An improver instance.

    Raises:
        FactoryError: If the improver is not found or the factory fails.
    """
    name = name.lower()

    # Get the factory from the registry
    factory = get_improver_factory(name)
    if factory is None:
        raise FactoryError(f"Improver '{name}' not found")

    # Convert model string to model instance if needed
    if isinstance(model, str):
        model = create_model_from_string(model, config=model_config)

    try:
        # Merge configuration with options
        merged_options = {}

        # Start with configuration if provided
        if config:
            # Convert config to dictionary, filtering out None values
            config_dict = {
                k: v for k, v in config.__dict__.items() if not k.startswith("_") and v is not None
            }
            merged_options.update(config_dict)

            # Add custom options from config
            if hasattr(config, "custom") and config.custom:
                merged_options.update(config.custom)

        # Override with explicit options
        merged_options.update(options)

        # Create the improver
        improver = factory(model, **merged_options)

        # Configure the improver with the merged options
        if hasattr(improver, "configure"):
            improver.configure(**merged_options)

        return improver
    except Exception as e:
        raise FactoryError(f"Error creating improver: {str(e)}") from e
