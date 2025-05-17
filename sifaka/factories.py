"""
Factory functions for Sifaka components.

This module provides factory functions for creating Sifaka components.
These factories use the registry to find the appropriate implementation
without creating circular dependencies.
"""

import logging
from typing import Optional, Any, Dict, List, Union, Type, TypeVar, cast

from sifaka.interfaces import Model, Validator, Improver
from sifaka.registry import (
    get_model_factory,
    get_validator_factory,
    get_improver_factory,
)

# Logger
logger = logging.getLogger(__name__)


class FactoryError(Exception):
    """Error raised when a factory function fails."""

    pass


def create_model(provider: str, model_name: str, **options: Any) -> Model:
    """Create a model instance.

    This function uses the registry to find the appropriate model factory
    for the specified provider.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").
        model_name: The model name (e.g., "gpt-4", "claude-3").
        **options: Additional options to pass to the model constructor.

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
        # Create the model
        return factory(model_name, **options)
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


def create_model_from_string(model_string: str, **options: Any) -> Model:
    """Create a model instance from a string.

    Args:
        model_string: A string in the format "provider:model_name".
        **options: Additional options to pass to the model constructor.

    Returns:
        A model instance.

    Raises:
        FactoryError: If the provider is not found or the factory fails.
        ValueError: If the model string is not in the correct format.
    """
    provider, model_name = parse_model_string(model_string)
    return create_model(provider, model_name, **options)


def create_validator(name: str, **options: Any) -> Validator:
    """Create a validator instance.

    This function uses the registry to find the appropriate validator factory
    for the specified name.

    Args:
        name: The name of the validator.
        **options: Options to pass to the validator constructor.

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
        # Create the validator
        return factory(**options)
    except Exception as e:
        raise FactoryError(f"Error creating validator: {str(e)}") from e


def create_improver(name: str, model: Union[str, Model], **options: Any) -> Improver:
    """Create an improver instance.

    This function uses the registry to find the appropriate improver factory
    for the specified name.

    Args:
        name: The name of the improver.
        model: The model to use for improvement, either a Model instance or a string.
        **options: Additional options to pass to the improver constructor.

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
        model = create_model_from_string(model)

    try:
        # Create the improver
        return factory(model, **options)
    except Exception as e:
        raise FactoryError(f"Error creating improver: {str(e)}") from e
