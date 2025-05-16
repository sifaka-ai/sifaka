"""
Sifaka: A simplified AI text processing framework.

This package provides a clean, intuitive API for working with language models,
validators, and critics in a streamlined way.
"""

import logging

__version__ = "0.1.0"

# Import key components for easier access
from .chain import Chain
from .models import create_model, ModelProvider
from .types import ValidationResult, ChainResult

# Make validators and critics available
from .validators.length import LengthValidator
from .critics.prompt import PromptCritic


# Initialize the DI container for the application
from .di.bootstrap import initialize_di_container, register_component_factories
from .di import resolve, register, register_factory, get_container

# Set up the DI container
initialize_di_container()
register_component_factories()


# Define commonly used functions at the package level
def model(model_type: str, **kwargs):
    """
    Create a model provider.

    Args:
        model_type: The type of model provider to create (e.g., "openai", "anthropic")
        **kwargs: Additional arguments for the model provider

    Returns:
        A model provider instance
    """
    factory = resolve("model.factory")
    return factory(model_type, **kwargs)
