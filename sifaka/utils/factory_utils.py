"""Simple factory utilities for Sifaka models.

This module provides minimal helper functions to reduce boilerplate
in model factory functions.
"""

import logging
from typing import Any, Callable, TypeVar

from sifaka.utils.error_handling import log_error

T = TypeVar('T')


def create_with_error_handling(
    factory_func: Callable[..., T],
    provider_name: str,
    model_name: str,
    **kwargs: Any
) -> T:
    """Create a model instance with consistent error handling and logging.
    
    Args:
        factory_func: The actual model constructor function.
        provider_name: Name of the provider (e.g., "OpenAI", "Anthropic").
        model_name: The model name being created.
        **kwargs: Arguments to pass to the factory function.
        
    Returns:
        The created model instance.
        
    Raises:
        Exception: Re-raises any exception from the factory function.
    """
    logger = logging.getLogger(__name__)
    
    # Log model creation attempt
    logger.debug(f"Creating {provider_name} model with name '{model_name}'")
    
    try:
        # Create the model
        model = factory_func(model_name=model_name, **kwargs)
        
        # Log successful model creation
        logger.debug(f"Successfully created {provider_name} model with name '{model_name}'")
        
        return model
    except Exception as e:
        # Log the error
        log_error(e, logger, component=f"{provider_name}Model", operation="creation")
        
        # Re-raise the exception
        raise
