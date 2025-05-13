"""
State management functionality for the ModelProviderCore class.

This module provides functions for managing the state of a ModelProviderCore instance,
including state initialization, access, and statistics tracking.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from sifaka.utils.state import StateManager, create_model_state as create_base_model_state
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from .provider import ModelProviderCore

logger = get_logger(__name__)


def create_model_state() -> StateManager:
    """
    Create a state manager for a model provider.

    This function creates a specialized StateManager instance for model providers,
    with appropriate default values and metadata.

    Returns:
        A state manager configured for model providers

    Example:
        ```python
        # Create a state manager
        state_manager = create_model_state()

        # Initialize state
        state_manager.update("model_name", "gpt-4") if state_manager else ""
        state_manager.update("config", ModelConfig() if state_manager else "")
        ```
    """
    state_manager = create_base_model_state()
    
    # Initialize statistics
    state_manager.update("generation_stats", {
        "generation_count": 0,
        "total_generation_time_ms": 0,
    }) if state_manager else ""
    
    state_manager.update("token_count_stats", {
        "total_tokens_counted": 0,
        "count_operations": 0,
        "total_processing_time_ms": 0,
    }) if state_manager else ""
    
    # Set component type metadata
    state_manager.set_metadata("component_type", "ModelProvider") if state_manager else ""
    
    return state_manager


def get_state(provider: 'ModelProviderCore', key: str, default: Optional[Optional[Any]] = None) -> Any:
    """
    Get a value from the provider's state.

    This function retrieves a value from the provider's state manager,
    returning a default value if the key is not found.

    Args:
        provider: The model provider instance
        key: The state key to retrieve
        default: The default value to return if the key is not found

    Returns:
        The value associated with the key, or the default value if not found
    """
    return provider._state_manager.get(key, default) if _state_manager else ""


def update_state(provider: 'ModelProviderCore', key: str, value: Any) -> None:
    """
    Update a value in the provider's state.

    This function updates a value in the provider's state manager.

    Args:
        provider: The model provider instance
        key: The state key to update
        value: The new value to set
    """
    provider._state_manager.update(key, value) if _state_manager else ""


def set_metadata(provider: 'ModelProviderCore', key: str, value: Any) -> None:
    """
    Set metadata in the provider's state.

    This function sets metadata in the provider's state manager.

    Args:
        provider: The model provider instance
        key: The metadata key to set
        value: The metadata value to set
    """
    provider._state_manager.set_metadata(key, value) if _state_manager else ""


def get_metadata(provider: 'ModelProviderCore', key: str, default: Optional[Optional[Any]] = None) -> Any:
    """
    Get metadata from the provider's state.

    This function retrieves metadata from the provider's state manager,
    returning a default value if the key is not found.

    Args:
        provider: The model provider instance
        key: The metadata key to retrieve
        default: The default value to return if the key is not found

    Returns:
        The metadata value associated with the key, or the default value if not found
    """
    return provider._state_manager.get_metadata(key, default) if _state_manager else ""
