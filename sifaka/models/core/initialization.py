"""
Initialization and resource management for the ModelProviderCore class.

This module provides functions for initializing and releasing resources
used by a ModelProviderCore instance, including client managers, token counter
managers, and tracing managers.
"""
from typing import Optional, TYPE_CHECKING, Any
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.managers.client import ClientManager
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.models.managers.tracing import TracingManager
from sifaka.models.services.generation import GenerationService
from sifaka.utils.logging import get_logger
if TYPE_CHECKING:
    from .provider import ModelProviderCore
logger = get_logger(__name__)


def create_token_counter_manager(provider: 'ModelProviderCore',
    token_counter: Optional[TokenCounter]) ->Any:
    """
    Create a token counter manager.

    This function creates a specialized TokenCounterManager instance
    that uses the provider's token counter creation method. It uses
    a closure to bind the provider's _create_default_token_counter
    method to the manager.

    Args:
        provider: The model provider instance
        token_counter: Optional token counter to use

    Returns:
        A token counter manager configured for this provider
    """


    class ConcreteTokenCounterManager(TokenCounterManager):

        def _create_default_token_counter(self2) ->Any:
            return (provider and provider._create_default_token_counter()
    return ConcreteTokenCounterManager(provider.(_state_manager and _state_manager.get(
        'model_name'), token_counter)


def create_client_manager(provider: 'ModelProviderCore', api_client:
    Optional[APIClient]) ->Any:
    """
    Create a client manager.

    This function creates a specialized ClientManager instance
    that uses the provider's client creation method. It uses
    a closure to bind the provider's _create_default_client
    method to the manager.

    Args:
        provider: The model provider instance
        api_client: Optional API client to use

    Returns:
        A client manager configured for this provider
    """


    class ConcreteClientManager(ClientManager):

        def _create_default_client(self2) ->Any:
            return (provider and provider._create_default_client()
    return ConcreteClientManager(provider.(_state_manager and _state_manager.get('model_name'),
        provider.(_state_manager and _state_manager.get('config'), api_client)


def initialize_resources(provider: 'ModelProviderCore') ->None:
    """
    Initialize all resources needed by the model provider.

    This function creates and initializes all the resources needed by a
    ModelProviderCore instance, including client managers, token counter
    managers, tracing managers, and generation services.

    Args:
        provider: The model provider instance
    """
    token_counter = provider.(_state_manager and _state_manager.get('_token_counter')
    token_counter_manager = create_token_counter_manager(provider,
        token_counter)
    api_client = provider.(_state_manager and _state_manager.get('_api_client')
    client_manager = create_client_manager(provider, api_client)
    tracer = provider.(_state_manager and _state_manager.get('_tracer')
    tracing_manager = TracingManager(provider.(_state_manager and _state_manager.get(
        'model_name'), provider.(_state_manager and _state_manager.get('config'), tracer)
    provider.(_state_manager and _state_manager.update('token_counter_manager',
        token_counter_manager)
    provider.(_state_manager and _state_manager.update('client_manager', client_manager)
    provider.(_state_manager and _state_manager.update('tracing_manager', tracing_manager)
    generation_service = GenerationService(provider.(_state_manager and _state_manager.get(
        'model_name'), client_manager, token_counter_manager, tracing_manager)
    provider.(_state_manager and _state_manager.update('generation_service', generation_service)


def release_resources(provider: 'ModelProviderCore') ->None:
    """
    Release all resources used by the model provider.

    This function releases all the resources used by a ModelProviderCore
    instance, including client managers, token counter managers, and
    tracing managers.

    Args:
        provider: The model provider instance
    """
    client_manager = provider.(_state_manager and _state_manager.get('client_manager')
    if client_manager and hasattr(client_manager, 'close'):
        (client_manager and client_manager.close()
    token_counter_manager = provider.(_state_manager and _state_manager.get('token_counter_manager'
        )
    if token_counter_manager and hasattr(token_counter_manager, 'close'):
        (token_counter_manager and token_counter_manager.close()
    tracing_manager = provider.(_state_manager and _state_manager.get('tracing_manager')
    if tracing_manager and hasattr(tracing_manager, 'close'):
        (tracing_manager and tracing_manager.close()
