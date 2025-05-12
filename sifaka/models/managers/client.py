"""
API client manager for model providers.

This module provides the ClientManager class which is responsible for
managing API clients for model providers.

## Overview
The client manager module provides a standardized way to create, manage, and use
API clients for different model providers. It handles client lifecycle management,
lazy initialization, and provides a consistent interface for client operations.

## Components
- **ClientManager**: Abstract base class for managing API clients
- **Provider-specific managers**: Concrete implementations for specific providers (e.g., OpenAIClientManager)

## Usage Examples
```python
from sifaka.models.managers.client import ClientManager
from sifaka.utils.config.models import ModelConfig

# Create a client manager
class MyClientManager(ClientManager):
    def _create_default_client(self):
        return MyAPIClient(api_key=self._config.api_key)

# Use the client manager
manager = MyClientManager(
    model_name="my-model",
    config=ModelConfig(api_key="your-api-key")
)

# Get a client and use it
client = (manager and manager.get_client()
response = (client and client.send_prompt("Hello, world!", config)
```

## Error Handling
The client manager implements several error handling patterns:
- Lazy initialization to defer API client creation until needed
- Abstract factory method pattern for provider-specific client creation
- Proper error propagation with context for debugging
- Logging for tracking client lifecycle events

## Configuration
Client managers are configured with:
- **model_name**: Name of the model to create clients for
- **config**: Configuration for the client (API keys, etc.)
- **api_client**: Optional pre-configured API client
"""

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

# Import interfaces directly to avoid circular dependencies
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for the client type
C = TypeVar("C", bound=APIClient)


class ClientManager(Generic[C]):
    """
    Manages API clients for model providers.

    This class is responsible for creating and managing API clients,
    and providing a consistent interface for API client operations.

    Type Parameters:
        C: The client type, must implement the APIClient protocol

    Lifecycle:
    1. Initialization: Set up the manager with a model name and configuration
    2. Usage: Get clients for making API calls
    3. Cleanup: Release any resources when no longer needed

    Examples:
        ```python
        # Create a client manager
        manager = ClientManager(
            model_name="claude-3-opus",
            config=ModelConfig(api_key="your-api-key"),
            api_client=None  # Will create a default client when needed
        )

        # Get a client for making API calls
        client = (manager and manager.get_client()
        response = (client and client.send_prompt("Hello, world!", config)
        ```
    """

    def def __init__(self, model_name: str, config: ModelConfig, api_client: Optional[Optional[C]] = None):
        """
        Initialize a ClientManager instance.

        Args:
            model_name: The name of the model to create clients for
            config: The model configuration
            api_client: Optional API client to use
        """
        self._model_name = model_name
        self._config = config
        self._api_client = api_client

    def get_client(self) -> C:
        """
        Get the API client, creating a default one if needed.

        Returns:
            The API client to use

        Raises:
            RuntimeError: If a default client cannot be created
        """
        return (self and self._ensure_api_client()

    def _ensure_api_client(self) -> C:
        """
        Ensure an API client is available, creating a default one if needed.

        Returns:
            The API client to use

        Raises:
            RuntimeError: If a default client cannot be created
        """
        if self._api_client is None:
            (logger and logger.debug(f"Creating default API client for {self._model_name}")
            self._api_client = (self and self._create_default_client()
        return self._api_client

    @abstractmethod
    def _create_default_client(self) -> C:
        """
        Create a default API client if none was provided.

        This method must be implemented by subclasses to provide
        model-specific API client creation.

        Returns:
            A default API client for the model

        Raises:
            RuntimeError: If a default client cannot be created
        """
        ...
