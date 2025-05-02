"""
API client manager for model providers.

This module provides the ClientManager class which is responsible for
managing API clients for model providers.
"""

from abc import ABC, abstractmethod
from typing import Optional

from sifaka.models.base import APIClient, ModelConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ClientManager:
    """
    Manages API clients for model providers.
    
    This class is responsible for creating and managing API clients,
    and providing a consistent interface for API client operations.
    """
    
    def __init__(self, model_name: str, config: ModelConfig, api_client: Optional[APIClient] = None):
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
        
    def get_client(self) -> APIClient:
        """
        Get the API client, creating a default one if needed.
        
        Returns:
            The API client to use
        """
        return self._ensure_api_client()
        
    def _ensure_api_client(self) -> APIClient:
        """
        Ensure an API client is available, creating a default one if needed.
        
        Returns:
            The API client to use
        """
        if self._api_client is None:
            logger.debug(f"Creating default API client for {self._model_name}")
            self._api_client = self._create_default_client()
        return self._api_client
        
    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """
        Create a default API client if none was provided.
        
        Returns:
            A default API client for the model
        """
        ...
