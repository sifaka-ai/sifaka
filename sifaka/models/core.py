"""
Core model provider implementation.

This module provides the ModelProviderCore class which is the main interface
for model providers, delegating to specialized components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

from sifaka.models.base import APIClient, ModelConfig, ModelProvider, TokenCounter
from sifaka.models.managers.client import ClientManager
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.models.managers.tracing import TracingManager
from sifaka.models.services.generation import GenerationService
from sifaka.utils.tracing import Tracer
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic model provider types
T = TypeVar("T", bound="ModelProviderCore")


class ModelProviderCore(ModelProvider):
    """
    Core model provider implementation that delegates to specialized components.
    
    This class implements the ModelProvider interface but delegates most of its
    functionality to specialized components for better separation of concerns.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        """
        Initialize a ModelProviderCore instance.
        
        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
            tracer: Optional tracer to use
        """
        self._model_name = model_name
        self._config = config or ModelConfig()
        
        # Create managers
        self._token_counter_manager = self._create_token_counter_manager(token_counter)
        self._client_manager = self._create_client_manager(api_client)
        self._tracing_manager = TracingManager(model_name, self._config, tracer)
        
        # Create services
        self._generation_service = GenerationService(
            model_name,
            self._client_manager,
            self._token_counter_manager,
            self._tracing_manager,
        )
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")
        
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
        
    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._config
        
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens in the text
            
        Raises:
            TypeError: If text is not a string
        """
        token_count = self._token_counter_manager.count_tokens(text)
        
        self._tracing_manager.trace_event(
            "token_count",
            {
                "text_length": len(text),
                "token_count": token_count,
            },
        )
        
        return token_count
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: The prompt to generate from
            **kwargs: Optional overrides for model configuration
            
        Returns:
            The generated text
            
        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            RuntimeError: If an error occurs during generation
        """
        # Update config with any override kwargs
        config = ModelConfig(
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            api_key=kwargs.pop("api_key", self.config.api_key),
            trace_enabled=kwargs.pop("trace_enabled", self.config.trace_enabled),
        )
        
        return self._generation_service.generate(prompt, config)
        
    def _create_token_counter_manager(self, token_counter: Optional[TokenCounter]) -> TokenCounterManager:
        """
        Create a token counter manager.
        
        Args:
            token_counter: Optional token counter to use
            
        Returns:
            A token counter manager
        """
        class ConcreteTokenCounterManager(TokenCounterManager):
            def _create_default_token_counter(self2) -> TokenCounter:
                return self._create_default_token_counter()
                
        return ConcreteTokenCounterManager(self._model_name, token_counter)
        
    def _create_client_manager(self, api_client: Optional[APIClient]) -> ClientManager:
        """
        Create a client manager.
        
        Args:
            api_client: Optional API client to use
            
        Returns:
            A client manager
        """
        class ConcreteClientManager(ClientManager):
            def _create_default_client(self2) -> APIClient:
                return self._create_default_client()
                
        return ConcreteClientManager(self._model_name, self._config, api_client)
        
    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """
        Create a default API client if none was provided.
        
        Returns:
            A default API client for the model
        """
        ...
        
    @abstractmethod
    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter if none was provided.
        
        Returns:
            A default token counter for the model
        """
        ...
