"""
Anthropic Model Provider

This module provides the AnthropicProvider class which implements the ModelProviderProtocol
interface for Anthropic Claude models.

## Overview
The Anthropic provider connects to Anthropic's API for text generation, offering access
to Claude models like Claude 3 Opus, Claude 3 Sonnet, and others. It handles authentication,
API communication, token counting, and response processing.
"""

import time
import importlib.util
from typing import Any, Dict, Optional, ClassVar

from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.interfaces.model import ModelProviderProtocol
from sifaka.models.managers.anthropic_client import AnthropicClientManager
from sifaka.models.managers.anthropic_token_counter import AnthropicTokenCounterManager
from sifaka.utils.config import ModelConfig
from sifaka.utils.errors import safely_execute_component_operation
from sifaka.utils.errors import ModelError
from sifaka.utils.common import record_error
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(ModelProviderProtocol):
    """
    Anthropic model provider implementation.

    This provider supports Anthropic Claude models with configurable parameters,
    built-in token counting, and execution tracking. It handles communication
    with Anthropic's API, token counting, and response processing.
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "claude-3-opus-20240229"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
        """
        # Verify Anthropic package is installed
        try:
            if importlib.util.find_spec("anthropic") is None:
                raise ImportError()
        except ImportError:
            raise ImportError("Anthropic package is required. Install with: pip install anthropic")

        # Initialize state manager
        from sifaka.utils.state import StateManager

        self._state_manager = StateManager()

        # Create managers
        self._client_manager = AnthropicClientManager(
            model_name=model_name,
            config=config or ModelConfig(),
            api_client=api_client,
        )
        self._token_counter_manager = AnthropicTokenCounterManager(
            model_name=model_name,
            token_counter=token_counter,
        )

        # Initialize state
        self._state_manager.update("model_name", model_name)
        self._state_manager.update("config", config or ModelConfig())
        self._state_manager.update("initialized", False)
        self._state_manager.update(
            "stats",
            {
                "generation_count": 0,
                "token_count_calls": 0,
                "error_count": 0,
                "total_processing_time": 0,
            },
        )

        logger.info(f"Created AnthropicProvider with model {model_name}")

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = time.time()

        # Define the operation
        def operation():
            # Actual processing logic
            return self.generate(prompt, **kwargs)

        # Use standardized error handling
        result = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            additional_metadata={"input_type": "prompt", "method": "invoke"},
        )

        # Update statistics
        processing_time = time.time() - start_time
        stats = self._state_manager.get("stats", {})
        stats["generation_count"] = stats.get("generation_count", 0) + 1
        stats["total_processing_time"] = (
            stats.get("total_processing_time", 0) + processing_time * 1000
        )
        self._state_manager.update("stats", stats)

        return result

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously invoke the model with a prompt.

        This method delegates to agenerate if it exists, or falls back to
        synchronous generate.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = time.time()

        try:
            # Define the async operation
            async def async_operation():
                if hasattr(self, "agenerate"):
                    return await self.agenerate(prompt, **kwargs)
                else:
                    # Fall back to synchronous generate
                    return self.generate(prompt, **kwargs)

            # Execute the async operation
            result = await async_operation()

            # Update statistics
            processing_time = time.time() - start_time
            stats = self._state_manager.get("stats", {})
            stats["generation_count"] = stats.get("generation_count", 0) + 1
            stats["total_processing_time"] = (
                stats.get("total_processing_time", 0) + processing_time * 1000
            )
            self._state_manager.update("stats", stats)

            return result

        except Exception as e:
            # Record the error using standardized error handling
            self._record_error(e)

            # Raise a standardized error
            raise ModelError(
                f"Error in async invocation: {str(e)}",
                metadata={
                    "component_name": self.name,
                    "model_name": self._state_manager.get("model_name"),
                    "method": "ainvoke",
                    "error_type": type(e).__name__,
                },
            ) from e

    def _record_error(self, error: Exception) -> None:
        """Record an error in the state manager."""
        # Update error count in state
        stats = self._state_manager.get("stats", {})
        stats["error_count"] = stats.get("error_count", 0) + 1
        self._state_manager.update("stats", stats)

        # Use common error recording utility
        record_error(
            error=error,
            component_name=self.name,
            component_type=self.__class__.__name__,
            state_manager=self._state_manager,
        )

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return f"Anthropic-{self._state_manager.get('model_name')}"

    def warm_up(self) -> None:
        """
        Initialize resources needed by the Anthropic provider.
        """
        # Ensure component is not already initialized
        if self._state_manager.get("initialized", False):
            logger.debug(f"Provider {self.name} already initialized")
            return

        # Initialize client
        client = self._client_manager.get_client()
        self._state_manager.update("client", client)

        # Initialize token counter
        token_counter = self._token_counter_manager.get_token_counter()
        self._state_manager.update("token_counter", token_counter)

        # Mark as initialized
        self._state_manager.update("initialized", True)
        logger.info(f"Provider {self.name} initialized successfully")

    def cleanup(self) -> None:
        """
        Release resources used by the Anthropic provider.
        """
        # Check if already cleaned up
        if not self._state_manager.get("initialized", False):
            logger.debug(f"Provider {self.name} not initialized, nothing to clean up")
            return

        # Release Anthropic-specific resources
        client = self._state_manager.get("client")
        if client and hasattr(client, "close"):
            client.close()

        # Clear provider-specific stats
        self._state_manager.update(
            "stats",
            {
                "generation_count": 0,
                "token_count_calls": 0,
                "error_count": 0,
                "total_processing_time": 0,
            },
        )

        # Mark as not initialized
        self._state_manager.update("initialized", False)
        logger.info(f"Provider {self.name} cleaned up successfully")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Get client from state
        client = self._state_manager.get("client")
        if client is None:
            client = self._client_manager.get_client()
            self._state_manager.update("client", client)

        # Get config from state
        config = self._state_manager.get("config")

        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config, "params"):
                config.params[key] = value

        # Send prompt to client
        return client.send_prompt(prompt, config)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Get token counter from state
        token_counter = self._state_manager.get("token_counter")
        if token_counter is None:
            token_counter = self._token_counter_manager.get_token_counter()
            self._state_manager.update("token_counter", token_counter)

        # Update statistics
        stats = self._state_manager.get("stats", {})
        stats["token_count_calls"] = stats.get("token_count_calls", 0) + 1
        self._state_manager.update("stats", stats)

        # Count tokens
        return token_counter.count_tokens(text)

    # Note: Text analysis functionality has been removed from the provider
    # For text analysis, use the critics component instead (e.g., SelfRefineCritic)
    # Example:
    # from sifaka.critics.implementations.self_refine import create_self_refine_critic
    # critic = create_self_refine_critic(llm_provider=anthropic_provider)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about provider usage.

        Returns:
            Dictionary with usage statistics
        """
        # Get statistics from tracing manager and state
        tracing_manager = self._state_manager.get("tracing_manager")
        tracing_stats = tracing_manager.get_statistics() if tracing_manager else {}

        # Combine with any other stats from state
        stats = self._state_manager.get("stats", {})

        return {**tracing_stats, **stats}
