"""
Mock provider for testing.

This module provides a mock model provider that can be used for testing
without making real API calls.
"""

from typing import Any, Dict, List, Optional
from pydantic import PrivateAttr
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.result import GenerationResult, TokenCountResult
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.state import StateManager, create_model_state
from sifaka.interfaces.model import ModelProviderProtocol


class MockAPIClient(APIClient):
    """Mock API client for testing."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the mock client.

        Args:
            api_key: Optional API key (not used but included for compatibility)
        """
        self.api_key = api_key or "mock-api-key"
        self.calls = []

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """
        Send a mock prompt and return a response.

        Args:
            prompt: The prompt to send
            config: Configuration for the request

        Returns:
            A mock response
        """
        # Record the call
        self.calls.append({"prompt": prompt, "config": config})

        # Simulate different responses based on configuration
        temperature = config.temperature if hasattr(config, "temperature") else 0.7
        prefix = "Detailed" if temperature < 0.5 else "Creative"

        return f"{prefix} mock response to: {prompt}"

    def reset_calls(self) -> None:
        """Reset the list of calls."""
        self.calls = []


class MockTokenCounter(TokenCounter):
    """Mock token counter for testing."""

    def __init__(self, model: str = "mock-model"):
        """
        Initialize the mock token counter.

        Args:
            model: The model name (not used but included for compatibility)
        """
        self.model = model
        self.calls = []

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text by splitting on whitespace.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens (words) in the text
        """
        self.calls.append(text)
        return len(text.split())

    def reset_calls(self) -> None:
        """Reset the list of calls."""
        self.calls = []


class MockProvider(ModelProviderProtocol):
    """
    Mock model provider for testing.

    This provider returns predefined responses for testing without making
    real API calls.
    """

    # State management
    _state_manager = PrivateAttr(default_factory=create_model_state)

    def __init__(
        self,
        model_name: str = "mock-model",
        name: str = "mock_provider",
        description: str = "Mock provider for testing",
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "This is a mock response.",
        token_count: int = 10,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize the mock provider.

        Args:
            model_name: The name of the model to use
            name: The name of the provider
            description: The description of the provider
            responses: Dictionary mapping prompts to responses
            default_response: Default response to return if prompt not in responses
            token_count: Number of tokens to report for token counting
            config: Configuration for the provider
        """
        # Create default config if not provided
        if config is None:
            config = ModelConfig(
                temperature=0.7,
                max_tokens=100,
                api_key="mock-api-key",
            )

        # Initialize state
        self._state_manager = create_model_state()
        self._state_manager.update("model_name", model_name)
        self._state_manager.update("name", name)
        self._state_manager.update("description", description)
        self._state_manager.update("config", config)
        self._state_manager.update("initialized", True)

        # Store mock-specific properties
        self._responses = responses or {}
        self._default_response = default_response
        self._token_count = token_count
        self._calls = []
        self._client = MockAPIClient()
        self._token_counter = MockTokenCounter(model=model_name)

        # Store in state
        self._state_manager.update("client", self._client)
        self._state_manager.update("token_counter", self._token_counter)

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The model name
        """
        return self._state_manager.get("model_name")

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return self._state_manager.get("name")

    @property
    def description(self) -> str:
        """
        Get the provider description.

        Returns:
            The provider description
        """
        return self._state_manager.get("description")

    @property
    def config(self) -> ModelConfig:
        """
        Get the model configuration.

        Returns:
            The model configuration
        """
        return self._state_manager.get("config")

    def update_config(self, config: ModelConfig) -> None:
        """
        Update the provider configuration.

        Args:
            config: The new configuration
        """
        self._state_manager.update("config", config)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The prompt to generate a response for
            **kwargs: Additional arguments

        Returns:
            str: The generated response
        """
        # Record the call
        self._calls.append({"prompt": prompt, "kwargs": kwargs})

        # Get the response
        response = self._responses.get(prompt, self._default_response)

        # Update statistics
        stats = self._state_manager.get("stats", {})
        stats["generation_count"] = stats.get("generation_count", 0) + 1
        self._state_manager.update("stats", stats)

        return response

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            int: The number of tokens in the text
        """
        # Update statistics
        stats = self._state_manager.get("stats", {})
        stats["token_count_calls"] = stats.get("token_count_calls", 0) + 1
        self._state_manager.update("stats", stats)

        return self._token_count

    def get_calls(self) -> List[Dict[str, Any]]:
        """
        Get the list of calls made to this provider.

        Returns:
            List[Dict[str, Any]]: The list of calls
        """
        return self._calls

    def reset_calls(self) -> None:
        """Reset the list of calls."""
        self._calls = []

    def set_response(self, prompt: str, response: str) -> None:
        """
        Set a response for a specific prompt.

        Args:
            prompt: The prompt to set a response for
            response: The response to return for the prompt
        """
        self._responses[prompt] = response

    def set_default_response(self, response: str) -> None:
        """
        Set the default response.

        Args:
            response: The default response to return
        """
        self._default_response = response

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.

        Returns:
            Dict[str, Any]: The current state
        """
        return {
            key: self._state_manager.get(key)
            for key in ["initialized", "model_name", "name", "description"]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dict[str, Any]: Usage statistics
        """
        return self._state_manager.get("stats", {})
