"""
OpenAI model provider implementation.

This module provides the OpenAIProvider class which implements the ModelProviderCore
interface for OpenAI models.
"""

from typing import Optional, Any, Dict, ClassVar, Union

import openai
import tiktoken
from openai import OpenAI
from pydantic import PrivateAttr

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient(APIClient):
    """OpenAI API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=api_key)
        logger.debug("Initialized OpenAI client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to OpenAI and return the response."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise


class OpenAITokenCounter(TokenCounter):
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """Initialize the token counter for a specific model."""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            logger.debug(f"Initialized token counter for model {model}")
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in the text using the model's encoding."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise


class OpenAIProvider(ModelProviderCore):
    """
    OpenAI model provider implementation.

    This provider supports OpenAI models with configurable parameters
    and built-in token counting.
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "gpt-4-turbo"

    # State management using StateManager
    def _create_model_state():
        from sifaka.utils.state import create_model_state

        return create_model_state()

    _state = PrivateAttr(default_factory=_create_model_state)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
        """
        # Verify OpenAI package is installed
        try:
            # Just importing the package to verify it's installed
            # We already imported it at the module level
            pass
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        # Initialize state
        state = self._state.get_state()
        state.initialized = False
        state.cache = {}

        # Store components in state
        state.cache["api_client"] = api_client
        state.cache["token_counter"] = token_counter

        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )

    def _create_default_client(self) -> APIClient:
        """Create a default OpenAI client."""
        # Get state
        state = self._state.get_state()

        # Check if client is already in state cache
        if "api_client" in state.cache and state.cache["api_client"]:
            return state.cache["api_client"]

        # Create new client
        client = OpenAIClient(api_key=self.config.api_key)

        # Store in state cache
        state.cache["api_client"] = client

        return client

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter for the current model."""
        # Get state
        state = self._state.get_state()

        # Check if token counter is already in state cache
        if "token_counter" in state.cache and state.cache["token_counter"]:
            return state.cache["token_counter"]

        # Create new token counter
        token_counter = OpenAITokenCounter(model=self.model_name)

        # Store in state cache
        state.cache["token_counter"] = token_counter

        return token_counter

    def invoke(self, prompt: str) -> Any:
        """
        Invoke the model with a prompt and return structured output.

        This is a convenience method that delegates to generate.

        Args:
            prompt: The prompt to generate from

        Returns:
            The generated text
        """
        # Get state
        state = self._state.get_state()

        # Ensure initialized
        if not state.initialized:
            state.initialized = True

        return self.generate(prompt)


def create_openai_provider(
    model_name: str = OpenAIProvider.DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    trace_enabled: bool = True,
    config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    api_client: Optional[APIClient] = None,
    token_counter: Optional[TokenCounter] = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """
    Create an OpenAI model provider.

    This factory function creates an OpenAIProvider with the specified
    configuration options.

    Args:
        model_name: Name of the model to use (e.g., "gpt-4-turbo", "gpt-3.5-turbo")
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum number of tokens to generate
        api_key: OpenAI API key
        trace_enabled: Whether to enable tracing
        config: Optional model configuration
        api_client: Optional API client to use
        token_counter: Optional token counter to use
        **kwargs: Additional configuration parameters

    Returns:
        An OpenAIProvider instance

    Examples:
        ```python
        from sifaka.models.openai import create_openai_provider
        import os

        # Create a provider with default settings
        provider = create_openai_provider(api_key=os.environ.get("OPENAI_API_KEY"))

        # Create a provider with custom settings
        provider = create_openai_provider(
            model_name="gpt-4-turbo",
            temperature=0.8,
            max_tokens=2000,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Generate text
        response = provider.generate("Explain quantum computing in simple terms.")
        print(response)
        ```
    """
    # Try to use standardize_model_config if available
    try:
        from sifaka.utils.config import standardize_model_config

        # Use standardize_model_config to handle different config formats
        model_config = standardize_model_config(
            config=config,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            trace_enabled=trace_enabled,
            **kwargs,
        )
    except (ImportError, AttributeError):
        # Create config manually
        if isinstance(config, ModelConfig):
            model_config = config
        elif isinstance(config, dict):
            model_config = ModelConfig(**config)
        else:
            model_config = ModelConfig(
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                trace_enabled=trace_enabled,
                **kwargs,
            )

    return OpenAIProvider(
        model_name=model_name,
        config=model_config,
        api_client=api_client,
        token_counter=token_counter,
    )
