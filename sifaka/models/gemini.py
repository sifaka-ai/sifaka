"""
Google Gemini model provider implementation.

This module provides the GeminiProvider class which implements the ModelProviderCore
interface for Google Gemini models.
"""

from typing import Optional, Dict, Any, ClassVar, Union

import google.generativeai as genai
import tiktoken
from pydantic import PrivateAttr

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_model_state

logger = get_logger(__name__)


class GeminiClient(APIClient):
    """Gemini API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Gemini client."""
        genai.configure(api_key=api_key)
        self.model = None  # Lazy initialization
        logger.debug("Initialized Gemini client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to Gemini and return the response."""
        try:
            if self.model is None:
                self.model = genai.GenerativeModel(
                    model_name="gemini-pro",
                    generation_config={
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_tokens,
                    },
                )

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise


class GeminiTokenCounter(TokenCounter):
    """Token counter using tiktoken for Gemini models."""

    def __init__(self, model: str = "gemini-pro") -> None:
        """Initialize the token counter for a specific model."""
        try:
            # Gemini uses similar tokenization to GPT-3.5
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug(f"Initialized token counter for model {model}")
        except Exception as e:
            logger.error(f"Error initializing token counter: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in the text using the model's encoding."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise


class GeminiProvider(ModelProviderCore):
    """
    Google Gemini model provider implementation.

    This provider supports Gemini Pro models with configurable parameters
    and built-in token counting.
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "gemini-pro"

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_model_state)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the Gemini provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
        """
        # Verify Google Generative AI package is installed
        try:
            # Just importing the package to verify it's installed
            # We already imported it at the module level
            pass
        except ImportError:
            raise ImportError(
                "Google Generative AI package is required. Install with: pip install google-generativeai"
            )

        # Initialize state
        state = self._state_manager.get_state()
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
        """
        Create a default Gemini client.

        Returns:
            A default Gemini API client
        """
        # Get state
        state = self._state_manager.get_state()

        # Check if client is already in state cache
        if "api_client" in state.cache and state.cache["api_client"]:
            return state.cache["api_client"]

        # Create new client
        client = GeminiClient(api_key=self.config.api_key)

        # Store in state cache
        state.cache["api_client"] = client

        return client

    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter for the current model.

        Returns:
            A default token counter for Gemini models
        """
        # Get state
        state = self._state_manager.get_state()

        # Check if token counter is already in state cache
        if "token_counter" in state.cache and state.cache["token_counter"]:
            return state.cache["token_counter"]

        # Create new token counter
        token_counter = GeminiTokenCounter(model=self.model_name)

        # Store in state cache
        state.cache["token_counter"] = token_counter

        return token_counter

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt and return structured output.

        This is a convenience method that delegates to generate.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            state.initialized = True

        return self.generate(prompt, **kwargs)


def create_gemini_provider(
    model_name: str = GeminiProvider.DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    trace_enabled: bool = True,
    config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    api_client: Optional[APIClient] = None,
    token_counter: Optional[TokenCounter] = None,
    **kwargs: Any,
) -> GeminiProvider:
    """
    Create a Google Gemini model provider.

    This factory function creates a GeminiProvider with the specified
    configuration options.

    Args:
        model_name: Name of the model to use (e.g., "gemini-pro")
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum number of tokens to generate
        api_key: Google API key
        trace_enabled: Whether to enable tracing
        config: Optional model configuration
        api_client: Optional API client to use
        token_counter: Optional token counter to use
        **kwargs: Additional configuration parameters

    Returns:
        A GeminiProvider instance

    Examples:
        ```python
        from sifaka.models.gemini import create_gemini_provider
        import os

        # Create a provider with default settings
        provider = create_gemini_provider(api_key=os.environ.get("GOOGLE_API_KEY"))

        # Create a provider with custom settings
        provider = create_gemini_provider(
            model_name="gemini-pro",
            temperature=0.8,
            max_tokens=2000,
            api_key=os.environ.get("GOOGLE_API_KEY")
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

    return GeminiProvider(
        model_name=model_name,
        config=model_config,
        api_client=api_client,
        token_counter=token_counter,
    )
