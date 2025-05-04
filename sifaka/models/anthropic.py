"""
Anthropic model provider implementation.

This module provides the AnthropicProvider class which implements the ModelProviderCore
interface for Anthropic Claude models, and additional Anthropic-specific functionality
like text reflection and analysis.
"""

from typing import Optional, Dict, Any, ClassVar, Union
import os

import anthropic
import tiktoken
from anthropic import Anthropic
from pydantic import BaseModel, PrivateAttr

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_model_state

logger = get_logger(__name__)


# ReflectionResult class from integrations/anthropic.py
class ReflectionResult(BaseModel):
    """Result of a text reflection operation."""

    text: str
    analysis: Dict[str, Any]
    suggestions: Optional[Dict[str, Any]] = None
    safety_score: Optional[float] = None


class AnthropicClient(APIClient):
    """Anthropic API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Anthropic client."""
        # Check for API key in environment if not provided
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            logger.debug(f"Retrieved API key from environment: {api_key[:10]}...")

        # Validate API key
        if not api_key:
            logger.warning(
                "No Anthropic API key provided and ANTHROPIC_API_KEY environment variable not set"
            )
        elif not api_key.startswith("sk-ant-api"):
            logger.warning(
                f"API key format appears incorrect. Expected to start with 'sk-ant-api', got: {api_key[:10]}..."
            )

        self.client = Anthropic(api_key=api_key)
        logger.debug("Initialized Anthropic client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to Anthropic and return the response."""
        # Get API key from config or client
        api_key = config.api_key or getattr(self.client, "api_key", None)
        logger.debug(f"Using API key: {api_key[:10]}...")

        # Check for missing API key
        if not api_key:
            raise ValueError(
                "No API key provided. Please provide an API key either by setting the "
                "ANTHROPIC_API_KEY environment variable or by passing it explicitly."
            )

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            return response.content[0].text
        except anthropic.AnthropicError as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise


class AnthropicTokenCounter(TokenCounter):
    """Token counter using tiktoken for Anthropic models."""

    def __init__(self, model: str = "claude-3-opus-20240229") -> None:
        """Initialize the token counter for a specific model."""
        try:
            # Anthropic uses cl100k_base encoding
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


class AnthropicReflector:
    """Reflector that uses Anthropic's API for text analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the Anthropic reflector.

        Args:
            api_key: Anthropic API key
            model: Model to use for reflection
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def reflect(self, text: str) -> ReflectionResult:
        """
        Reflect on the given text using Anthropic's API.

        Args:
            text: Text to reflect on

        Returns:
            ReflectionResult containing analysis and suggestions
        """
        try:
            # Prepare the prompt
            prompt = f"""Please analyze the following text and provide:
1. A detailed analysis of its content, style, and tone
2. Suggestions for improvement
3. A safety score (0-1) indicating potential issues

Text to analyze:
{text}

Please provide your analysis in a structured format."""

            # Call the API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse the response
            analysis = {
                "content": response.content[0].text,
                "model": self.model,
                "temperature": self.temperature,
            }

            return ReflectionResult(
                text=text,
                analysis=analysis,
                safety_score=0.8,  # Placeholder - would need to parse from response
            )

        except Exception as e:
            logger.error(f"Error in Anthropic reflection: {str(e)}")
            raise


class AnthropicProvider(ModelProviderCore):
    """
    Anthropic model provider implementation.

    This provider supports Claude models with configurable parameters
    and built-in token counting.
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
            # Just importing the package to verify it's installed
            # We already imported it at the module level
            pass
        except ImportError:
            raise ImportError("Anthropic package is required. Install with: pip install anthropic")

        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        This method is needed for compatibility with the critique service
        which expects an 'invoke' method.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        return self.generate(prompt, **kwargs)

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
        if hasattr(self, "agenerate"):
            return await self.agenerate(prompt, **kwargs)

        # Fall back to synchronous generate
        return self.generate(prompt, **kwargs)

    def _create_default_client(self) -> APIClient:
        """Create a default Anthropic client."""
        return AnthropicClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter for the current model."""
        return AnthropicTokenCounter(model=self.model_name)

    def create_reflector(
        self, temperature: float = 0.7, max_tokens: int = 1000
    ) -> AnthropicReflector:
        """
        Create a reflector for text analysis.

        Args:
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            An AnthropicReflector instance
        """
        return AnthropicReflector(
            api_key=self.config.api_key,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def create_anthropic_provider(
    model_name: str = AnthropicProvider.DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    trace_enabled: bool = True,
    config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    api_client: Optional[APIClient] = None,
    token_counter: Optional[TokenCounter] = None,
    **kwargs: Any,
) -> AnthropicProvider:
    """
    Create an Anthropic model provider.

    This factory function creates an AnthropicProvider with the specified
    configuration options.

    Args:
        model_name: Name of the model to use (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229")
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum number of tokens to generate
        api_key: Anthropic API key
        trace_enabled: Whether to enable tracing
        config: Optional model configuration
        api_client: Optional API client to use
        token_counter: Optional token counter to use
        **kwargs: Additional configuration parameters

    Returns:
        An AnthropicProvider instance

    Examples:
        ```python
        from sifaka.models.anthropic import create_anthropic_provider
        import os

        # Create a provider with default settings
        provider = create_anthropic_provider(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Create a provider with custom settings
        provider = create_anthropic_provider(
            model_name="claude-3-sonnet-20240229",
            temperature=0.8,
            max_tokens=2000,
            api_key=os.environ.get("ANTHROPIC_API_KEY")
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

    return AnthropicProvider(
        model_name=model_name,
        config=model_config,
        api_client=api_client,
        token_counter=token_counter,
    )
