"""Anthropic model implementation for Sifaka.

This module provides an implementation of the Model protocol for Anthropic models,
supporting the Claude family of models. It handles token counting, error handling,
and configuration for Anthropic models.

The AnthropicModel class implements the Model protocol and provides methods for
generating text and counting tokens. It also provides a configure method for
updating model options after initialization.

Example:
    ```python
    from sifaka.models.anthropic import AnthropicModel, create_anthropic_model

    # Create a model directly
    model1 = AnthropicModel(model_name="claude-3-opus-20240229", api_key="your-api-key")

    # Create a model using the factory function
    model2 = create_anthropic_model(model_name="claude-3-sonnet-20240229", api_key="your-api-key")

    # Generate text
    response = model1.generate(
        "Write a short story about a robot.",
        temperature=0.7,
        max_tokens=500,
        system_message="You are a creative writer."
    )
    print(response)

    # Count tokens
    token_count = model1.count_tokens("This is a test.")
    print(f"Token count: {token_count}")
    ```
"""

from typing import Any, Optional

try:
    from anthropic import Anthropic, APIError, RateLimitError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from sifaka.core.thought import Thought
from sifaka.models.shared import BaseModelImplementation, create_factory_function
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class AnthropicModel(BaseModelImplementation):
    """Anthropic model implementation for generating text and counting tokens.

    This class implements the Model protocol for Anthropic models, supporting
    the Claude family of models. It handles token counting, error handling,
    and configuration for Anthropic models.

    The class also provides methods for counting tokens and configuring the model
    after initialization.

    Example:
        ```python
        from sifaka.models.anthropic import AnthropicModel

        # Create a model
        model = AnthropicModel(
            model_name="claude-3-opus-20240229",
            api_key="your-api-key",
            temperature=0.7
        )

        # Generate text with a system message
        response = model.generate(
            "Write a short story about a robot.",
            system_message="You are a creative writer.",
            max_tokens=500
        )
        print(response)
        ```
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **options: Any,
    ):
        """Initialize the Anthropic model with the specified parameters.

        Args:
            model_name: The name of the Anthropic model to use (e.g., "claude-3-opus-20240229").
            api_key: Optional API key to use. If not provided, it will be read from the
                ANTHROPIC_API_KEY environment variable.
            **options: Additional options to pass to the Anthropic API, such as:
                - temperature: Controls randomness in generation (0.0 to 1.0).
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - top_k: Controls diversity by limiting to top k tokens.

        Raises:
            ConfigurationError: If the Anthropic package is not installed.
            ModelError: If the API key is not provided and not available in the environment.

        Example:
            ```python
            # Create a model with API key from environment variable
            model1 = AnthropicModel(model_name="claude-3-opus-20240229")

            # Create a model with explicit API key and options
            model2 = AnthropicModel(
                model_name="claude-3-sonnet-20240229",
                api_key="your-api-key",
                temperature=0.7,
                max_tokens=500
            )
            ```
        """
        # Check if Anthropic package is available
        if not ANTHROPIC_AVAILABLE:
            from sifaka.utils.error_handling import ConfigurationError

            raise ConfigurationError(
                "Required packages not available: 'anthropic'",
                component="Anthropic",
                operation="initialization",
                suggestions=[
                    "Install missing packages: pip install anthropic",
                    "Check the Anthropic documentation for installation instructions",
                ],
            )

        # Initialize base class with Anthropic-specific configuration
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            **options,
        )

        # Initialize the Anthropic client
        self.client = Anthropic(api_key=self.api_key)

    def _generate_impl(self, prompt: str, **options: Any) -> str:
        """Generate text using the Anthropic API.

        This is the internal implementation called by the base class generate method.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error calling the Anthropic API.
        """
        # Extract system message if provided
        system_message = options.pop("system_message", None)

        # Extract max_tokens if provided
        max_tokens = options.get("max_tokens", 1024)

        try:
            # Extract specific parameters that are supported
            stop_sequences = options.get("stop_sequences", None)

            # Build the API call parameters
            api_params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": options.get("temperature", 0.7),
            }

            # Add system message if provided
            if system_message:
                api_params["system"] = system_message

            # Add stop_sequences if provided
            if stop_sequences:
                api_params["stop_sequences"] = stop_sequences

            # Add other supported parameters
            for param in ["top_p", "top_k"]:
                if param in options:
                    api_params[param] = options[param]

            # Make the API call
            response = self.client.messages.create(
                messages=[{"role": "user", "content": prompt}], **api_params
            )

            # Extract the response text
            response_text: str = response.content[0].text
            return response_text

        except RateLimitError as e:
            # Use base class error handling
            self._handle_api_error(e, "generation")
            raise  # Re-raise after handling

        except APIError as e:
            # Use base class error handling
            self._handle_api_error(e, "generation")
            raise  # Re-raise after handling

        except Exception as e:
            # Use base class error handling
            self._handle_api_error(e, "generation")
            raise  # Re-raise after handling

    def _supports_system_prompt(self) -> bool:
        """Check if the model supports system prompts.

        Returns:
            True since Anthropic models support system prompts.
        """
        return True

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using a simple approximation.

        This method provides a simple approximation of token count for Anthropic models.
        For more accurate token counting, consider using a proper tokenizer.

        Args:
            text: The text to count tokens in.

        Returns:
            The estimated number of tokens in the text.

        Example:
            ```python
            # Count tokens in text
            token_count = model.count_tokens("This is a test.")
            print(f"Token count: {token_count}")
            ```
        """
        # Simple approximation: count words
        # This is not accurate but provides a reasonable estimate
        # In a production environment, you would want to use a proper tokenizer
        return len(text.split()) if text else 0

    # Async methods required by Model protocol
    async def _generate_async(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt asynchronously.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            The generated text.
        """
        # For now, just call the sync method
        # In a real implementation, you would use the async Anthropic client
        return self.generate(prompt, **options)

    async def _generate_with_thought_async(
        self, thought: Thought, **options: Any
    ) -> tuple[str, str]:
        """Generate text using a Thought container asynchronously.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        # For now, just call the sync method
        # In a real implementation, you would use the async Anthropic client
        return self.generate_with_thought(thought, **options)


create_anthropic_model = create_factory_function(
    model_class=AnthropicModel,
    provider_name="Anthropic",
    env_var_name="ANTHROPIC_API_KEY",
    required_packages=["anthropic"] if not ANTHROPIC_AVAILABLE else None,
)
