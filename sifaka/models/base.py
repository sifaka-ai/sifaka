"""Base model interface for Sifaka.

This module defines the Model protocol that all model implementations must follow
and provides a factory function for creating model instances based on provider
and model name.

The Model protocol requires the following methods:
- generate: Generate text from a prompt
- generate_with_thought: Generate text using a Thought container
- count_tokens: Count tokens in text

Example:
    ```python
    from sifaka.models.base import create_model

    # Create a model using the factory function
    model = create_model("openai:gpt-4", api_key="your-api-key")

    # Generate text
    response = model.generate(
        "Write a short story about a robot.",
        temperature=0.7,
        max_tokens=500
    )
    print(response)

    # Count tokens
    token_count = model.count_tokens("This is a test.")
    print(f"Token count: {token_count}")
    ```
"""

from typing import Any

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.utils.error_handling import ConfigurationError
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


def create_model(
    model_spec: str,
    **kwargs: Any,
) -> Model:
    """Create a model instance based on provider and model name.

    This factory function creates a model instance based on the provider and model name.
    The model_spec can be in the format "provider:model_name" or just "model_name".
    If only "model_name" is provided, the provider is inferred from the model name.

    Args:
        model_spec: The model specification in the format "provider:model_name" or "model_name".
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        A model instance.

    Raises:
        ConfigurationError: If the provider is not supported or the model cannot be created.
    """
    # Parse the model specification
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        # Infer provider from model name
        model_name = model_spec
        if model_name.startswith("gpt-"):
            provider = "openai"
        elif model_name.startswith("claude-"):
            provider = "anthropic"
        else:
            provider = "mock"  # Default to mock for testing

    # Create the model based on the provider
    try:
        if provider == "openai":
            # Import the OpenAI model implementation
            from sifaka.models.openai import create_openai_model

            return create_openai_model(model_name=model_name, **kwargs)
        elif provider == "anthropic":
            # Import the Anthropic model implementation
            from sifaka.models.anthropic import create_anthropic_model

            return create_anthropic_model(model_name=model_name, **kwargs)  # type: ignore
        elif provider == "huggingface":
            # Import the HuggingFace model implementation
            from sifaka.models.huggingface import create_huggingface_model

            return create_huggingface_model(model_name=model_name, **kwargs)
        elif provider == "ollama":
            # Import the Ollama model implementation
            from sifaka.models.ollama import create_ollama_model

            return create_ollama_model(model_name=model_name, **kwargs)
        elif provider == "mock":
            # Create a mock model for testing
            return MockModel(model_name=model_name, **kwargs)
        else:
            raise ConfigurationError(
                f"Unsupported model provider: {provider}",
                suggestions=[
                    "Use 'openai' for OpenAI models",
                    "Use 'anthropic' for Anthropic models",
                    "Use 'huggingface' for HuggingFace models",
                    "Use 'ollama' for Ollama models",
                    "Use 'mock' for mock models",
                ],
            )
    except ImportError as e:
        # Handle import errors gracefully
        raise ConfigurationError(
            f"Failed to import model implementation for provider '{provider}': {str(e)}",
            suggestions=[
                f"Install the required package for {provider} models",
                "Check that the package is installed correctly",
            ],
        ) from e
    except Exception as e:
        # Handle other errors
        raise ConfigurationError(
            f"Failed to create model for provider '{provider}': {str(e)}",
            suggestions=[
                "Check that the model name is correct",
                "Check that the API key is correct",
                "Check that the provider is supported",
            ],
        ) from e


class MockModel(ContextAwareMixin):
    """Mock model implementation for testing.

    This class provides a simple mock implementation of the Model protocol
    for testing purposes. It returns predefined responses for generate and
    count_tokens methods.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize the mock model.

        Args:
            model_name: The name of the model.
            **kwargs: Additional keyword arguments including response_text.
        """
        self.model_name = model_name
        self.kwargs = kwargs
        # Support custom response text for testing
        self.response_text = kwargs.get("response_text", None)
        logger.debug(f"Created mock model with name: {model_name}")

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            A mock response.
        """
        logger.debug(f"Generating text with mock model: {self.model_name}")
        if self.response_text is not None:
            return str(self.response_text)
        return f"Mock response from {self.model_name} for: {prompt[:50]}..."

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using a Thought container.

        The model no longer handles retrieval - the Chain orchestrates all retrieval.
        The model just uses whatever context is already in the Thought container.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        logger.debug(f"Generating text with mock model using Thought: {self.model_name}")

        # Use mixin to build contextualized prompt
        full_prompt = self._build_contextualized_prompt(thought, max_docs=5)

        # Add system prompt if available
        if thought.system_prompt:
            full_prompt = f"{thought.system_prompt}\n\n{full_prompt}"

        # Log context usage
        if self._has_context(thought):
            context_summary = self._get_context_summary(thought)
            logger.debug(f"MockModel using context: {context_summary}")

        if self.response_text is not None:
            generated_text = self.response_text
        else:
            generated_text = f"Mock response from {self.model_name} for: {full_prompt[:50]}..."
        return generated_text, full_prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens in.

        Returns:
            A mock token count.
        """
        # Simple approximation: count words
        return len(text.split())

    # Internal async methods (implementing the Model protocol)
    async def _generate_async(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt asynchronously.

        This is the internal async implementation for the mock model.
        Since it's a mock, we just call the sync version.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            A mock response.
        """
        return self.generate(prompt, **options)

    async def _generate_with_thought_async(
        self, thought: "Thought", **options: Any
    ) -> tuple[str, str]:
        """Generate text from a thought asynchronously.

        This is the internal async implementation for the mock model.
        Since it's a mock, we just call the sync version.

        Args:
            thought: The Thought container with prompt and context.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        return self.generate_with_thought(thought, **options)

    async def _count_tokens_async(self, text: str) -> int:
        """Count tokens in text asynchronously.

        This is the internal async implementation for the mock model.
        Since it's a mock, we just call the sync version.

        Args:
            text: The text to count tokens in.

        Returns:
            A mock token count.
        """
        return self.count_tokens(text)
