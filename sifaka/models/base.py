"""Base model interface for Sifaka.

This module defines the Model protocol that all model implementations must follow
and provides a factory function for creating model instances based on provider
and model name.

The Model protocol requires two methods:
- generate: Generate text from a prompt
- count_tokens: Count tokens in text

Example:
    ```python
    from sifaka.models.base import Model, create_model

    # Create a model using the factory function
    model = create_model("openai:gpt-4", api_key="your-api-key")

    # Or create a model using a combined provider:model string
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

from typing import Any, Protocol


class Model(Protocol):
    """Protocol defining the interface for language model providers.

    This protocol defines the minimum interface that all language model
    implementations must follow. It requires two methods:
    - generate: Generate text from a prompt
    - count_tokens: Count tokens in text

    All model implementations in Sifaka must implement this protocol.

    Example:
        ```python
        from typing import Protocol, Any

        class Model(Protocol):
            def generate(self, prompt: str, **options: Any) -> str:
                ...

            def count_tokens(self, text: str) -> int:
                ...

        def use_model(model: Model, prompt: str) -> str:
            return model.generate(prompt, temperature=0.7)
        ```
    """

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt (str): The prompt to generate text from.
            **options (Any): Additional options to pass to the model, such as:
                - temperature: Controls randomness (0.0 to 1.0)
                - max_tokens: Maximum number of tokens to generate
                - top_p: Controls diversity via nucleus sampling
                - stop: Sequences where the model should stop generating
                - system_prompt: System prompt for the model

        Returns:
            str: The generated text.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        This method counts the number of tokens in the given text according
        to the model's tokenization scheme. Different models may tokenize
        text differently, so token counts may vary between model implementations.

        Args:
            text (str): The text to count tokens in.

        Returns:
            int: The number of tokens in the text.
        """
        ...


def create_model(provider: str, model_name: str = "", **options: Any) -> Model:
    """Create a model instance based on provider and model name.

    This function creates a model instance for the specified provider and model name.
    It first tries to use the registry to find the appropriate model factory,
    and falls back to direct imports if the registry fails.

    The provider can be specified in two ways:
    1. As separate provider and model_name parameters: create_model("openai", "gpt-4")
    2. As a combined string with provider:model format: create_model("openai:gpt-4")

    If the combined format is used, the model_name parameter should be empty.

    Args:
        provider (str): The provider name (e.g., "openai", "anthropic") or a combined
            provider:model string (e.g., "openai:gpt-4").
        model_name (str): The model name (e.g., "gpt-4", "claude-3"). Not needed if
            using the combined provider:model format.
        **options (Any): Additional options to pass to the model constructor, such as:
            - api_key: API key for the provider
            - temperature: Controls randomness (0.0 to 1.0)
            - max_tokens: Maximum number of tokens to generate

    Returns:
        Model: A model instance implementing the Model protocol.

    Raises:
        ModelNotFoundError: If the provider or model is not found.
        ConfigurationError: If the required package for the provider is not installed.
        ModelError: If there is an error initializing the model.

    Example:
        ```python
        from sifaka.models.base import create_model

        # Create a model using separate provider and model name
        model1 = create_model("openai", "gpt-4", api_key="your-api-key")

        # Create a model using combined provider:model format
        model2 = create_model("openai:gpt-4", api_key="your-api-key")

        # Generate text
        response = model1.generate("Write a short story about a robot.")
        print(response)
        ```
    """
    from sifaka.errors import ConfigurationError, ModelNotFoundError
    from sifaka.factories import create_model as factory_create_model

    # Use the factory function from sifaka.factories
    try:
        return factory_create_model(provider, model_name, **options)
    except Exception:
        # Fall back to direct imports if factory fails
        provider = provider.lower()

        if provider == "openai":
            try:
                from sifaka.models.openai import OPENAI_AVAILABLE, OpenAIModel

                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI package not available")
                return OpenAIModel(model_name=model_name, **options)
            except ImportError:
                raise ConfigurationError(
                    "OpenAI package not installed. Install it with 'pip install openai tiktoken'."
                )
        elif provider == "anthropic":
            try:
                from sifaka.models.anthropic import AnthropicModel

                return AnthropicModel(model_name=model_name, **options)
            except ImportError:
                raise ConfigurationError(
                    "Anthropic package not installed or Anthropic model not yet implemented."
                )
        elif provider == "gemini":
            try:
                from sifaka.models.gemini import GeminiModel

                return GeminiModel(model_name=model_name, **options)
            except ImportError:
                raise ConfigurationError(
                    "Google Gemini package not installed or Gemini model not yet implemented."
                )
        elif provider == "mock":
            # Create a simple mock model for testing
            class MockModel:
                def __init__(self, model_name: str, **kwargs: Any):
                    self.model_name = model_name
                    self.kwargs = kwargs

                def generate(self, prompt: str, **_: Any) -> str:
                    return f"Mock response from {self.model_name} for: {prompt}"

                def count_tokens(self, text: str) -> int:
                    return len(text.split())

            return MockModel(model_name, **options)
        else:
            raise ModelNotFoundError(f"Provider '{provider}' not found")
