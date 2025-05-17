"""Model implementations for different LLM providers.

This module provides model implementations for various language model providers,
including OpenAI, Anthropic, and Google Gemini. Each model implementation follows
the Model protocol and registers itself with the registry system for dependency
injection.

The models can be created either directly through their constructors or through
the create_model factory function, which uses the registry to create the appropriate
model based on the provider prefix.

Example:
    ```python
    from sifaka.models import create_model
    from sifaka.models.openai import OpenAIModel

    # Create a model directly
    model1 = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Create a model using the factory function
    model2 = create_model("openai:gpt-4", api_key="your-api-key")

    # Generate text
    response = model1.generate("Write a short story about a robot.")
    print(response)

    # Count tokens
    token_count = model1.count_tokens("This is a test.")
    print(f"Token count: {token_count}")
    ```
"""

from typing import Any
from sifaka.models.base import Model, create_model
from sifaka.registry import register_model

# Import models with error handling
__all__ = ["Model", "create_model"]

# OpenAI
try:
    from sifaka.models.openai import OpenAIModel, OPENAI_AVAILABLE, create_openai_model

    if OPENAI_AVAILABLE:
        __all__.append("OpenAIModel")
        __all__.append("create_openai_model")
except ImportError:
    pass

# Anthropic
try:
    from sifaka.models.anthropic import AnthropicModel, create_anthropic_model

    __all__.append("AnthropicModel")
    __all__.append("create_anthropic_model")
except ImportError:
    pass

# Google Gemini
try:
    from sifaka.models.gemini import GeminiModel, create_gemini_model

    __all__.append("GeminiModel")
    __all__.append("create_gemini_model")
except ImportError:
    pass


# Register mock model factory
@register_model("mock")
def create_mock_model(model_name: str, **options: Any) -> Any:
    """Create a mock model for testing.

    This function creates a simple mock model that can be used for testing
    without requiring actual API calls to language model providers. The mock
    model implements the same interface as real models but returns a predefined
    response.

    Args:
        model_name (str): The name of the mock model.
        **options (Any): Additional options to pass to the mock model.

    Returns:
        Any: A mock model instance that implements the Model protocol.

    Example:
        ```python
        from sifaka.models import create_model

        # Create a mock model
        model = create_model("mock:test-model")

        # Generate text
        response = model.generate("Write a short story about a robot.")
        print(response)  # "Mock response from test-model for: Write a short story about a robot."
        ```
    """

    class MockModel:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

        def generate(self, prompt: str, **_: Any) -> str:
            return f"Mock response from {self.model_name} for: {prompt}"

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    return MockModel(model_name, **options)


__all__.append("create_mock_model")
