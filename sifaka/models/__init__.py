"""
Model implementations for Sifaka.

This module provides model implementations for various language model providers,
including OpenAI, Anthropic, and Google Gemini. Each model implementation follows
the Model interface and can be created through the factory functions.

Example:
    ```python
    from sifaka.models import create_model
    from sifaka.core.thought import Thought

    # Create a model using the factory function
    model = create_model("openai:gpt-4", api_key="your-api-key")

    # Create a thought
    thought = Thought(prompt="Write a short story about a robot.")

    # Generate text
    text = model.generate(thought)
    print(text)
    ```
"""

from sifaka.models.anthropic_model import AnthropicModel
from sifaka.models.factory import (
    ModelConfigurationError,
    ModelError,
    ModelNotFoundError,
    create_anthropic_model,
    create_gemini_model,
    create_mock_model,
    create_model,
    create_openai_model,
)
from sifaka.models.mock_model import MockModel
from sifaka.models.openai_model import OpenAIModel

# Try to import optional models
try:
    from sifaka.models.gemini_model import GeminiModel
except ImportError:
    pass

__all__ = [
    # Factory functions
    "create_model",
    "create_openai_model",
    "create_anthropic_model",
    "create_gemini_model",
    "create_mock_model",
    # Model implementations
    "OpenAIModel",
    "AnthropicModel",
    "MockModel",
    # Exceptions
    "ModelError",
    "ModelNotFoundError",
    "ModelConfigurationError",
]
