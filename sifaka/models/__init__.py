"""
Model implementations for different LLM providers.

This module provides model implementations for various LLM providers.
Each model implementation registers itself with the registry.
"""

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
def create_mock_model(model_name: str, **options):
    """Create a mock model for testing."""

    class MockModel:
        def __init__(self, model_name: str, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

        def generate(self, prompt: str, **_) -> str:
            return f"Mock response from {self.model_name} for: {prompt}"

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    return MockModel(model_name, **options)


__all__.append("create_mock_model")
