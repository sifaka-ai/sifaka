"""
Model provider integrations for Sifaka.

This package provides model provider implementations for different LLM services:
- OpenAIProvider: Provider for OpenAI models
- AnthropicProvider: Provider for Anthropic models
- GeminiProvider: Provider for Google Gemini models

It also provides base classes and interfaces:
- ModelProvider: Abstract base class for model providers
- ModelProviderCore: Core implementation with component-based architecture
"""

from sifaka.critics.prompt import LanguageModel

from .anthropic import AnthropicProvider, create_anthropic_provider
from .base import ModelProvider, APIClient, TokenCounter, ModelConfig
from .core import ModelProviderCore
from .gemini import GeminiProvider, create_gemini_provider
from .mock import MockProvider, create_mock_provider
from .openai import OpenAIProvider, create_openai_provider

# Import managers and services for advanced usage
from .managers import ClientManager, TokenCounterManager, TracingManager
from .services import GenerationService

__all__ = [
    # Main provider classes
    "ModelProvider",
    "ModelProviderCore",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MockProvider",
    "LanguageModel",
    # Factory functions
    "create_openai_provider",
    "create_anthropic_provider",
    "create_gemini_provider",
    "create_mock_provider",
    # Base interfaces and configs
    "APIClient",
    "TokenCounter",
    "ModelConfig",
    # Managers
    "ClientManager",
    "TokenCounterManager",
    "TracingManager",
    # Services
    "GenerationService",
]
