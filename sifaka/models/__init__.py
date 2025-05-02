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

from .anthropic import AnthropicProvider
from .base import ModelProvider, APIClient, TokenCounter, ModelConfig
from .core import ModelProviderCore
from .gemini import GeminiProvider
from .openai import OpenAIProvider

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
    "LanguageModel",
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
