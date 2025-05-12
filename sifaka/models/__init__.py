"""
Model provider integrations for Sifaka.

This package provides model provider implementations for different LLM services:
- OpenAIProvider: Provider for OpenAI models
- AnthropicProvider: Provider for Anthropic models
- GeminiProvider: Provider for Google Gemini models

It also provides base classes and interfaces:
- ModelProvider: Abstract base class for model providers
- ModelProviderCore: Core implementation with component-based architecture
- Interfaces: Protocol interfaces for model providers, clients, and token counters
"""

# Import removed to fix circular dependency
# from sifaka.critics.prompt import LanguageModel

# Import base classes and interfaces
from .base import ModelProvider
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.utils.config.models import ModelConfig
from .core.provider import ModelProviderCore
from sifaka.interfaces import ModelProviderProtocol, APIClientProtocol, TokenCounterProtocol

# Import providers
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.mock import MockProvider

# Import factory functions
from .factories import (
    create_model_provider,
    create_openai_provider,
    create_anthropic_provider,
    create_gemini_provider,
    create_mock_provider,
)

# Import managers and services for advanced usage
from .managers import ClientManager, TokenCounterManager, TracingManager
from .services import GenerationService

# Import result models
from .result import ModelResult, GenerationResult, TokenCountResult

# Import utility functions
from .utils import create_generation_result, create_error_result, try_generate

__all__ = [
    # Main provider classes
    "ModelProvider",
    "ModelProviderCore",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MockProvider",
    # "LanguageModel",  # Removed to fix circular dependency
    # Factory functions
    "create_model_provider",
    "create_openai_provider",
    "create_anthropic_provider",
    "create_gemini_provider",
    "create_mock_provider",
    # Base interfaces and configs
    "APIClient",
    "TokenCounter",
    "ModelConfig",
    # Protocol interfaces
    "ModelProviderProtocol",
    "APIClientProtocol",
    "TokenCounterProtocol",
    # Managers
    "ClientManager",
    "TokenCounterManager",
    "TracingManager",
    # Services
    "GenerationService",
    # Result models
    "ModelResult",
    "GenerationResult",
    "TokenCountResult",
    # Utility functions
    "create_generation_result",
    "create_error_result",
    "try_generate",
]
