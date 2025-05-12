from typing import Any, List
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
from .base import ModelProvider
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.utils.config.models import ModelConfig
from .core.provider import ModelProviderCore
from sifaka.interfaces import ModelProviderProtocol, APIClientProtocol, TokenCounterProtocol
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.mock import MockProvider
from .factories import create_model_provider, create_openai_provider, create_anthropic_provider, create_gemini_provider, create_mock_provider
from .managers import ClientManager, TokenCounterManager, TracingManager
from .services import GenerationService
from .result import ModelResult, GenerationResult, TokenCountResult
from .utils import create_generation_result, create_error_result, try_generate
__all__: List[Any] = ['ModelProvider', 'ModelProviderCore',
    'OpenAIProvider', 'AnthropicProvider', 'GeminiProvider', 'MockProvider',
    'create_model_provider', 'create_openai_provider',
    'create_anthropic_provider', 'create_gemini_provider',
    'create_mock_provider', 'APIClient', 'TokenCounter', 'ModelConfig',
    'ModelProviderProtocol', 'APIClientProtocol', 'TokenCounterProtocol',
    'ClientManager', 'TokenCounterManager', 'TracingManager',
    'GenerationService', 'ModelResult', 'GenerationResult',
    'TokenCountResult', 'create_generation_result', 'create_error_result',
    'try_generate']
