"""Sifaka: AI-powered text improvement through iterative critique.

Sifaka is a Python framework for improving text using Large Language Models
(LLMs) and research-backed critique techniques. It provides a simple API
for iteratively refining text based on structured feedback from multiple
critics.

## Quick Start:

    >>> from sifaka import improve
    >>> result = await improve("Write about artificial intelligence")
    >>> print(result.final_text)

## Key Features:

- **Multiple Critics**: Choose from various critique strategies like
  reflexion, self-refine, constitutional AI, and more
- **Iterative Improvement**: Automatically refines text through multiple
  rounds based on critic feedback  
- **Quality Validators**: Ensure text meets specific requirements
- **Full Observability**: Track every step of the improvement process
- **Extensible**: Add custom critics, validators, and storage backends

## Main Components:

- `improve()`: The primary async function for text improvement
- `improve_sync()`: Synchronous wrapper for non-async environments
- `Config`: Configuration object for customizing behavior
- `SifakaResult`: Contains improved text and complete audit trail

For more information, see the documentation at:
https://docs.sifaka.ai/
"""

# Core API
from .api import improve, improve_sync

# Core classes
from .core.models import SifakaResult
from .core.config import Config
from .core.interfaces import Validator
from .core.engine import SifakaEngine
from .core.retry import RetryConfig

# Storage
from .storage import StorageBackend, MemoryStorage, FileStorage

# Critics
from .critics import register_critic, CriticRegistry

# Middleware
from .core.middleware import (
    MiddlewarePipeline,
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    RateLimitingMiddleware,
    monitor as monitor_context,
)

# Monitoring
from .core.monitoring import monitor, get_global_monitor, PerformanceMetrics

# Exceptions
from .core.exceptions import (
    SifakaError,
    ConfigurationError,
    ModelProviderError,
    CriticError,
    ValidationError,
    StorageError,
    PluginError,
    TimeoutError,
)

# Plugin system
from .core.plugins import (
    register_storage_backend,
    get_storage_backend,
    list_storage_backends,
    create_storage_backend,
)

# Expose key classes for advanced usage
__all__ = [
    # Main API
    "improve",
    "improve_sync",
    # Core classes
    "SifakaResult",
    "Config",
    "SifakaEngine",
    "Validator",
    "StorageBackend",
    "MemoryStorage",
    "FileStorage",
    # Critics
    "register_critic",
    "CriticRegistry",
    # Retry configuration
    "RetryConfig",
    # Plugin system
    "register_storage_backend",
    "get_storage_backend",
    "list_storage_backends",
    "create_storage_backend",
    # Exceptions
    "SifakaError",
    "ConfigurationError",
    "ModelProviderError",
    "CriticError",
    "ValidationError",
    "StorageError",
    "PluginError",
    "TimeoutError",
    # Middleware
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RateLimitingMiddleware",
    # Monitoring
    "monitor",
    "monitor_context",
    "get_global_monitor",
    "PerformanceMetrics",
]
