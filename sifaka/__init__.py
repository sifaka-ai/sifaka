"""Sifaka: AI-powered text improvement through iterative critique.

__version__ = "0.1.6"

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
https://sifaka-ai.github.io/sifaka/
"""

# Configure monitoring before any imports
import os

# Set up OpenTelemetry service info early
os.environ.setdefault("OTEL_SERVICE_NAME", "sifaka")
os.environ.setdefault("OTEL_SERVICE_VERSION", "0.1.6")

from dotenv import load_dotenv

# Core API
from .api import improve, improve_sync
from .core.config import Config
from .core.engine import SifakaEngine

# Exceptions
from .core.exceptions import (
    ConfigurationError,
    CriticError,
    ModelProviderError,
    PluginError,
    SifakaError,
    StorageError,
    TimeoutError,
    ValidationError,
)
from .core.interfaces import Validator

# Middleware
from .core.middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewarePipeline,
    RateLimitingMiddleware,
    monitor as monitor_context,
)

# Core classes
from .core.models import SifakaResult

# Monitoring
from .core.monitoring import PerformanceMetrics, get_global_monitor, monitor

# Plugin system
from .core.plugins import (
    create_storage_backend,
    get_storage_backend,
    list_storage_backends,
    register_storage_backend,
)
from .core.retry import RetryConfig

# Type definitions
from .core.types import CriticType, Provider, StorageType, ValidatorType

# Critics
from .critics import CriticRegistry, register_critic

# Storage
from .storage import FileStorage, MemoryStorage, StorageBackend

# Load environment variables from .env file
load_dotenv()

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
    # Type definitions
    "CriticType",
    "ValidatorType",
    "StorageType",
    "Provider",
]
