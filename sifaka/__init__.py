"""Sifaka: Simple AI text improvement through research-backed critique.

This is the main API for Sifaka - simple functions that do everything you need.
"""

# Core API
from .api import improve, improve_sync, improve_advanced

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
    "improve_advanced",
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
