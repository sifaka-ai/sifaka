"""Utilities for Sifaka.

This module provides various utility functions and classes for the Sifaka framework,
including error handling, logging, performance monitoring, and enhanced error recovery.
"""

from sifaka.utils.error_handling import (
    SifakaError,
    ModelError,
    ModelAPIError,
    ValidationError,
    ImproverError,
    RetrieverError,
    ChainError,
    ConfigurationError,
    CircuitBreakerError,
    RetryExhaustedError,
    FallbackError,
    ServiceUnavailableError,
    DegradedServiceError,
    error_context,
    model_context,
    validation_context,
    critic_context,
    chain_context,
    circuit_breaker_context,
    retry_context,
    fallback_context,
    log_error,
    format_error_message,
    create_actionable_suggestions,
    enhance_error_message,
)

from sifaka.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerStats,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)

from sifaka.utils.retry import (
    RetryConfig,
    RetryManager,
    RetryStats,
    BackoffStrategy,
    retry_with_backoff,
    DEFAULT_RETRY_CONFIG,
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
    API_RETRY_CONFIG,
)

from sifaka.utils.fallback import (
    FallbackChain,
    FallbackConfig,
    FallbackOption,
    FallbackStats,
    FallbackStrategy,
    create_model_fallback_chain,
    create_retriever_fallback_chain,
)

from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin
from sifaka.utils.performance import time_operation, PerformanceMonitor

__all__ = [
    # Error handling
    "SifakaError",
    "ModelError",
    "ModelAPIError",
    "ValidationError",
    "ImproverError",
    "RetrieverError",
    "ChainError",
    "ConfigurationError",
    "CircuitBreakerError",
    "RetryExhaustedError",
    "FallbackError",
    "ServiceUnavailableError",
    "DegradedServiceError",
    "error_context",
    "model_context",
    "validation_context",
    "critic_context",
    "chain_context",
    "circuit_breaker_context",
    "retry_context",
    "fallback_context",
    "log_error",
    "format_error_message",
    "create_actionable_suggestions",
    "enhance_error_message",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitBreakerStats",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    # Retry mechanisms
    "RetryConfig",
    "RetryManager",
    "RetryStats",
    "BackoffStrategy",
    "retry_with_backoff",
    "DEFAULT_RETRY_CONFIG",
    "AGGRESSIVE_RETRY_CONFIG",
    "CONSERVATIVE_RETRY_CONFIG",
    "API_RETRY_CONFIG",
    # Fallback mechanisms
    "FallbackChain",
    "FallbackConfig",
    "FallbackOption",
    "FallbackStats",
    "FallbackStrategy",
    "create_model_fallback_chain",
    "create_retriever_fallback_chain",
    # Other utilities
    "get_logger",
    "ContextAwareMixin",
    "time_operation",
    "PerformanceMonitor",
]
