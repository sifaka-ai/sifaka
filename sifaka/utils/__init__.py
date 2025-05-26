"""Utilities for Sifaka.

This module provides essential utility functions and classes for the Sifaka framework,
including error handling, logging, and performance monitoring.
"""

from sifaka.utils.error_handling import (
    ChainError,
    ConfigurationError,
    ImproverError,
    ModelAPIError,
    ModelError,
    RetrieverError,
    SifakaError,
    ValidationError,
    chain_context,
    create_actionable_suggestions,
    critic_context,
    enhance_error_message,
    error_context,
    format_error_message,
    log_error,
    model_context,
    validation_context,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin
from sifaka.utils.performance import PerformanceMonitor, time_operation

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
    "error_context",
    "model_context",
    "validation_context",
    "critic_context",
    "chain_context",
    "log_error",
    "format_error_message",
    "create_actionable_suggestions",
    "enhance_error_message",
    # Other utilities
    "get_logger",
    "ContextAwareMixin",
    "time_operation",
    "PerformanceMonitor",
]
