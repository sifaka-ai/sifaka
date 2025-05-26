"""Utilities for Sifaka.

This module provides essential utility functions and classes for the Sifaka framework,
including error handling, logging, and performance monitoring.
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
    error_context,
    model_context,
    validation_context,
    critic_context,
    chain_context,
    log_error,
    format_error_message,
    create_actionable_suggestions,
    enhance_error_message,
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
