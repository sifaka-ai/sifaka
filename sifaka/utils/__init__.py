"""
Utility functions and classes for the Sifaka framework.

This module provides various utility functions and classes that are used throughout
the Sifaka framework. These utilities include configuration management, logging,
state management, error handling, and other common functionality.
"""

from sifaka.utils.config import standardize_rule_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager
from sifaka.utils.errors import (
    SifakaError,
    ValidationError,
    ConfigurationError,
    RuntimeError,
    TimeoutError,
    ModelError,
    ClassifierError,
    CriticError,
    ChainError,
    format_error_metadata,
    handle_errors,
    with_error_handling,
)

__all__ = [
    # Configuration utilities
    "standardize_rule_config",
    # Logging utilities
    "get_logger",
    # State management
    "StateManager",
    # Error handling - base exceptions
    "SifakaError",
    "ValidationError",
    "ConfigurationError",
    "RuntimeError",
    "TimeoutError",
    # Error handling - component-specific exceptions
    "ModelError",
    "ClassifierError",
    "CriticError",
    "ChainError",
    # Error handling utilities
    "format_error_metadata",
    "handle_errors",
    "with_error_handling",
]
