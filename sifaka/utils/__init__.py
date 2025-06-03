"""Utility functions and helpers for Sifaka.

This module provides common utilities:
- Error handling and exception hierarchy
- Configuration management
- Logging configuration
- Input validation helpers
"""

from sifaka.utils.errors import (
    SifakaError,
    ValidationError,
    CritiqueError,
    GraphExecutionError,
    ConfigurationError,
)
from sifaka.utils.config import SifakaConfig
from sifaka.utils.validation import (
    validate_prompt,
    validate_max_iterations,
    validate_model_name,
    validate_timeout,
)

# ValidationAwareMixin imported separately to avoid circular imports
from sifaka.utils.logging import (
    get_logger,
    setup_logging,
    configure_for_development,
    configure_for_production,
)

__all__ = [
    # Error types
    "SifakaError",
    "ValidationError",
    "CritiqueError",
    "GraphExecutionError",
    "ConfigurationError",
    # Configuration
    "SifakaConfig",
    # Validation helpers
    "validate_prompt",
    "validate_max_iterations",
    "validate_model_name",
    "validate_timeout",
    # Validation context - import separately to avoid circular imports
    # "ValidationAwareMixin",
    # Logging
    "get_logger",
    "setup_logging",
    "configure_for_development",
    "configure_for_production",
]
