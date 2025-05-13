"""
Error Logging Utilities

This module provides utilities for configuring and using error logging
in the Sifaka framework. It includes functions for configuring error logging,
creating error loggers, and formatting error logs.

## Functions
- **configure_error_logging**: Configure error logging settings
- **get_error_logger**: Get a logger configured for error logging
"""

import logging
from typing import Any, Dict, Optional

from ..logging import get_logger


def configure_error_logging(
    log_level: str = "ERROR",
    include_traceback: bool = True,
    log_format: Optional[Optional[str]] = None,
    additional_handlers: Optional[Optional[list]] = None,
) -> None:
    """Configure error logging settings.

    This function configures error logging settings for the Sifaka framework,
    including log level, format, and handlers.

    Args:
        log_level: Log level to use for errors (default: "ERROR")
        include_traceback: Whether to include traceback in error logs
        log_format: Custom log format to use
        additional_handlers: Additional log handlers to add

    Examples:
        ```python
        from sifaka.utils.errors.logging import configure_error_logging

        # Configure error logging with default settings
        configure_error_logging()

        # Configure error logging with custom settings
        configure_error_logging(
            log_level="WARNING",
            include_traceback=False,
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configure error logging with additional handlers
        file_handler = logging.FileHandler("errors.log")
        configure_error_logging(additional_handlers=[file_handler])
        ```
    """
    # Get the root logger
    root_logger = logging.getLogger("sifaka")

    # Set log level
    if root_logger:
        root_logger.setLevel(getattr(logging, log_level.upper() if log_level else "ERROR"))

    # Configure log format
    if log_format is None:
        if include_traceback:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s"
        else:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Configure handlers
    for handler in root_logger.handlers:
        if handler:
            handler.setFormatter(formatter)

    # Add additional handlers
    if additional_handlers:
        for handler in additional_handlers:
            if handler:
                handler.setFormatter(formatter)
                if root_logger:
                    root_logger.addHandler(handler)


def get_error_logger(name: str) -> logging.Logger:
    """Get a logger configured for error logging.

    This function returns a logger configured for error logging,
    using the standard Sifaka logging configuration.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        A configured logger instance

    Examples:
        ```python
        from sifaka.utils.errors.logging import get_error_logger

        # Get a logger for the current module
        logger = get_error_logger(__name__)

        # Log an error
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            if logger:
                logger.error(f"Error processing data: {str(e)}", exc_info=True)
        ```
    """
    return get_logger(name)


def format_error_metadata(metadata: Dict[str, Any]) -> str:
    """Format error metadata for logging.

    This function formats error metadata for logging, creating a
    human-readable string representation of the metadata.

    Args:
        metadata: Error metadata dictionary

    Returns:
        Formatted metadata string

    Examples:
        ```python
        from sifaka.utils.errors.logging import format_error_metadata

        # Format error metadata
        metadata = {
            "error_type": "ValidationError",
            "component": "TextValidator",
            "field": "text",
            "max_length": 100
        }
        formatted = format_error_metadata(metadata)
        print(formatted)
        ```
    """
    lines = []
    if metadata:
        for key, value in metadata.items():
            if key == "traceback":
                continue  # Skip traceback in formatted output
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
