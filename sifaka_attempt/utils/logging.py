"""
Logging utilities for Sifaka.

This module provides simple functions for configuring logging throughout
the Sifaka library.
"""

import logging
import sys
from typing import Optional, Dict, Any


def configure_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    log_file: Optional[str] = None,
    handlers: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Configure the Python logging system for Sifaka.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom log format string
        log_file: Path to a log file (if not provided, logs go to stderr)
        handlers: Additional logging handlers to add
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(format_str)

    # Create and configure handlers
    if not root_logger.handlers:
        # Create console handler if no handlers exist
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        # Create file handler if a log file is specified
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if handlers:
        # Add any additional handlers
        for handler in handlers.values():
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger, typically the module name

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)
