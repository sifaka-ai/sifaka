"""
Logging utilities for Sifaka.

This module provides functions for setting up and using logging in Sifaka.
"""

import logging
from typing import Optional, Union


def configure_logging(
    level: Union[int, str] = logging.INFO, log_file: Optional[str] = None
) -> None:
    """Configure logging for Sifaka.

    Args:
        level: The logging level to use. Can be an integer or a string like "INFO", "DEBUG", etc.
        log_file: Optional path to a log file.
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(level.upper(), logging.INFO)
    logger = logging.getLogger("sifaka")
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: The name of the module.

    Returns:
        A logger instance.
    """
    return logging.getLogger(f"sifaka.{name}")
