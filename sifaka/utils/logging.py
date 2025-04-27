"""
Logging utilities for Sifaka.
"""

import logging
from typing import Optional
from pathlib import Path


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the given name and level.

    Args:
        name: The name of the logger
        level: The logging level to use (default: logging.INFO)

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(level or logging.INFO)

        # Create a formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Add a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Add a file handler if logs directory exists
        logs_dir = Path("logs")
        if logs_dir.exists():
            file_handler = logging.FileHandler(logs_dir / f"{name}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def set_log_level(level: int) -> None:
    """
    Set the logging level for all Sifaka loggers.

    Args:
        level: The logging level to set
    """
    logging.getLogger("sifaka").setLevel(level)


def disable_logging() -> None:
    """Disable all Sifaka logging."""
    logging.getLogger("sifaka").setLevel(logging.CRITICAL)
