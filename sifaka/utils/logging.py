"""
Logging utilities for Sifaka.
"""

import logging
from typing import Optional, Protocol, runtime_checkable, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@runtime_checkable
class LogFormatter(Protocol):
    """Protocol for log formatters."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record."""
        ...


@runtime_checkable
class LogHandler(Protocol):
    """Protocol for log handlers."""

    def setFormatter(self, formatter: LogFormatter) -> None:
        """Set the formatter for this handler."""
        ...

    def handle(self, record: logging.LogRecord) -> bool:
        """Handle a log record."""
        ...


@dataclass(frozen=True)
class LogConfig:
    """Immutable configuration for loggers."""

    name: str
    level: int = logging.INFO
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = True
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.level, int):
            raise ValueError("level must be an integer")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")


class LoggerFactory:
    """Factory for creating and configuring loggers."""

    def __init__(self, config: Optional[LogConfig] = None) -> None:
        """Initialize the logger factory."""
        self.config = config or LogConfig(name="sifaka")
        self._loggers: Dict[str, logging.Logger] = {}

    def create_formatter(self) -> LogFormatter:
        """Create a log formatter."""
        return logging.Formatter(
            fmt=self.config.format_string,
            datefmt=self.config.date_format,
        )

    def create_console_handler(self) -> LogHandler:
        """Create a console handler."""
        handler = logging.StreamHandler()
        handler.setFormatter(self.create_formatter())
        return handler

    def create_file_handler(self, logger_name: str) -> Optional[LogHandler]:
        """Create a file handler if configured."""
        if not self.config.log_to_file:
            return None

        log_dir = self.config.log_dir
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        handler = logging.FileHandler(log_dir / f"{logger_name}.log")
        handler.setFormatter(self.create_formatter())
        return handler

    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """Get or create a logger with the given name and level."""
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level or self.config.level)

        if not logger.handlers:
            # Add console handler
            logger.addHandler(self.create_console_handler())

            # Add file handler if configured
            file_handler = self.create_file_handler(name)
            if file_handler:
                logger.addHandler(file_handler)

        self._loggers[name] = logger
        return logger


# Global logger factory instance
_logger_factory = LoggerFactory()


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with the given name and level."""
    return _logger_factory.get_logger(name, level)


def set_log_level(level: int) -> None:
    """Set the logging level for all Sifaka loggers."""
    if not isinstance(level, int):
        raise TypeError("level must be an integer")
    logging.getLogger("sifaka").setLevel(level)


def disable_logging() -> None:
    """Disable all Sifaka logging."""
    logging.getLogger("sifaka").setLevel(logging.CRITICAL)


def configure_logging(config: LogConfig) -> None:
    """Configure logging with the given configuration."""
    global _logger_factory
    _logger_factory = LoggerFactory(config)
