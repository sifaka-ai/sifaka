"""
Logging utilities for Sifaka.

This module provides enhanced logging capabilities for the Sifaka library,
including structured logging, customizable formatting, and convenience methods
for common logging patterns.
"""

import logging
import json
from typing import Optional, Protocol, runtime_checkable, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from functools import wraps
import inspect
from contextlib import contextmanager
import time


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


class StructuredFormatter(logging.Formatter):
    """A formatter that supports structured logging with optional color output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def __init__(
        self,
        fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        use_colors: bool = True,
        structured: bool = False,
    ):
        """Initialize the formatter with optional color and structured output."""
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        self.structured = structured

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with optional color and structure."""
        # Handle structured logging data
        if hasattr(record, "structured_data"):
            structured_data = record.structured_data
            if isinstance(structured_data, dict):
                record.message = json.dumps(structured_data, indent=2)

        # Apply base formatting
        formatted = super().format(record)

        # Apply colors if enabled
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            formatted = f"{color}{formatted}{self.RESET}"

        return formatted


@dataclass(frozen=True)
class LogConfig:
    """Immutable configuration for loggers."""

    name: str
    level: int = logging.INFO
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = True
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    use_colors: bool = True
    structured: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.level, int):
            raise ValueError("level must be an integer")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")


class EnhancedLogger(logging.Logger):
    """Enhanced logger with additional convenience methods."""

    def structured(self, level: int, message: str, **kwargs: Any) -> None:
        """Log a structured message with additional data."""
        if self.isEnabledFor(level):
            data = {"message": message, **kwargs}
            record = self.makeRecord(self.name, level, "(structured)", 0, message, (), None)
            record.structured_data = data
            self.handle(record)

    def success(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a success message at INFO level with green color."""
        self.info(f"✓ {msg}", *args, **kwargs)

    def start_operation(self, operation: str) -> None:
        """Log the start of an operation."""
        self.info(f"Starting: {operation}...")

    def end_operation(self, operation: str, success: bool = True) -> None:
        """Log the end of an operation."""
        status = "completed successfully" if success else "failed"
        self.info(f"Operation '{operation}' {status}")

    @contextmanager
    def operation_context(self, operation: str) -> None:
        """Context manager for logging operation start/end with timing."""
        start_time = time.time()
        try:
            self.start_operation(operation)
            yield
        except Exception as e:
            self.error(f"Operation '{operation}' failed: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            self.info(f"Operation '{operation}' completed in {duration:.2f}s")


class LoggerFactory:
    """Factory for creating and configuring loggers."""

    def __init__(self, config: Optional[LogConfig] = None) -> None:
        """Initialize the logger factory."""
        self.config = config or LogConfig(name="sifaka")
        self._loggers: Dict[str, logging.Logger] = {}

        # Register our enhanced logger class
        logging.setLoggerClass(EnhancedLogger)

    def create_formatter(self) -> LogFormatter:
        """Create a structured formatter."""
        return StructuredFormatter(
            fmt=self.config.format_string,
            datefmt=self.config.date_format,
            use_colors=self.config.use_colors,
            structured=self.config.structured,
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

        timestamp = datetime.now().strftime("%Y%m%d")
        handler = logging.FileHandler(log_dir / f"{logger_name}_{timestamp}.log")
        # Use non-colored formatter for files
        handler.setFormatter(
            StructuredFormatter(
                fmt=self.config.format_string,
                datefmt=self.config.date_format,
                use_colors=False,
                structured=self.config.structured,
            )
        )
        return handler

    def get_logger(self, name: str, level: Optional[int] = None) -> EnhancedLogger:
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


def get_logger(name: str, level: Optional[int] = None) -> EnhancedLogger:
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


def log_operation(logger: Optional[EnhancedLogger] = None):
    """Decorator to log function entry/exit with timing."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            func_name = func.__name__
            logger.start_operation(func_name)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.success(f"{func_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed after {time.time() - start_time:.2f}s: {str(e)}")
                raise

        return wrapper

    return decorator
