"""Logging configuration and utilities for Sifaka.

This module provides comprehensive logging setup with support for:
- Structured logging with context
- Performance tracking
- Error correlation
- Development and production configurations
- Integration with PydanticAI observability

Example Usage:
    ```python
    from sifaka.utils.logging import get_logger, setup_logging

    # Setup logging (usually done once at startup)
    setup_logging(level="INFO", format="structured")

    # Get logger for your module
    logger = get_logger(__name__)

    # Log with context
    logger.info("Processing thought", extra={
        "thought_id": "123",
        "iteration": 2,
        "model": "openai:gpt-4"
    })

    # Log performance
    with logger.performance_timer("generation"):
        result = await generate_text(prompt)
    ```
"""

import logging
import logging.config
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from sifaka.utils.errors import ConfigurationError


class SifakaFormatter(logging.Formatter):
    """Custom formatter for Sifaka logs with structured output."""

    def __init__(self, format_type: str = "structured"):
        """Initialize formatter.

        Args:
            format_type: Either "structured" or "simple"
        """
        self.format_type = format_type

        if format_type == "structured":
            # Structured format for production
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | " "%(message)s | %(extra_fields)s"
        else:
            # Simple format for development
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with extra fields."""
        # Extract extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "asctime",
                "message",
            }:
                extra_fields[key] = value

        # Add extra_fields to record for structured format
        if self.format_type == "structured" and extra_fields:
            record.extra_fields = " | ".join(f"{k}={v}" for k, v in extra_fields.items())
        else:
            record.extra_fields = ""

        return super().format(record)


class PerformanceLogger:
    """Logger with performance tracking capabilities."""

    def __init__(self, logger: logging.Logger):
        """Initialize performance logger.

        Args:
            logger: Base logger instance
        """
        self.logger = logger

    # Delegate standard logging methods to the underlying logger
    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        return self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message."""
        return self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        return self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        return self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log a critical message."""
        return self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """Log an exception message."""
        return self.logger.exception(msg, *args, **kwargs)

    @contextmanager
    def performance_timer(self, operation: str, **context):
        """Context manager for timing operations.

        Args:
            operation: Name of the operation being timed
            **context: Additional context to include in logs
        """
        start_time = time.time()

        self.logger.debug(
            f"Starting {operation}", extra={"operation": operation, "stage": "start", **context}
        )

        try:
            yield
            duration = time.time() - start_time
            self.logger.info(
                f"Completed {operation}",
                extra={
                    "operation": operation,
                    "stage": "complete",
                    "duration_seconds": round(duration, 3),
                    **context,
                },
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Failed {operation}",
                extra={
                    "operation": operation,
                    "stage": "error",
                    "duration_seconds": round(duration, 3),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    **context,
                },
                exc_info=True,
            )
            raise

    def log_thought_event(
        self, event: str, thought_id: str, iteration: Optional[int] = None, **context
    ):
        """Log thought-related events with consistent structure.

        Args:
            event: Event name (e.g., "generation_start", "validation_complete")
            thought_id: Unique thought identifier
            iteration: Current iteration number
            **context: Additional context
        """
        extra = {"event": event, "thought_id": thought_id, **context}

        if iteration is not None:
            extra["iteration"] = iteration

        self.logger.info(f"Thought event: {event}", extra=extra)

    def log_model_call(
        self,
        model: str,
        operation: str,
        duration: Optional[float] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        **context,
    ):
        """Log model API calls with performance metrics.

        Args:
            model: Model identifier
            operation: Operation type (e.g., "generation", "critique")
            duration: Call duration in seconds
            tokens_used: Number of tokens consumed
            cost: API call cost
            **context: Additional context
        """
        extra = {"model": model, "operation": operation, **context}

        if duration is not None:
            extra["duration_seconds"] = round(duration, 3)
        if tokens_used is not None:
            extra["tokens_used"] = tokens_used
        if cost is not None:
            extra["cost_usd"] = cost

        self.logger.info(f"Model call: {model} {operation}", extra=extra)


def setup_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: Optional[Union[str, Path]] = None,
    enable_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """Setup logging configuration for Sifaka.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("structured" or "simple")
        log_file: Optional file path for log output
        enable_console: Whether to enable console output
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Raises:
        ConfigurationError: If logging setup fails
    """
    try:
        # Validate level
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ConfigurationError(
                f"Invalid log level: {level}",
                config_key="log_level",
                config_value=level,
                suggestions=[
                    "Use one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
                    "Check spelling and case",
                ],
            )

        # Create formatter
        formatter = SifakaFormatter(format_type)

        # Setup handlers
        handlers = []

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(numeric_level)
            handlers.append(console_handler)

        # File handler with rotation
        if log_file:
            from logging.handlers import RotatingFileHandler

            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_path, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            handlers=handlers,
            force=True,  # Override any existing configuration
        )

        # Set specific logger levels
        logging.getLogger("sifaka").setLevel(numeric_level)

        # Reduce noise from external libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        # Log successful setup
        logger = logging.getLogger("sifaka.logging")
        logger.info(
            "Logging configured",
            extra={
                "level": level,
                "format_type": format_type,
                "console_enabled": enable_console,
                "file_enabled": log_file is not None,
                "log_file": str(log_file) if log_file else None,
            },
        )

    except Exception as e:
        raise ConfigurationError(
            f"Failed to setup logging: {str(e)}",
            config_key="logging_setup",
            context={"error": str(e), "error_type": type(e).__name__},
            suggestions=[
                "Check log file path permissions",
                "Verify log level is valid",
                "Ensure log directory exists and is writable",
            ],
        ) from e


def get_logger(name: str) -> PerformanceLogger:
    """Get a logger instance with performance tracking.

    Args:
        name: Logger name (usually __name__)

    Returns:
        PerformanceLogger instance with enhanced capabilities

    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("Processing started")

        with logger.performance_timer("generation"):
            result = await generate_text(prompt)
        ```
    """
    base_logger = logging.getLogger(name)
    return PerformanceLogger(base_logger)


def configure_for_development() -> None:
    """Quick setup for development environment."""
    setup_logging(level="DEBUG", format_type="simple", enable_console=True, log_file=None)


def configure_for_production(log_dir: Union[str, Path] = "logs") -> None:
    """Quick setup for production environment.

    Args:
        log_dir: Directory for log files
    """
    log_path = Path(log_dir) / f"sifaka_{datetime.now().strftime('%Y%m%d')}.log"

    setup_logging(
        level="INFO",
        format_type="structured",
        enable_console=True,
        log_file=log_path,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10,
    )
