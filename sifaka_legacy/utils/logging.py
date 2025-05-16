from typing import Any

"""
Logging Module

A module providing enhanced logging capabilities for the Sifaka library.

## Overview
This module provides enhanced logging capabilities for the Sifaka library,
including structured logging, customizable formatting, and convenience methods
for common logging patterns. It extends Python's standard logging module with
additional functionality specific to Sifaka's needs.

## Components
- LogFormatter: Protocol for log formatters
- LogHandler: Protocol for log handlers
- StructuredFormatter: Formatter supporting structured logging with color output
- LogConfig: Configuration for loggers
- EnhancedLogger: Logger with additional convenience methods
- LoggerFactory: Factory for creating and configuring loggers
- Utility functions for common logging operations

## Usage Examples
```python
from sifaka.utils.logging import get_logger, configure_logging

# Get a logger with default configuration
logger = get_logger("my_component")
logger.info("This is an info message") if logger else ""
logger.error("This is an error message") if logger else ""

# Use structured logging
logger.structured(
    logger.INFO,
    "Processing data",
    data_size=1024,
    processing_time=0.5,
    status="success"
) if logger else ""

# Use operation context
with logger.operation_context("data_processing") if logger else "":
    # Do some work
    process_data()
    # The operation start/end and timing will be logged automatically

# Configure logging with custom settings
from sifaka.utils.logging import LogConfig
from pathlib import Path

config = LogConfig(
    name="my_app",
    level=logging.DEBUG,
    log_to_file=True,
    log_dir=Path("./logs"),
    use_colors=True
)
configure_logging(config)
```

## Error Handling
The module handles various error conditions:
- Invalid configuration values
- Logger initialization failures
- Structured data formatting errors
- Operation context exceptions

## Configuration
Logging can be configured with:
- Log levels
- Output formats
- File logging options
- Color output options
- Structured logging options
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
    IO,
    TextIO,
)
from logging import Handler, Formatter, Logger, StreamHandler, FileHandler


@runtime_checkable
class LogFormatter(Protocol):
    """
    Protocol for log formatters.

    This protocol defines the interface that all log formatters must implement,
    providing a format method for formatting log records.

    ## Architecture
    Uses Python's Protocol type to define a structural interface
    that log formatters must satisfy.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record.

        Args:
            record (logging.LogRecord): The log record to format

        Returns:
            str: The formatted log record
        """
        ...


@runtime_checkable
class LogHandler(Protocol):
    """
    Protocol for log handlers.

    This protocol defines the interface that all log handlers must implement,
    providing methods for setting formatters and handling log records.

    ## Architecture
    Uses Python's Protocol type to define a structural interface
    that log handlers must satisfy.
    """

    def setFormatter(self, formatter: Optional[Formatter]) -> None:
        """
        Set the formatter for this handler.

        Args:
            formatter (Optional[Formatter]): The formatter to use
        """
        ...

    def handle(self, record: logging.LogRecord) -> bool:
        """
        Handle a log record.

        Args:
            record (logging.LogRecord): The log record to handle

        Returns:
            bool: True if the record was handled successfully, False otherwise
        """
        ...


class StructuredFormatter(logging.Formatter):
    """
    A formatter that supports structured logging with optional color output.

    This formatter extends the standard logging.Formatter to support structured
    logging with JSON formatting and optional color output for different log levels.

    ## Architecture
    Extends Python's Formatter class with additional functionality for:
    - Structured data formatting with JSON
    - Color-coded output for different log levels
    - Customizable date and message formatting

    ## Examples
    ```python
    # Create a formatter with color output
    formatter = StructuredFormatter(use_colors=True)

    # Create a formatter with structured output
    formatter = StructuredFormatter(structured=True)

    # Create a formatter with custom format
    formatter = StructuredFormatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ```

    Attributes:
        COLORS (Dict[str, str]): Color codes for different log levels
        RESET (str): ANSI code to reset colors
    """

    COLORS = {
        "DEBUG": "\x1b[36m",
        "INFO": "\x1b[32m",
        "WARNING": "\x1b[33m",
        "ERROR": "\x1b[31m",
        "CRITICAL": "\x1b[41m",
    }
    RESET = "\x1b[0m"

    def __init__(
        self,
        fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        use_colors: bool = True,
        structured: bool = False,
    ) -> None:
        """Initialize the formatter with optional color and structured output."""
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        self.structured = structured

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with optional color and structure.

        This method formats a log record, applying structured data formatting
        if available and color coding if enabled.

        Args:
            record (logging.LogRecord): The log record to format

        Returns:
            str: The formatted log record

        Examples:
            When used with structured logging:
            ```python
            logger.structured(
                logger.INFO,
                "Processing data",
                data_size=1024,
                status="success"
            ) if logger else ""
            # Output: {"message": "Processing data", "data_size": 1024, "status": "success"}
            ```
        """
        if hasattr(record, "structured_data"):
            structured_data = record.structured_data
            if isinstance(structured_data, dict):
                record.message = json.dumps(structured_data, indent=2)
        formatted = super().format(record)
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            formatted = f"{color}{formatted}{self.RESET}"
        return formatted


@dataclass(frozen=True)
class LogConfig:
    """
    Immutable configuration for loggers.

    This class provides a standardized way to configure loggers in the Sifaka
    framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Uses Python's dataclass with frozen=True to create an immutable configuration
    object with validation in __post_init__.

    ## Examples
    ```python
    from sifaka.utils.logging import LogConfig
    from pathlib import Path
    import logging

    # Create a basic configuration
    config = LogConfig(name="my_app")

    # Create a detailed configuration
    config = LogConfig(
        name="my_app",
        level=logging.DEBUG,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        log_to_file=True,
        log_dir=Path("./logs"),
        use_colors=True,
        structured=False
    )
    ```

    Attributes:
        name (str): Logger name
        level (int): Logging level (from logging module)
        format_string (str): Format string for log messages
        date_format (str): Format string for dates in log messages
        log_to_file (bool): Whether to log to a file
        log_dir (Path): Directory for log files
        use_colors (bool): Whether to use colors in console output
        structured (bool): Whether to enable structured logging
    """

    name: str
    level: int = logging.INFO
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = True
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    use_colors: bool = True
    structured: bool = False

    def __post_init__(self) -> None:
        """
        Validate configuration values.

        This method is called after initialization to validate the configuration
        values and ensure they are valid.

        Raises:
            ValueError: If level is not an integer
            ValueError: If name is not a non-empty string
        """
        if not isinstance(self.level, int):
            raise ValueError("level must be an integer")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")


class EnhancedLogger(logging.Logger):
    """
    Enhanced logger with additional convenience methods.

    This class extends the standard logging.Logger with additional convenience
    methods for structured logging, operation tracking, and success messages.

    ## Architecture
    Extends Python's Logger class with additional methods for:
    - Structured logging with metadata
    - Operation tracking with start/end messages
    - Success messages with visual indicators
    - Context managers for operation timing

    ## Examples
    ```python
    from sifaka.utils.logging import get_logger

    # Get an enhanced logger
    logger = get_logger("my_component")

    # Use structured logging
    logger.structured(
        logger.INFO,
        "Processing data",
        data_size=1024,
        status="success"
    ) if logger else ""

    # Log operation start/end
    logger.start_operation("data_processing") if logger else ""
    # Do some work
    logger.end_operation("data_processing") if logger else ""

    # Log success message
    logger.success("Data processing completed") if logger else ""

    # Use operation context
    with logger.operation_context("data_processing") if logger else "":
        # Do some work
        process_data()
    ```
    """

    def structured(self, level: int, message: str, **kwargs: Any) -> None:
        """
        Log a structured message with additional data.

        This method logs a structured message with additional data as key-value pairs,
        which can be formatted as JSON by the StructuredFormatter.

        Args:
            level (int): Log level (e.g., logging.INFO, logging.ERROR)
            message (str): Main log message
            **kwargs: Additional key-value pairs to include in the structured data

        Examples:
            ```python
            logger.structured(
                logging.INFO,
                "Processing data",
                data_size=1024,
                processing_time=0.5,
                status="success"
            )
            ```
        """
        if self.isEnabledFor(level):
            data = {"message": message, **kwargs}
            record = self.makeRecord(self.name, level, "(structured)", 0, message, (), None)
            record.structured_data = data
            self.handle(record)

    def success(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a success message at INFO level with green color.

        This method logs a success message at INFO level, prefixed with a checkmark
        symbol for visual indication of success.

        Args:
            msg (str): Success message
            *args: Additional positional arguments for message formatting
            **kwargs: Additional keyword arguments for the logger

        Examples:
            ```python
            logger.success("Data processing completed")
            logger.success("Processed %d items", 100)
            ```
        """
        self.info(f"✓ {msg}", *args, **kwargs)

    def start_operation(self, operation: str) -> None:
        """
        Log the start of an operation.

        This method logs the start of an operation at INFO level.

        Args:
            operation (str): Name of the operation

        Examples:
            ```python
            logger.start_operation("data_processing")
            ```
        """
        self.info(f"Starting: {operation}...")

    def end_operation(self, operation: str, success: bool = True) -> None:
        """
        Log the end of an operation.

        This method logs the end of an operation at INFO level, indicating
        whether it completed successfully or failed.

        Args:
            operation (str): Name of the operation
            success (bool, optional): Whether the operation succeeded. Defaults to True.

        Examples:
            ```python
            logger.end_operation("data_processing")
            logger.end_operation("data_processing", success=False)
            ```
        """
        status = "completed successfully" if success else "failed"
        self.info(f"Operation '{operation}' {status}")

    @contextmanager
    def operation_context(self, operation: str) -> Any:
        """
        Context manager for logging operation start/end with timing.

        This context manager logs the start of an operation, executes the code
        in the context, and logs the end of the operation with timing information.
        If an exception occurs, it logs the failure and re-raises the exception.

        Args:
            operation (str): Name of the operation

        Yields:
            None

        Raises:
            Exception: Any exception raised in the context

        Examples:
            ```python
            with logger.operation_context("data_processing"):
                # Do some work
                process_data()
            ```
        """
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
    """
    Factory for creating and configuring loggers.

    This class provides a factory for creating and configuring loggers with
    consistent settings, including console and file handlers with appropriate
    formatters.

    ## Architecture
    Implements the factory pattern for creating and configuring loggers with:
    - Consistent configuration through LogConfig
    - Console and file handlers
    - Structured formatters
    - Caching of loggers for reuse

    ## Examples
    ```python
    from sifaka.utils.logging import LoggerFactory, LogConfig

    # Create a factory with default configuration
    factory = LoggerFactory()

    # Create a factory with custom configuration
    config = LogConfig(name="my_app", level=logging.DEBUG)
    factory = LoggerFactory(config)

    # Get a logger
    logger = factory.get_logger("my_component") if factory else ""

    # Get a logger with custom level
    logger = factory.get_logger("my_component", level=logging.DEBUG) if factory else ""
    ```

    Attributes:
        config (LogConfig): Logger configuration
        _loggers (Dict[str, logging.Logger]): Cache of created loggers
    """

    def __init__(self, config: Optional[LogConfig] = None) -> None:
        """
        Initialize the logger factory.

        This method initializes the logger factory with the given configuration,
        or a default configuration if none is provided.

        Args:
            config (Optional[LogConfig], optional): Logger configuration. Defaults to None.
        """
        self.config = config or LogConfig(name="sifaka")
        self._loggers: Dict[str, logging.Logger] = {}
        logging.setLoggerClass(EnhancedLogger)

    def create_formatter(self) -> Formatter:
        """
        Create a structured formatter.

        This method creates a structured formatter with the configuration
        settings from the factory's config.

        Returns:
            Formatter: A configured formatter
        """
        return StructuredFormatter(
            fmt=self.config.format_string,
            datefmt=self.config.date_format,
            use_colors=self.config.use_colors,
            structured=self.config.structured,
        )

    def create_console_handler(self) -> Handler:
        """
        Create a console handler.

        This method creates a console handler with the appropriate formatter.

        Returns:
            Handler: A configured console handler
        """
        handler = StreamHandler()
        handler.setFormatter(self.create_formatter())
        return handler

    def create_file_handler(self, logger_name: str) -> Optional[Handler]:
        """
        Create a file handler if configured.

        This method creates a file handler for the given logger name if file
        logging is enabled in the configuration.

        Args:
            logger_name (str): Name of the logger

        Returns:
            Optional[Handler]: A configured file handler, or None if file logging is disabled

        Raises:
            OSError: If the log directory cannot be created
        """
        if not self.config.log_to_file:
            return None
        log_dir = self.config.log_dir
        if log_dir and not log_dir.exists():
            log_dir.mkdir(parents=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        handler = FileHandler(log_dir / f"{logger_name}_{timestamp}.log")
        handler.setFormatter(
            StructuredFormatter(
                fmt=self.config.format_string,
                datefmt=self.config.date_format,
                use_colors=False,
                structured=self.config.structured,
            )
        )
        return handler

    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """
        Get or create a logger with the given name and level.

        This method gets an existing logger from the cache or creates a new one
        with the given name and level.

        Args:
            name (str): Logger name
            level (Optional[int], optional): Log level. Defaults to None.

        Returns:
            EnhancedLogger: A configured logger

        Examples:
            ```python
            # Get a logger with default level
            logger = factory.get_logger("my_component") if factory else ""

            # Get a logger with custom level
            logger = factory.get_logger("my_component", level=logging.DEBUG) if factory else ""
            ```
        """
        if name in self._loggers:
            return self._loggers[name]
        logger = logging.getLogger(name)
        logger.setLevel(level or self.config.level)
        if not logger.handlers:
            logger.addHandler(self.create_console_handler())
            file_handler = self.create_file_handler(name)
            if file_handler:
                logger.addHandler(file_handler)
        self._loggers[name] = logger
        return logger


_logger_factory = LoggerFactory()


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the given name and level.

    This function is a convenience wrapper around the global logger factory's
    get_logger method, providing a simple way to get a logger with the given
    name and level.

    Args:
        name (str): Logger name
        level (Optional[int], optional): Log level. Defaults to None.

    Returns:
        EnhancedLogger: A configured logger

    Examples:
        ```python
        from sifaka.utils.logging import get_logger
        import logging

        # Get a logger with default level
        logger = get_logger("my_component")

        # Get a logger with custom level
        logger = get_logger("my_component", level=logging.DEBUG)
        ```
    """
    return _logger_factory.get_logger(name, level)


def set_log_level(level: int) -> None:
    """
    Set the logging level for all Sifaka loggers.

    This function sets the logging level for the root Sifaka logger,
    which affects all loggers in the Sifaka namespace.

    Args:
        level (int): Log level (e.g., logging.DEBUG, logging.INFO)

    Raises:
        TypeError: If level is not an integer

    Examples:
        ```python
        from sifaka.utils.logging import set_log_level
        import logging

        # Set log level to DEBUG
        set_log_level(logging.DEBUG)

        # Set log level to INFO
        set_log_level(logging.INFO)
        ```
    """
    if not isinstance(level, int):
        raise TypeError("level must be an integer")
    logging.getLogger("sifaka").setLevel(level)


def disable_logging() -> None:
    """
    Disable all Sifaka logging.

    This function disables all logging in the Sifaka namespace by setting
    the log level to CRITICAL, which effectively silences all logs except
    for critical errors.

    Examples:
        ```python
        from sifaka.utils.logging import disable_logging

        # Disable all logging
        disable_logging()
        ```
    """
    logging.getLogger("sifaka").setLevel(logging.CRITICAL)


def configure_logging(config: Optional[LogConfig] = None, level: Optional[str] = None) -> None:
    """
    Configure logging with the given configuration or level.

    This function configures the global logger factory with the given
    configuration or level, affecting all loggers created after this call.

    Args:
        config (Optional[LogConfig], optional): A LogConfig object to use for configuration. Defaults to None.
        level (Optional[str], optional): A string log level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
              This is ignored if config is provided. Defaults to None.

    Raises:
        ValueError: If level is not a valid log level string

    Examples:
        ```python
        from sifaka.utils.logging import configure_logging, LogConfig
        from pathlib import Path

        # Configure with a string level
        configure_logging(level="DEBUG")

        # Configure with a LogConfig object
        config = LogConfig(
            name="my_app",
            level=logging.DEBUG,
            log_to_file=True,
            log_dir=Path("./logs")
        )
        configure_logging(config=config)
        ```
    """
    global _logger_factory
    if config:
        _logger_factory = LoggerFactory(config)
    elif level:
        numeric_level = getattr(logging, level.upper() if level else "", None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        current_config = _logger_factory.config
        new_config = LogConfig(
            name=current_config.name,
            level=numeric_level,
            format_string=current_config.format_string,
            date_format=current_config.date_format,
            log_to_file=current_config.log_to_file,
            log_dir=current_config.log_dir,
            use_colors=current_config.use_colors,
            structured=current_config.structured,
        )
        _logger_factory = LoggerFactory(new_config)


F = TypeVar("F", bound=Callable[..., Any])


def log_operation(logger: Optional[EnhancedLogger] = None) -> Callable[[F], Callable[..., Any]]:
    """
    Decorator to log function entry/exit with timing.

    This decorator logs the start and end of a function call, including timing
    information and success/failure status. If an exception occurs, it logs the
    error and re-raises the exception.

    Args:
        logger (Optional[EnhancedLogger], optional): Logger to use. If None, a logger
            will be created using the function's module name. Defaults to None.

    Returns:
        Callable: Decorator function

    Examples:
        ```python
        from sifaka.utils.logging import log_operation, get_logger

        # Use with default logger (from function's module)
        @log_operation()
        def process_data(data: Any) -> None:
            # Process data
            return result

        # Use with custom logger
        logger = get_logger("my_component")
        @log_operation(logger=logger)
        def process_data(data: Any) -> None:
            # Process data
            return result
        ```
    """

    def decorator(func: F) -> Callable[..., Any]:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = cast(Optional[EnhancedLogger], get_logger(func.__module__))
            func_name = func.__name__
            if logger:
                logger.start_operation(func_name)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if logger:
                    logger.success(f"{func_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                if logger:
                    logger.error(
                        f"{func_name} failed after {time.time() - start_time:.2f}s: {str(e)}"
                    )
                raise

        return wrapper

    return decorator
