"""
Error handling utilities for Sifaka.

This module provides a standardized exception hierarchy and error handling utilities
for the Sifaka framework. It defines base exception classes, component-specific
exceptions, and utilities for consistent error handling across the framework.

The error hierarchy is designed to provide clear categorization of errors and
consistent error handling patterns. The utilities help ensure that errors are
properly logged, formatted, and propagated throughout the framework.

Examples:
    ```python
    from sifaka.utils.errors import (
        SifakaError, ValidationError, handle_errors,
        with_error_handling, format_error_metadata
    )

    # Using the exception hierarchy
    try:
        # Some operation
        pass
    except ValidationError as e:
        # Handle validation error
        pass
    except SifakaError as e:
        # Handle any Sifaka error
        pass

    # Using the error handling decorator
    @handle_errors(fallback_value=None, log_errors=True)
    def my_function():
        # Function implementation
        pass

    # Using the context manager
    with with_error_handling("operation name", logger=logger):
        # Operation implementation
        pass

    # Formatting error metadata
    metadata = format_error_metadata(exception)
    ```
"""

import functools
import inspect
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel

from sifaka.utils.logging import get_logger

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")

# Default logger for error handling
_logger = get_logger(__name__)


# Base Exception Classes
class SifakaError(Exception):
    """
    Base exception for all Sifaka errors.

    This is the root exception class for all errors in the Sifaka framework.
    All other exception classes should inherit from this class.

    Attributes:
        message: The error message
        cause: The original exception that caused this error (if any)
    """

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        """
        Initialize a SifakaError.

        Args:
            message: The error message
            cause: The original exception that caused this error (if any)
        """
        self.message = message
        self.cause = cause
        super().__init__(message)


# Category-specific Exceptions
class ValidationError(SifakaError):
    """
    Exception raised for validation errors.

    This exception should be raised when validation cannot be performed
    due to issues with the input or validation process.

    Examples:
        ```python
        from sifaka.utils.errors import ValidationError

        def validate_number(value: str) -> bool:
            try:
                float(value)
                return True
            except ValueError:
                raise ValidationError(f"'{value}' is not a valid number")
        ```
    """

    pass


class ConfigurationError(SifakaError):
    """
    Exception raised for configuration errors.

    This exception should be raised when a component is incorrectly configured.
    It helps identify configuration issues early, before operations are attempted.

    Examples:
        ```python
        from sifaka.utils.errors import ConfigurationError

        def configure_component(config: Dict[str, Any]) -> None:
            if "required_param" not in config:
                raise ConfigurationError("Missing required parameter: required_param")
        ```
    """

    pass


class RuntimeError(SifakaError):
    """
    Exception raised for runtime errors.

    This exception should be raised when an error occurs during the execution
    of a component or operation. It indicates that the operation could not be
    completed due to an unexpected condition.

    Examples:
        ```python
        from sifaka.utils.errors import RuntimeError

        def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Process data
                return processed_data
            except Exception as e:
                raise RuntimeError(f"Failed to process data: {e}", cause=e)
        ```
    """

    pass


class TimeoutError(SifakaError):
    """
    Exception raised for timeout errors.

    This exception should be raised when an operation times out.
    It helps identify performance issues and prevent hanging operations.

    Examples:
        ```python
        from sifaka.utils.errors import TimeoutError

        def operation_with_timeout(timeout: float = 5.0) -> None:
            start_time = time.time()
            while not is_complete():
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")
                time.sleep(0.1)
        ```
    """

    pass


# Component-specific Exceptions
class ModelError(RuntimeError):
    """
    Exception raised for model-related errors.

    This exception should be raised when an error occurs during model operations,
    such as generation, inference, or API calls.

    Examples:
        ```python
        from sifaka.utils.errors import ModelError

        def generate_text(prompt: str) -> str:
            try:
                # Call model API
                return response_text
            except Exception as e:
                raise ModelError(f"Failed to generate text: {e}", cause=e)
        ```
    """

    pass


class ClassifierError(RuntimeError):
    """
    Exception raised for classifier-related errors.

    This exception should be raised when an error occurs during classification
    operations, such as model loading, inference, or processing.

    Examples:
        ```python
        from sifaka.utils.errors import ClassifierError

        def classify_text(text: str) -> str:
            try:
                # Classify text
                return label
            except Exception as e:
                raise ClassifierError(f"Failed to classify text: {e}", cause=e)
        ```
    """

    pass


class CriticError(RuntimeError):
    """
    Exception raised for critic-related errors.

    This exception should be raised when an error occurs during critique
    operations, such as model loading, inference, or processing.

    Examples:
        ```python
        from sifaka.utils.errors import CriticError

        def critique_text(text: str) -> Dict[str, Any]:
            try:
                # Critique text
                return critique_result
            except Exception as e:
                raise CriticError(f"Failed to critique text: {e}", cause=e)
        ```
    """

    pass


class ChainError(RuntimeError):
    """
    Exception raised for chain-related errors.

    This exception should be raised when an error occurs during chain
    operations, such as execution, validation, or processing.

    Examples:
        ```python
        from sifaka.utils.errors import ChainError

        def run_chain(input_text: str) -> str:
            try:
                # Run chain
                return output_text
            except Exception as e:
                raise ChainError(f"Failed to run chain: {e}", cause=e)
        ```
    """

    pass


# Error Handling Utilities
def format_error_metadata(exception: Exception) -> Dict[str, Any]:
    """
    Format exception information as metadata.

    This utility function formats exception information as a dictionary
    that can be included in result metadata.

    Args:
        exception: The exception to format

    Returns:
        Dictionary containing error information

    Examples:
        ```python
        try:
            # Some operation
            pass
        except Exception as e:
            metadata = format_error_metadata(e)
            return Result(success=False, metadata=metadata)
        ```
    """
    error_type = type(exception).__name__
    error_message = str(exception)
    
    metadata = {
        "error": error_message,
        "error_type": error_type,
        "reason": f"{error_type.lower()}_error",
    }
    
    # Add traceback for debugging (truncated to avoid excessive size)
    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    metadata["traceback"] = "".join(tb_lines[-10:])  # Last 10 lines
    
    # Add cause information if available
    if hasattr(exception, "cause") and exception.cause is not None:
        metadata["cause"] = str(exception.cause)
        metadata["cause_type"] = type(exception.cause).__name__
    
    return metadata


def handle_errors(
    fallback_value: Optional[R] = None,
    log_errors: bool = True,
    reraise: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator for standardized error handling.

    This decorator wraps a function with standardized error handling.
    It catches exceptions, logs them, and either returns a fallback value
    or re-raises the exception.

    Args:
        fallback_value: Value to return if an exception occurs
        log_errors: Whether to log errors
        reraise: Whether to re-raise exceptions
        logger: Logger to use (defaults to module logger)

    Returns:
        Decorated function

    Examples:
        ```python
        from sifaka.utils.errors import handle_errors

        @handle_errors(fallback_value=None, log_errors=True)
        def my_function():
            # Function implementation
            pass
        ```
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            nonlocal logger
            if logger is None:
                logger = _logger
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                logger.error(f"Error in {func_name}: {e}")
                
                if reraise:
                    raise
                
                if fallback_value is not None:
                    return fallback_value
                
                # If no fallback value is provided and not reraising, raise a RuntimeError
                raise RuntimeError(f"Error in {func_name}: {e}", cause=e)
        
        return wrapper
    
    return decorator


@contextmanager
def with_error_handling(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
) -> None:
    """
    Context manager for standardized error handling.

    This context manager provides standardized error handling for operations.
    It logs the start and end of the operation, catches exceptions, and
    optionally re-raises them.

    Args:
        operation_name: Name of the operation
        logger: Logger to use (defaults to module logger)
        reraise: Whether to re-raise exceptions

    Examples:
        ```python
        from sifaka.utils.errors import with_error_handling

        with with_error_handling("data processing", logger=logger):
            # Operation implementation
            pass
        ```
    """
    if logger is None:
        logger = _logger
    
    start_time = time.time()
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Operation '{operation_name}' completed successfully in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Operation '{operation_name}' failed after {duration:.2f}s: {e}")
        
        if reraise:
            raise
