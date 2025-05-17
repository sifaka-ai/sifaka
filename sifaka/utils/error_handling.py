"""
Error handling utilities for Sifaka.

This module provides utility functions and context managers for consistent
error handling across the Sifaka framework.
"""

import logging
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast
from contextlib import contextmanager

from sifaka.errors import (
    SifakaError,
    ValidationError,
    ImproverError,
    ModelError,
    ChainError,
    RetrieverError,
)

# Configure logger
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


def format_error_message(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
) -> str:
    """
    Format an error message with component, operation, and suggestions.
    
    Args:
        message: The base error message
        component: Optional component name
        operation: Optional operation name
        suggestions: Optional list of suggestions
        
    Returns:
        A formatted error message
    """
    formatted_message = message
    
    if component:
        formatted_message = f"[{component}] {formatted_message}"
    
    if operation:
        formatted_message = f"{formatted_message} (during {operation})"
    
    if suggestions:
        suggestion_text = "; ".join(suggestions)
        formatted_message = f"{formatted_message}. Suggestions: {suggestion_text}"
    
    return formatted_message


def log_error(
    error: Exception,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    include_traceback: bool = True,
    component: Optional[str] = None,
    operation: Optional[str] = None,
) -> None:
    """
    Log an error with consistent formatting.
    
    Args:
        error: The exception to log
        logger_instance: Optional logger instance to use
        level: Logging level
        include_traceback: Whether to include the traceback
        component: Optional component name
        operation: Optional operation name
    """
    logger_to_use = logger_instance or logger
    
    # Format the error message
    error_message = str(error)
    if component or operation:
        error_message = format_error_message(error_message, component, operation)
    
    # Log the error
    if include_traceback:
        logger_to_use.log(level, error_message, exc_info=True)
    else:
        logger_to_use.log(level, error_message)


def convert_exception(
    exception: Exception,
    target_exception_class: Type[Exception],
    message_prefix: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Exception:
    """
    Convert an exception to a different type with additional context.
    
    Args:
        exception: The original exception
        target_exception_class: The target exception class
        message_prefix: Optional prefix for the error message
        component: Optional component name
        operation: Optional operation name
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        
    Returns:
        A new exception of the target type
    """
    # Get the original error message
    original_message = str(exception)
    
    # Create the new message
    new_message = original_message
    if message_prefix:
        new_message = f"{message_prefix}: {new_message}"
    
    # Create the new exception
    if issubclass(target_exception_class, SifakaError):
        # If the target is a SifakaError, use its constructor with all parameters
        new_exception = target_exception_class(
            message=new_message,
            component=component,
            operation=operation,
            suggestions=suggestions,
            metadata=metadata,
        )
    else:
        # Otherwise, use the standard constructor
        new_exception = target_exception_class(new_message)
    
    # Preserve the traceback
    if hasattr(exception, "__traceback__"):
        new_exception.__traceback__ = exception.__traceback__
    
    return new_exception


@contextmanager
def error_context(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    error_class: Type[Exception] = SifakaError,
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    logger_instance: Optional[logging.Logger] = None,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
):
    """
    Context manager for consistent error handling.
    
    Args:
        component: Optional component name
        operation: Optional operation name
        error_class: Exception class to raise
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        logger_instance: Optional logger instance to use
        log_level: Logging level
        include_traceback: Whether to include the traceback
        
    Yields:
        None
        
    Raises:
        The specified error_class with additional context
    """
    try:
        yield
    except Exception as e:
        # Log the error
        log_error(
            e,
            logger_instance=logger_instance,
            level=log_level,
            include_traceback=include_traceback,
            component=component,
            operation=operation,
        )
        
        # Convert the exception
        new_exception = convert_exception(
            e,
            error_class,
            message_prefix=message_prefix,
            component=component,
            operation=operation,
            suggestions=suggestions,
            metadata=metadata,
        )
        
        # Raise the new exception
        raise new_exception from e


def with_error_handling(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    error_class: Type[Exception] = SifakaError,
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    logger_instance: Optional[logging.Logger] = None,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator for consistent error handling.
    
    Args:
        component: Optional component name
        operation: Optional operation name
        error_class: Exception class to raise
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        logger_instance: Optional logger instance to use
        log_level: Logging level
        include_traceback: Whether to include the traceback
        
    Returns:
        A decorator function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            with error_context(
                component=component,
                operation=operation or func.__name__,
                error_class=error_class,
                message_prefix=message_prefix,
                suggestions=suggestions,
                metadata=metadata,
                logger_instance=logger_instance,
                log_level=log_level,
                include_traceback=include_traceback,
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Specialized context managers for common operations

@contextmanager
def validation_context(
    validator_name: Optional[str] = None,
    operation: Optional[str] = "validation",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for validation operations.
    
    Args:
        validator_name: Optional validator name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        
    Yields:
        None
        
    Raises:
        ValidationError with additional context
    """
    with error_context(
        component="Validator",
        operation=operation,
        error_class=ValidationError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add validator name to the context if provided
        if validator_name:
            context.metadata = context.metadata or {}
            context.metadata["validator_name"] = validator_name
        yield


@contextmanager
def improvement_context(
    improver_name: Optional[str] = None,
    operation: Optional[str] = "improvement",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for improvement operations.
    
    Args:
        improver_name: Optional improver name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        
    Yields:
        None
        
    Raises:
        ImproverError with additional context
    """
    with error_context(
        component="Improver",
        operation=operation,
        error_class=ImproverError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add improver name to the context if provided
        if improver_name:
            context.metadata = context.metadata or {}
            context.metadata["improver_name"] = improver_name
        yield


@contextmanager
def model_context(
    model_name: Optional[str] = None,
    operation: Optional[str] = "generation",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for model operations.
    
    Args:
        model_name: Optional model name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        
    Yields:
        None
        
    Raises:
        ModelError with additional context
    """
    with error_context(
        component="Model",
        operation=operation,
        error_class=ModelError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add model name to the context if provided
        if model_name:
            context.metadata = context.metadata or {}
            context.metadata["model_name"] = model_name
        yield


@contextmanager
def retrieval_context(
    retriever_name: Optional[str] = None,
    operation: Optional[str] = "retrieval",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for retrieval operations.
    
    Args:
        retriever_name: Optional retriever name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        
    Yields:
        None
        
    Raises:
        RetrieverError with additional context
    """
    with error_context(
        component="Retriever",
        operation=operation,
        error_class=RetrieverError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add retriever name to the context if provided
        if retriever_name:
            context.metadata = context.metadata or {}
            context.metadata["retriever_name"] = retriever_name
        yield
