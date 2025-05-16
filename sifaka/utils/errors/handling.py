"""
Error Handling Functions

This module provides functions for standardized error handling in the Sifaka framework.
These functions handle errors in a consistent way, providing detailed error information
and appropriate logging.

## Functions
- **handle_error**: Process an error and return standardized error metadata
- **try_operation**: Execute an operation with standardized error handling
- **log_error**: Log an error with standardized formatting
- **handle_component_error**: Handle errors from generic components
- **create_error_handler**: Create a component-specific error handler
"""

import time
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast, Union, List

from ..logging import get_logger
from .base import SifakaError
from .component import (
    ChainError,
    ClassifierError,
    CriticError,
    ModelError,
    RetrievalError,
    RuleError,
)

# Import ErrorResult type for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import ErrorResult

# Configure logger
logger = get_logger(__name__)

# Type variable for return type
T = TypeVar("T")


def create_actionable_error_message(error: Exception, component_type: Optional[str] = None) -> str:
    """
    Create an actionable error message with specific recommendations.

    Args:
        error: The exception that was raised
        component_type: Optional type of component (model, validator, etc.)

    Returns:
        Actionable error message with suggestions
    """
    from sifaka.utils.errors.base import (
        ValidationError,
        ConfigurationError,
        ProcessingError,
        ResourceError,
        TimeoutError,
        InputError,
        StateError,
        DependencyError,
        InitializationError,
    )

    error_type = type(error).__name__
    error_message = str(error)
    component_context = f" in {component_type}" if component_type else ""

    # Base message
    message = f"Error{component_context}: {error_message}"
    suggestions = []

    # Add specific suggestions based on error type
    if isinstance(error, ValidationError):
        suggestions = [
            "Check your validation rules for conflicting criteria",
            "Ensure your content meets all required criteria",
            "Try using fewer or simpler validation rules",
        ]
    elif isinstance(error, ConfigurationError):
        suggestions = [
            "Verify your configuration parameters",
            "Check for typos in parameter names",
            "Ensure all required parameters are provided",
            "Refer to the documentation for correct parameter formats",
        ]
    elif isinstance(error, ProcessingError):
        suggestions = [
            "Check the input data for unexpected formats",
            "Try breaking down your request into smaller steps",
            "Verify that inputs match the expected types",
        ]
    elif isinstance(error, ResourceError):
        suggestions = [
            "Check your API keys and permissions",
            "Verify your network connection",
            "Ensure you have sufficient quota with your provider",
        ]
    elif isinstance(error, TimeoutError):
        suggestions = [
            "Try again with a simpler request",
            "Increase the timeout setting if available",
            "Break your request into smaller chunks",
        ]
    elif isinstance(error, InputError):
        suggestions = [
            "Verify your input data format",
            "Check for special characters that might cause issues",
            "Ensure your input meets size requirements",
        ]
    elif isinstance(error, StateError):
        suggestions = [
            "Ensure components are initialized before use",
            "Check the lifecycle of your components",
            "Verify the sequence of operations",
        ]
    elif isinstance(error, DependencyError):
        suggestions = [
            "Check that all required dependencies are registered",
            "Look for circular dependencies in your component setup",
            "Ensure dependency providers are properly initialized",
        ]
    elif isinstance(error, InitializationError):
        suggestions = [
            "Verify your setup code runs before using components",
            "Check that all required parameters are provided during initialization",
            "Ensure environment variables are set correctly",
        ]
    elif "OpenAI" in error_type or "openai" in error_message.lower():
        suggestions = [
            "Check your OpenAI API key",
            "Verify you have proper permissions for the model you're using",
            "Ensure you have sufficient quota in your OpenAI account",
        ]
    elif "Anthropic" in error_type or "anthropic" in error_message.lower():
        suggestions = [
            "Check your Anthropic API key",
            "Verify you have proper permissions for the model you're using",
            "Ensure you have sufficient quota in your Anthropic account",
        ]
    elif "import" in error_message.lower() or "module" in error_message.lower():
        suggestions = [
            "Ensure all required packages are installed",
            "Check for typos in import statements",
            "Verify your virtual environment is activated",
        ]
    else:
        # Generic suggestions for unknown errors
        suggestions = [
            "Check the input parameters and data format",
            "Refer to the documentation for correct usage",
            "Try breaking down complex operations into simpler steps",
        ]

    # Format suggestions
    if suggestions:
        message += "\n\nSuggestions to fix the issue:"
        for i, suggestion in enumerate(suggestions, 1):
            message += f"\n{i}. {suggestion}"

    # Add documentation reference when appropriate
    if not isinstance(error, (ResourceError, TimeoutError)):
        message += "\n\nFor more information, see our documentation at https://docs.sifaka.ai/troubleshooting"

    return message


def handle_error(
    error: Exception,
    component_type: Optional[str] = None,
    operation: Optional[str] = None,
    logger_instance: Any = None,
) -> None:
    """
    Handle an error with standardized logging and context.

    Args:
        error: The exception that was raised
        component_type: Optional type of component (model, validator, etc.)
        operation: Optional operation being performed
        logger_instance: Optional logger to use

    Raises:
        Same exception with enhanced message
    """
    # Use the default logger if none provided
    logger_instance = logger_instance or get_logger("error_handler")

    # Create operation context string
    op_context = f" during {operation}" if operation else ""

    # Get the original error type
    error_type = type(error)

    # Log the error
    logger_instance.error(f"Error{op_context}: {str(error)}", exc_info=True)

    # Create enhanced error message with actionable suggestions
    enhanced_message = create_actionable_error_message(error, component_type)

    # Raise a new exception of the same type with the enhanced message
    try:
        # Try to create a new exception of the same type with the enhanced message
        new_error = error_type(enhanced_message)
        # Copy the original exception's traceback
        new_error.__traceback__ = error.__traceback__
        raise new_error
    except TypeError:
        # If we can't create a new exception of the same type, re-raise the original
        raise error


def try_operation(
    operation: Callable[[], T],
    component_name: str,
    component_type: Optional[str] = None,
    error_class: Optional[Type[SifakaError]] = None,
    default_value: Optional[T] = None,
    log_level: str = "error",
    error_handler: Optional[Callable[[Exception], Optional[T]]] = None,
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Execute an operation with standardized error handling.

    Args:
        operation: The operation to execute
        component_name: Name of the component where the operation is being executed
        component_type: Optional type of component (model, validator, etc.)
        error_class: Optional SifakaError subclass to use for wrapping errors
        default_value: Optional default value to return if an error occurs
        log_level: Log level to use for errors
        error_handler: Optional custom error handler function
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Result of the operation or default value if an error occurs

    Raises:
        Exception: If no default value is provided and no error handler is specified
    """
    try:
        return operation()
    except Exception as e:
        # Log the error
        log_error(e, component_name, log_level)

        # Call custom error handler if provided
        if error_handler:
            handler_result = error_handler(e)
            if handler_result is not None:
                return handler_result

        # Return default value if provided
        if default_value is not None:
            return default_value

        # Re-raise the error if no default value or handler result
        raise


def log_error(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    additional_message: Optional[str] = None,
) -> None:
    """
    Log an error with standardized formatting.

    Args:
        error: The exception to log
        component_name: Name of the component where the error occurred
        log_level: Log level to use (default: "error")
        additional_message: Optional additional message to include
    """
    # Get the logger for the component
    component_logger = get_logger(component_name)

    # Get the log function based on the log level
    log_func = getattr(component_logger, log_level, component_logger.error)

    # Format the error message
    error_message = f"{type(error).__name__}: {str(error)}"
    if additional_message:
        error_message = f"{additional_message}: {error_message}"

    # Log the error with traceback
    log_func(error_message, exc_info=True)


def handle_component_error(
    error: Exception,
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> "ErrorResult":
    """
    Handle errors from generic components.

    Args:
        error: The exception that occurred
        component_name: Name of the component where the error occurred
        component_type: Type of component (model, validator, etc.)
        error_class: SifakaError subclass to use for wrapping errors
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        ErrorResult with standardized error information
    """
    # Import locally to avoid circular imports
    from sifaka.core.results import ErrorResult

    # Get error type and message
    error_type = type(error).__name__
    error_message = str(error)

    # Log the error
    log_error(error, component_name, log_level)

    # Create metadata
    metadata = {
        "component": component_name,
        "component_type": component_type,
        "error_type": error_type,
        "timestamp": time.time(),
        **(additional_metadata or {}),
    }

    # Include traceback if requested
    if include_traceback:
        metadata["traceback"] = traceback.format_exc()

    # Create error result
    return ErrorResult(
        error_type=error_type,
        error_message=error_message,
        passed=False,
        message=f"Error in {component_name}: {error_message}",
        metadata=metadata,
        score=0.0,
    )


def create_error_handler(
    component_type: str, error_class: Type[SifakaError]
) -> Callable[[Exception, str, str, bool, Optional[Dict[str, Any]]], "ErrorResult"]:
    """
    Create a component-specific error handler.

    Args:
        component_type: Type of component (model, validator, etc.)
        error_class: SifakaError subclass to use for wrapping errors

    Returns:
        A component-specific error handler function
    """

    def handler(
        error: Exception,
        component_name: str,
        log_level: str = "error",
        include_traceback: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> "ErrorResult":
        return handle_component_error(
            error=error,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

    return handler


# Create specific error handlers using the factory
handle_chain_error = create_error_handler("Chain", ChainError)
handle_model_error = create_error_handler("Model", ModelError)
handle_rule_error = create_error_handler("Rule", RuleError)
handle_critic_error = create_error_handler("Critic", CriticError)
handle_classifier_error = create_error_handler("Classifier", ClassifierError)
handle_retrieval_error = create_error_handler("Retrieval", RetrievalError)
