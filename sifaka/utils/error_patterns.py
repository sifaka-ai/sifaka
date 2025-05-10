"""
Error handling patterns for Sifaka components.

This module provides standardized error handling patterns for different component types
in the Sifaka framework. It uses a factory pattern to create error handlers for specific
component types, reducing code duplication and ensuring consistent error handling.

## Component Types

The module provides error handling patterns for the following component types:
- Chain components
- Model providers
- Rules
- Critics
- Classifiers
- Retrieval components

## Factory Pattern

The module uses a factory pattern to create error handlers for specific component types:
- `create_error_handler`: Creates an error handler for a specific component type
- `handle_component_error`: Generic error handler used by all component-specific handlers

## Usage Examples

```python
from sifaka.utils.error_patterns import handle_chain_error, handle_model_error

# Handle chain errors
try:
    # Chain operation
    result = chain.run(prompt)
except Exception as e:
    # Handle chain error
    error_result = handle_chain_error(e, chain_name="MyChain")
    print(f"Chain error: {error_result.error_message}")

# Handle model errors
try:
    # Model operation
    response = model.generate(prompt)
except Exception as e:
    # Handle model error
    error_result = handle_model_error(e, model_name="gpt-4")
    print(f"Model error: {error_result.error_message}")

# Create a custom error handler for a new component type
from sifaka.utils.error_patterns import create_error_handler
handle_custom_error = create_error_handler("Custom", CustomError)

try:
    # Custom operation
    result = custom_component.process(data)
except Exception as e:
    # Handle custom error
    error_result = handle_custom_error(e, component_name="MyCustomComponent")
    print(f"Custom error: {error_result.error_message}")
```
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast
from pydantic import BaseModel

from sifaka.utils.errors import (
    SifakaError,
    ChainError,
    ModelError,
    RuleError,
    CriticError,
    ClassifierError,
    RetrievalError,
    handle_error,
    try_operation,
)

# Type variable for return type
T = TypeVar("T")


class ErrorResult(BaseModel):
    """Result of an error handling operation.

    This model provides a standardized structure for error results,
    including error type, message, and metadata.

    Attributes:
        error_type: Type of the error
        error_message: Human-readable error message
        component_name: Name of the component where the error occurred
        metadata: Additional error context and details
    """

    error_type: str
    error_message: str
    component_name: str
    metadata: Dict[str, Any] = {}


# Generic component error handling


def handle_component_error(
    error: Exception,
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Generic error handler for any component type.

    This function handles errors for any component type, converting
    generic exceptions to specific SifakaError types and returning
    standardized error results.

    Args:
        error: The exception to handle
        component_name: Name of the component where the error occurred
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to specific error type if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = error_class(
            f"{component_type} error in {component_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"{component_type}:{component_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=component_name,
        metadata=error_metadata,
    )


# Error handler factory


def create_error_handler(
    component_type: str, error_class: Type[SifakaError]
) -> Callable[[Exception, str, str, bool, Optional[Dict[str, Any]]], ErrorResult]:
    """Create an error handler for a specific component type.

    This factory function creates an error handler for a specific component type,
    using the generic handle_component_error function with the appropriate
    component type and error class.

    Args:
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion

    Returns:
        An error handler function for the specified component type
    """

    def handler(
        error: Exception,
        component_name: str,
        log_level: str = "error",
        include_traceback: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> ErrorResult:
        return handle_component_error(
            error=error,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

    # Set function name and docstring
    handler.__name__ = f"handle_{component_type.lower()}_error"
    handler.__doc__ = f"""Handle a {component_type.lower()} error and return a standardized error result.

    Args:
        error: The exception to handle
        component_name: Name of the {component_type.lower()} where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """

    return handler


# Create specific error handlers using the factory
handle_chain_error = create_error_handler("Chain", ChainError)
handle_model_error = create_error_handler("Model", ModelError)
handle_rule_error = create_error_handler("Rule", RuleError)
handle_critic_error = create_error_handler("Critic", CriticError)
handle_classifier_error = create_error_handler("Classifier", ClassifierError)
handle_retrieval_error = create_error_handler("Retrieval", RetrievalError)


# Generic error result creation function
def create_error_result(
    error: Exception,
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Create a standardized error result for any component type.

    This function creates a standardized error result for any component type,
    using the appropriate error handler based on the component type.

    Args:
        error: The exception that occurred
        component_name: Name of the component where the error occurred
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    return handle_component_error(
        error=error,
        component_name=component_name,
        component_type=component_type,
        error_class=error_class,
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )


# Factory function for creating component-specific error result functions
def create_error_result_factory(component_type: str, error_class: Type[SifakaError]) -> Callable:
    """Create an error result factory for a specific component type.

    This factory function creates an error result function for a specific component type,
    using the generic create_error_result function with the appropriate component type and error class.

    Args:
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion

    Returns:
        An error result function for the specified component type
    """

    def factory(
        error: Exception,
        component_name: str,
        log_level: str = "error",
        include_traceback: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> ErrorResult:
        """Create a standardized error result for a specific component type."""
        return create_error_result(
            error=error,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

    return factory


# Create component-specific error result functions
create_chain_error_result = create_error_result_factory("Chain", ChainError)
create_model_error_result = create_error_result_factory("Model", ModelError)
create_rule_error_result = create_error_result_factory("Rule", RuleError)
create_critic_error_result = create_error_result_factory("Critic", CriticError)
create_classifier_error_result = create_error_result_factory("Classifier", ClassifierError)
create_retrieval_error_result = create_error_result_factory("Retrieval", RetrievalError)


# Add __all__ list for exports
__all__ = [
    "ErrorResult",
    "handle_component_error",
    "create_error_handler",
    "handle_chain_error",
    "handle_model_error",
    "handle_rule_error",
    "handle_critic_error",
    "handle_classifier_error",
    "handle_retrieval_error",
    "create_error_result",
    "create_error_result_factory",
    "create_chain_error_result",
    "create_model_error_result",
    "create_rule_error_result",
    "create_critic_error_result",
    "create_classifier_error_result",
    "create_retrieval_error_result",
]
