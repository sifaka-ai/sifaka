"""
Error Result Classes and Factories

This module provides classes and functions for creating standardized error results.
These are used to represent errors in a structured way that can be returned from
operations instead of raising exceptions.

## Classes
- **ErrorResult**: Result of an error handling operation

## Functions
- **create_error_result**: Create a standardized error result
- **create_error_result_factory**: Create a component-specific error result factory
- Component-specific error result creation functions:
  - **create_chain_error_result**
  - **create_model_error_result**
  - **create_rule_error_result**
  - **create_critic_error_result**
  - **create_classifier_error_result**
  - **create_retrieval_error_result**
"""

from typing import Any, Callable, Dict, Optional, Type
from pydantic import BaseModel
from .base import SifakaError
from .component import (
    ChainError,
    ClassifierError,
    CriticError,
    ModelError,
    RetrievalError,
    RuleError,
)
from .handling import handle_error


class ErrorResult(BaseModel):
    """Result of an error handling operation.

    This model provides a standardized structure for error results,
    including error type, message, component name, and metadata.
    It is used to represent errors in a structured way that can be
    returned from operations instead of raising exceptions.

    ## Architecture
    ErrorResult is a Pydantic model that represents errors in a structured format.
    It is used by the safe execution functions to return errors as values rather
    than raising exceptions, allowing for more flexible error handling patterns.

    ## Examples
    ```python
    # Creating an ErrorResult
    error = ErrorResult(
        error_type="ValidationError",
        error_message="Invalid input",
        component_name="TextValidator",
        metadata={"field": "text", "max_length": 100}
    )

    # Using ErrorResult in a function
    def process_data(data):
        if not validate(data):
            return ErrorResult(
                error_type="ValidationError",
                error_message="Invalid data",
                component_name="DataProcessor",
                metadata={"data": data}
            )
        # Process data
        return result

    # Handling ErrorResult
    result = process_data(input_data)
    if isinstance(result, ErrorResult):
        print(f"Error: {result.error_message}")
        print(f"Component: {result.component_name}")
        print(f"Error type: {result.error_type}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"Result: {result}")
    ```

    Attributes:
        error_type (str): Type of the error (e.g., "ValidationError")
        error_message (str): Human-readable error message
        component_name (str): Name of the component where the error occurred
        metadata (Dict[str, Any]): Additional error context and details
    """

    error_type: str
    error_message: str
    component_name: str
    metadata: Dict[str, Any] = {}


def create_error_result(
    error: Exception,
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    log_level: Optional[str] = "error",
    include_traceback: Optional[bool] = True,
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
    if not isinstance(error, SifakaError):
        error = error_class(
            f"{component_type} error in {component_name}: {str(error)}",
            metadata=additional_metadata,
        )
    error_metadata = handle_error(
        error,
        component_name=f"{component_type}:{component_name}",
        log_level=log_level if log_level is not None else "error",
        include_traceback=include_traceback if include_traceback is not None else True,
        additional_metadata=additional_metadata,
    )
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=component_name,
        metadata=error_metadata,
    )


def create_error_result_factory(
    component_type: str, error_class: Type[SifakaError]
) -> Callable[
    [Exception, str, Optional[str], Optional[bool], Optional[Dict[str, Any]]], ErrorResult
]:
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
        log_level: Optional[str] = "error",
        include_traceback: Optional[bool] = True,
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


create_chain_error_result: Callable[
    [Exception, str, Optional[str], Optional[bool], Optional[Dict[str, Any]]], ErrorResult
] = create_error_result_factory("Chain", ChainError)
create_model_error_result: Callable[
    [Exception, str, Optional[str], Optional[bool], Optional[Dict[str, Any]]], ErrorResult
] = create_error_result_factory("Model", ModelError)
create_rule_error_result: Callable[
    [Exception, str, Optional[str], Optional[bool], Optional[Dict[str, Any]]], ErrorResult
] = create_error_result_factory("Rule", RuleError)
create_critic_error_result: Callable[
    [Exception, str, Optional[str], Optional[bool], Optional[Dict[str, Any]]], ErrorResult
] = create_error_result_factory("Critic", CriticError)
create_classifier_error_result: Callable[
    [Exception, str, Optional[str], Optional[bool], Optional[Dict[str, Any]]], ErrorResult
] = create_error_result_factory("Classifier", ClassifierError)
create_retrieval_error_result: Callable[
    [Exception, str, Optional[str], Optional[bool], Optional[Dict[str, Any]]], ErrorResult
] = create_error_result_factory("Retrieval", RetrievalError)
