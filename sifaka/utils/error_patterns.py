"""
Error handling patterns for Sifaka components.

This module provides standardized error handling patterns for different component types
in the Sifaka framework. Each component type has specific error handling requirements
and patterns, which are implemented in this module.

## Component Types

The module provides error handling patterns for the following component types:
- Chain components
- Model providers
- Rules
- Critics
- Classifiers
- Retrieval components

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
```
"""

from typing import Any, Dict, Optional, TypeVar, Union, cast
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


# Chain error handling


def handle_chain_error(
    error: Exception,
    chain_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Handle a chain error and return a standardized error result.

    Args:
        error: The exception to handle
        chain_name: Name of the chain where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to ChainError if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = ChainError(
            f"Chain error in {chain_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"Chain:{chain_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=chain_name,
        metadata=error_metadata,
    )


# Model error handling


def handle_model_error(
    error: Exception,
    model_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Handle a model error and return a standardized error result.

    Args:
        error: The exception to handle
        model_name: Name of the model where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to ModelError if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = ModelError(
            f"Model error in {model_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"Model:{model_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=model_name,
        metadata=error_metadata,
    )


# Rule error handling


def handle_rule_error(
    error: Exception,
    rule_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Handle a rule error and return a standardized error result.

    Args:
        error: The exception to handle
        rule_name: Name of the rule where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to RuleError if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = RuleError(
            f"Rule error in {rule_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"Rule:{rule_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=rule_name,
        metadata=error_metadata,
    )


# Critic error handling


def handle_critic_error(
    error: Exception,
    critic_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Handle a critic error and return a standardized error result.

    Args:
        error: The exception to handle
        critic_name: Name of the critic where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to CriticError if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = CriticError(
            f"Critic error in {critic_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"Critic:{critic_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=critic_name,
        metadata=error_metadata,
    )


# Classifier error handling


def handle_classifier_error(
    error: Exception,
    classifier_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Handle a classifier error and return a standardized error result.

    Args:
        error: The exception to handle
        classifier_name: Name of the classifier where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to ClassifierError if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = ClassifierError(
            f"Classifier error in {classifier_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"Classifier:{classifier_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=classifier_name,
        metadata=error_metadata,
    )


# Retrieval error handling


def handle_retrieval_error(
    error: Exception,
    retriever_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Handle a retrieval error and return a standardized error result.

    Args:
        error: The exception to handle
        retriever_name: Name of the retriever where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to RetrievalError if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = RetrievalError(
            f"Retrieval error in {retriever_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"Retrieval:{retriever_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=retriever_name,
        metadata=error_metadata,
    )


# Add __all__ list for exports
__all__ = [
    "ErrorResult",
    "handle_chain_error",
    "handle_model_error",
    "handle_rule_error",
    "handle_critic_error",
    "handle_classifier_error",
    "handle_retrieval_error",
]
