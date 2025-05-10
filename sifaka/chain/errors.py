"""
Chain Errors Module

This module provides standardized error classes for the Sifaka chain system.
It defines a hierarchy of error classes for different types of errors that can occur
during chain execution, with consistent error handling and reporting.

## Error Hierarchy
1. **ChainError**: Base class for all chain errors
   - **ModelError**: Raised when model generation fails
   - **ValidationError**: Raised when validation fails
   - **ImproverError**: Raised when improver refinement fails
   - **FormatterError**: Raised when result formatting fails
   - **PluginError**: Raised when plugin operations fail
   - **ConfigError**: Raised when configuration is invalid
   - **StateError**: Raised when state operations fail

## Usage Examples
```python
from sifaka.chain.v2.errors import (
    ChainError, ModelError, ValidationError, ImproverError
)

# Raise a model error
try:
    # Model operation
    output = model.generate(prompt)
except Exception as e:
    raise ModelError(f"Model generation failed: {str(e)}", metadata={"model": model.name})

# Handle chain errors
try:
    # Chain operation
    result = chain.run(prompt)
except ValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"Validation metadata: {e.metadata}")
except ModelError as e:
    print(f"Model error: {e.message}")
    print(f"Model metadata: {e.metadata}")
except ChainError as e:
    print(f"Chain error: {e.message}")
    print(f"Chain metadata: {e.metadata}")
```
"""

from typing import Any, Dict, Optional


class ChainError(Exception):
    """Base class for all chain errors."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a chain error.
        
        Args:
            message: Error message
            metadata: Additional error metadata
        """
        self.message = message
        self.metadata = metadata or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        """Get string representation of the error."""
        if self.metadata:
            return f"{self.message} (metadata: {self.metadata})"
        return self.message


class ModelError(ChainError):
    """Error raised when model generation fails."""
    pass


class ValidationError(ChainError):
    """Error raised when validation fails."""
    pass


class ImproverError(ChainError):
    """Error raised when improver refinement fails."""
    pass


class FormatterError(ChainError):
    """Error raised when result formatting fails."""
    pass


class PluginError(ChainError):
    """Error raised when plugin operations fail."""
    pass


class ConfigError(ChainError):
    """Error raised when configuration is invalid."""
    pass


class StateError(ChainError):
    """Error raised when state operations fail."""
    pass


def handle_error(
    error: Exception,
    component_name: str,
    component_type: str = "Chain",
    error_class: type[ChainError] = ChainError,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ChainError:
    """
    Handle an error by converting it to a chain error.
    
    Args:
        error: The error to handle
        component_name: Name of the component where the error occurred
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: Chain error class to use
        additional_metadata: Additional metadata to include
        
    Returns:
        A chain error
    """
    # If already a chain error, add metadata and return
    if isinstance(error, ChainError):
        if additional_metadata:
            error.metadata.update(additional_metadata)
        return error
    
    # Convert to chain error
    metadata = additional_metadata or {}
    metadata.update({
        "component_name": component_name,
        "component_type": component_type,
        "error_type": type(error).__name__,
    })
    
    return error_class(
        f"{component_type} error in {component_name}: {str(error)}",
        metadata=metadata,
    )


def safely_execute(
    operation: callable,
    component_name: str,
    component_type: str = "Chain",
    error_class: type[ChainError] = ChainError,
    additional_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Execute an operation with standardized error handling.
    
    Args:
        operation: The operation to execute
        component_name: Name of the component
        component_type: Type of the component
        error_class: Chain error class to use
        additional_metadata: Additional metadata to include
        
    Returns:
        The result of the operation
        
    Raises:
        ChainError: If the operation fails
    """
    try:
        return operation()
    except Exception as e:
        raise handle_error(
            error=e,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            additional_metadata=additional_metadata,
        )
