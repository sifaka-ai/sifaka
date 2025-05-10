"""
Classifier Errors Module

This module provides standardized error classes for the Sifaka classifiers system.
It defines a hierarchy of error classes for different types of errors that can occur
during classification, with consistent error handling and reporting.

## Error Hierarchy
1. **ClassifierError**: Base class for all classifier errors
   - **ImplementationError**: Raised when classifier implementation fails
   - **ConfigError**: Raised when configuration is invalid
   - **StateError**: Raised when state operations fail
   - **PluginError**: Raised when plugin operations fail

## Usage Examples
```python
from sifaka.classifiers.v2.errors import (
    ClassifierError, ImplementationError, ConfigError, StateError
)

# Raise an implementation error
try:
    # Classification operation
    result = implementation.classify(text)
except Exception as e:
    raise ImplementationError(f"Classification failed: {str(e)}", metadata={"text_length": len(text)})

# Handle classifier errors
try:
    # Classification operation
    result = classifier.classify(text)
except ImplementationError as e:
    print(f"Implementation error: {e.message}")
    print(f"Implementation metadata: {e.metadata}")
except ConfigError as e:
    print(f"Configuration error: {e.message}")
    print(f"Configuration metadata: {e.metadata}")
except ClassifierError as e:
    print(f"Classifier error: {e.message}")
    print(f"Classifier metadata: {e.metadata}")
```
"""

from typing import Any, Dict, Optional


class ClassifierError(Exception):
    """Base class for all classifier errors."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a classifier error.
        
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


class ImplementationError(ClassifierError):
    """Error raised when classifier implementation fails."""
    pass


class ConfigError(ClassifierError):
    """Error raised when configuration is invalid."""
    pass


class StateError(ClassifierError):
    """Error raised when state operations fail."""
    pass


class PluginError(ClassifierError):
    """Error raised when plugin operations fail."""
    pass


def handle_error(
    error: Exception,
    component_name: str,
    component_type: str = "Classifier",
    error_class: type[ClassifierError] = ClassifierError,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ClassifierError:
    """
    Handle an error by converting it to a classifier error.
    
    Args:
        error: The error to handle
        component_name: Name of the component where the error occurred
        component_type: Type of the component (e.g., "Classifier", "Implementation")
        error_class: Classifier error class to use
        additional_metadata: Additional metadata to include
        
    Returns:
        A classifier error
    """
    # If already a classifier error, add metadata and return
    if isinstance(error, ClassifierError):
        if additional_metadata:
            error.metadata.update(additional_metadata)
        return error
    
    # Convert to classifier error
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
    component_type: str = "Classifier",
    error_class: type[ClassifierError] = ClassifierError,
    additional_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Execute an operation with standardized error handling.
    
    Args:
        operation: The operation to execute
        component_name: Name of the component
        component_type: Type of the component
        error_class: Classifier error class to use
        additional_metadata: Additional metadata to include
        
    Returns:
        The result of the operation
        
    Raises:
        ClassifierError: If the operation fails
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
