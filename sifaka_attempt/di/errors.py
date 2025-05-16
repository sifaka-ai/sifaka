"""
Error types for the dependency injection system.

This module defines custom error types for the dependency injection system,
providing clear and specific error messages for different failure cases.
"""

from typing import Any, List, Optional, Type


class DependencyError(Exception):
    """Base class for all dependency injection errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class DependencyNotFoundError(DependencyError):
    """Error raised when a dependency is not found."""

    def __init__(self, name: str):
        self.name = name
        message = f"Dependency '{name}' not found"
        super().__init__(message)


class TypeNotFoundError(DependencyError):
    """Error raised when a type is not found."""

    def __init__(self, type_: Type):
        self.type_ = type_
        message = f"Type '{type_.__name__}' not found"
        super().__init__(message)


class CircularDependencyError(DependencyError):
    """Error raised when a circular dependency is detected."""

    def __init__(self, dependency_chain: List[str]):
        self.dependency_chain = dependency_chain
        chain_str = " -> ".join(dependency_chain)
        message = f"Circular dependency detected: {chain_str}"
        super().__init__(message)


class DependencyResolutionError(DependencyError):
    """Error raised when a dependency cannot be resolved."""

    def __init__(self, name: str, cause: Optional[Exception] = None):
        self.name = name
        self.cause = cause
        message = f"Failed to resolve dependency '{name}'"
        if cause:
            message += f": {str(cause)}"
        super().__init__(message)


class TypeResolutionError(DependencyError):
    """Error raised when a type cannot be resolved."""

    def __init__(self, type_: Type, cause: Optional[Exception] = None):
        self.type_ = type_
        self.cause = cause
        message = f"Failed to resolve type '{type_.__name__}'"
        if cause:
            message += f": {str(cause)}"
        super().__init__(message)


class ScopeError(DependencyError):
    """Error raised when there is a problem with scope management."""

    def __init__(self, scope: str, message: str):
        self.scope = scope
        full_message = f"Scope error for '{scope}': {message}"
        super().__init__(full_message)


class TypeMismatchError(DependencyError):
    """Error raised when a dependency's type does not match the expected type."""

    def __init__(self, name: str, expected_type: Type, actual_type: Type):
        self.name = name
        self.expected_type = expected_type
        self.actual_type = actual_type
        message = (
            f"Type mismatch for dependency '{name}': "
            f"expected {expected_type.__name__}, "
            f"got {actual_type.__name__}"
        )
        super().__init__(message)


class DependencyConfigurationError(DependencyError):
    """Error raised when there is a problem with the dependency configuration."""

    def __init__(self, message: str):
        super().__init__(message)


class DependencyAlreadyRegisteredError(DependencyError):
    """Error raised when attempting to register a dependency that already exists."""

    def __init__(self, name: str):
        self.name = name
        message = f"Dependency '{name}' is already registered"
        super().__init__(message)


class TypeAlreadyRegisteredError(DependencyError):
    """Error raised when attempting to register a type that already exists."""

    def __init__(self, type_: Type):
        self.type_ = type_
        message = f"Type '{type_.__name__}' is already registered"
        super().__init__(message)


class DependencyValidationError(DependencyError):
    """Error raised when a dependency fails validation."""

    def __init__(self, name: str, message: str):
        self.name = name
        full_message = f"Validation error for dependency '{name}': {message}"
        super().__init__(full_message)


class InvalidDependencyError(DependencyError):
    """Error raised when a dependency is invalid."""

    def __init__(self, name: str, message: str):
        self.name = name
        full_message = f"Invalid dependency '{name}': {message}"
        super().__init__(full_message)
