"""
Dependency Injection System for Sifaka

This package provides a clean, straightforward dependency injection system
designed to avoid circular dependencies and enable static analysis.
"""

from sifaka.di.container import (
    Container,
    Dependency,
    DIError,
    DependencyNotFoundError,
    CircularDependencyError,
)

from sifaka.di.inject import create_inject, injectable

__all__ = [
    # Container
    "Container",
    "Dependency",
    # Errors
    "DIError",
    "DependencyNotFoundError",
    "CircularDependencyError",
    # Factory functions
    "create_inject",
    "injectable",
    # Create a default container
    "create_container",
    "get_default_container",
    # Decorators (bound to default container)
    "inject",
    "inject_by_type",
    "make_injectable",
]


def create_container() -> Container:
    """
    Create a new DI container.

    Returns:
        A new Container instance
    """
    return Container()


# Create a default container for convenience
_default_container = create_container()


def get_default_container() -> Container:
    """
    Get the default container instance.

    Returns:
        The default Container instance
    """
    return _default_container


# Create decorators bound to the default container
inject, inject_by_type = create_inject(_default_container)
make_injectable = injectable(_default_container)
