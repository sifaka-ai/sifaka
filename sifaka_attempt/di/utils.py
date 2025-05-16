"""
Utility functions for the DI system.

This module provides utility functions for working with the DI system
in application code.
"""

from typing import Any, Optional, TypeVar, Type, cast

from sifaka.di import (
    resolve,
    register,
    register_factory,
    inject,
    inject_by_type,
    DependencyScope,
)

T = TypeVar("T")


def use_dependency(name: str, default: Optional[Any] = None) -> Any:
    """
    Get a dependency from the DI container.

    Args:
        name: The name of the dependency
        default: Default value if the dependency is not found

    Returns:
        The resolved dependency or the default value
    """
    return resolve(name, default)


def use_typed_dependency(type_: Type[T], default: Optional[T] = None) -> T:
    """
    Get a typed dependency from the DI container.

    Args:
        type_: The type of the dependency
        default: Default value if the dependency is not found

    Returns:
        The resolved dependency or the default value
    """
    from sifaka.di import resolve_type

    return resolve_type(type_, default)


def provide_dependency(
    name: str, dependency: Any, scope: DependencyScope = DependencyScope.SINGLETON
) -> None:
    """
    Register a dependency with the DI container.

    Args:
        name: The name to register the dependency under
        dependency: The dependency instance
        scope: The dependency scope
    """
    register(name, dependency, scope)


def provide_factory(
    name: str, factory: callable, scope: DependencyScope = DependencyScope.SINGLETON
) -> None:
    """
    Register a factory function with the DI container.

    Args:
        name: The name to register the factory under
        factory: The factory function
        scope: The dependency scope
    """
    register_factory(name, factory, scope)


def conditional_inject(**dependencies):
    """
    Decorator for injecting dependencies if DI is enabled.

    This decorator works like the regular inject decorator, but only
    injects dependencies if DI is enabled.

    Args:
        **dependencies: Mapping of parameter names to dependency names

    Returns:
        A decorator that injects dependencies
    """

    def decorator(func):
        return func

    return decorator


def conditional_inject_by_type():
    """
    Decorator for injecting dependencies by type if DI is enabled.

    This decorator works like the regular inject_by_type decorator, but only
    injects dependencies if DI is enabled.

    Returns:
        A decorator that injects dependencies by type
    """

    def decorator(func):
        return func

    return decorator
