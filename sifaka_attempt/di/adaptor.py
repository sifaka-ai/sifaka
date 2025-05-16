"""
Temporary adaptor for legacy code compatibility.

This module provides temporary compatibility functions to help
migrate from the old DI system to the new one. This is not meant
for long-term use and should be removed once migration is complete.
"""

import warnings
from typing import Any, Callable, Optional, Type, TypeVar

from sifaka.di import (
    Container,
    get_default_container,
    inject,
    inject_by_type,
    make_injectable,
)

T = TypeVar("T")


# Show deprecation warnings for all functions in this module
def _deprecated(message: str) -> Callable:
    """Decorator to mark functions as deprecated."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Legacy API functions
@_deprecated("get_container() is deprecated, use get_default_container() instead")
def get_container() -> Container:
    """
    Get the default container (legacy API).

    Returns:
        The default container
    """
    return get_default_container()


@_deprecated("register() is deprecated, use container.register() instead")
def register(name: str, instance: Any) -> None:
    """
    Register an instance with the default container (legacy API).

    Args:
        name: The dependency name
        instance: The instance to register
    """
    get_default_container().register(name, instance)


@_deprecated("register_factory() is deprecated, use container.register_factory() instead")
def register_factory(name: str, factory: Callable[[], Any], singleton: bool = True) -> None:
    """
    Register a factory with the default container (legacy API).

    Args:
        name: The dependency name
        factory: The factory function
        singleton: Whether to cache the factory result
    """
    get_default_container().register_factory(name, factory, singleton)


@_deprecated("resolve() is deprecated, use container.resolve() instead")
def resolve(name: str) -> Any:
    """
    Resolve a dependency by name (legacy API).

    Args:
        name: The dependency name

    Returns:
        The resolved dependency
    """
    return get_default_container().resolve(name)


@_deprecated("resolve_type() is deprecated, use container.resolve_type() instead")
def resolve_type(type_: Type[T]) -> T:
    """
    Resolve a dependency by type (legacy API).

    Args:
        type_: The type to resolve

    Returns:
        The resolved dependency
    """
    return get_default_container().resolve_type(type_)


# Legacy bootstrapping functions
@_deprecated("initialize_di_container() is deprecated, use new DI system instead")
def initialize_di_container() -> None:
    """Legacy initialization function."""
    pass


@_deprecated("register_model() is deprecated, use container.register_factory() instead")
def register_model(name: str, model_type: str, **kwargs) -> None:
    """Legacy model registration function."""
    from sifaka.models import create_model

    get_default_container().register_factory(name, lambda: create_model(model_type, **kwargs))


@_deprecated("register_validator() is deprecated, use container.register() instead")
def register_validator(name: str, validator: Any) -> None:
    """Legacy validator registration function."""
    get_default_container().register(name, validator)


@_deprecated("register_critic() is deprecated, use container.register() instead")
def register_critic(name: str, critic: Any) -> None:
    """Legacy critic registration function."""
    get_default_container().register(name, critic)
