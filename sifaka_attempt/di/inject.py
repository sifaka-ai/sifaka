"""
Dependency injection decorators.

This module provides simple, clean decorators for dependency injection
that work with static analysis tools and don't rely on global state.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast, get_type_hints

from sifaka.di.container import Container, DependencyNotFoundError

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def create_inject(container: Container):
    """
    Create an inject decorator bound to a specific container.

    Args:
        container: The DI container to use for dependency resolution

    Returns:
        A decorator factory for injecting dependencies
    """

    def inject(*dependencies, **named_dependencies):
        """
        Inject dependencies into a function or method.

        This decorator injects dependencies into a function based on
        explicit mapping of parameter names to dependency names.

        Args:
            *dependencies: Dependencies to inject positionally
            **named_dependencies: Dependencies to inject by parameter name

        Returns:
            Decorated function that injects dependencies
        """

        def decorator(func: F) -> F:
            sig = inspect.signature(func)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Determine if this is a method call (has self/cls parameter)
                is_method = False
                if args and len(args) > 0 and len(sig.parameters) > 0:
                    first_param_name = list(sig.parameters.keys())[0]
                    if first_param_name in ("self", "cls"):
                        is_method = True

                # Offset for methods (skip self/cls)
                offset = 1 if is_method else 0

                # Inject positional dependencies
                for i, dep_name in enumerate(dependencies):
                    param_index = i + offset
                    param_names = list(sig.parameters.keys())

                    if param_index < len(param_names) and param_names[param_index] not in kwargs:
                        # Skip if a positional arg was provided
                        if param_index >= len(args):
                            kwargs[param_names[param_index]] = container.resolve(dep_name)

                # Inject named dependencies
                for param_name, dep_name in named_dependencies.items():
                    if param_name in sig.parameters and param_name not in kwargs:
                        kwargs[param_name] = container.resolve(dep_name)

                return func(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    def inject_by_type(func: F) -> F:
        """
        Inject dependencies by parameter type.

        This decorator injects dependencies based on parameter type annotations.

        Args:
            func: The function to decorate

        Returns:
            Decorated function that injects dependencies by type
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if this is a method call (has self/cls parameter)
            offset = 0
            if args and len(args) > 0 and len(sig.parameters) > 0:
                first_param_name = list(sig.parameters.keys())[0]
                if first_param_name in ("self", "cls"):
                    offset = 1

            # Process parameters with type annotations
            for i, (param_name, param) in enumerate(list(sig.parameters.items())[offset:], offset):
                # Skip if parameter already has a value
                if param_name in kwargs or i < len(args):
                    continue

                # Skip parameters without type annotations
                if param_name not in type_hints:
                    continue

                param_type = type_hints[param_name]

                # Skip if parameter has a default value
                if param.default is not inspect.Parameter.empty:
                    continue

                try:
                    # Try to resolve by type
                    kwargs[param_name] = container.resolve_type(param_type)
                except DependencyNotFoundError:
                    # Let it fail naturally if no dependency is found and no default
                    pass

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return inject, inject_by_type


def injectable(container: Container):
    """
    Create a decorator for injectable classes.

    Args:
        container: The DI container to register with

    Returns:
        A decorator for making classes injectable
    """

    def decorator(cls: Type[T]) -> Type[T]:
        """
        Make a class injectable.

        This decorator registers a class with the DI container so it can
        be resolved by type.

        Args:
            cls: The class to make injectable

        Returns:
            The original class
        """
        container.register_type_factory(cls, cls)
        return cls

    return decorator
