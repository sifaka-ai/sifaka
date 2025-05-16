"""
Decorators for dependency injection.

This module provides decorators to inject dependencies into
functions and methods.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints

from sifaka.di import get_container
from sifaka.di.errors import DependencyError, DependencyNotFoundError

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def inject(*dependencies: str, **named_dependencies: str) -> Callable[[F], F]:
    """
    Inject dependencies into a function or method.

    This decorator injects dependencies into a function or method
    based on positional or keyword arguments.

    Args:
        *dependencies: Dependencies to inject as positional arguments
        **named_dependencies: Dependencies to inject as keyword arguments,
                           where keys are parameter names and values are
                           dependency names

    Returns:
        Decorated function

    Example:
        ```python
        @inject("logger", db="database")
        def process_data(data, logger, db):
            logger.info("Processing data")
            db.save(data)
        ```
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()

            # Determine self parameter for methods
            is_method = False
            if args and len(args) > 0 and len(sig.parameters) > 0:
                # This heuristic detects if this is likely a method call
                first_param_name = list(sig.parameters.keys())[0]
                if first_param_name in ("self", "cls"):
                    is_method = True

            # Adjust offset for methods (skip self/cls)
            offset = 1 if is_method else 0

            # Inject positional dependencies
            for i, dep_name in enumerate(dependencies):
                param_index = i + offset
                param_names = list(sig.parameters.keys())

                if param_index < len(param_names) and param_names[param_index] not in kwargs:
                    # Ensure positional arg wasn't already provided
                    if param_index >= len(args):
                        try:
                            kwargs[param_names[param_index]] = container.resolve(dep_name)
                        except DependencyError as e:
                            raise DependencyError(
                                f"Failed to inject positional dependency '{dep_name}' into "
                                f"parameter '{param_names[param_index]}' of {func.__name__}: {str(e)}"
                            )

            # Inject named dependencies
            for param_name, dep_name in named_dependencies.items():
                if param_name in sig.parameters and param_name not in kwargs:
                    try:
                        kwargs[param_name] = container.resolve(dep_name)
                    except DependencyError as e:
                        raise DependencyError(
                            f"Failed to inject named dependency '{dep_name}' into "
                            f"parameter '{param_name}' of {func.__name__}: {str(e)}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def inject_by_type(*, use_names: bool = False) -> Callable[[F], F]:
    """
    Inject dependencies by parameter type annotations.

    This decorator injects dependencies based on the type annotations
    of the function parameters.

    Args:
        use_names: Whether to use parameter names as dependency names
                   if a type is not registered

    Returns:
        Decorated function

    Example:
        ```python
        @inject_by_type()
        def process_data(data: str, logger: Logger, db: Database):
            logger.info("Processing data")
            db.save(data)
        ```
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()

            # Determine self parameter for methods
            offset = 0
            if args and len(args) > 0 and len(sig.parameters) > 0:
                first_param_name = list(sig.parameters.keys())[0]
                if first_param_name in ("self", "cls"):
                    offset = 1

            # Process parameters
            for i, (param_name, param) in enumerate(list(sig.parameters.items())[offset:], offset):
                # Skip if already in kwargs or provided as positional arg
                if param_name in kwargs or i < len(args):
                    continue

                # Skip parameters without type annotations
                if param_name not in type_hints:
                    continue

                param_type = type_hints[param_name]

                # Try to resolve by type
                try:
                    if container.has_type(param_type):
                        kwargs[param_name] = container.resolve_type(param_type)
                        continue

                    # If type resolution failed and use_names is enabled, try by name
                    if use_names and container.has_dependency(param_name):
                        kwargs[param_name] = container.resolve(param_name)
                        continue

                    # Special case for optional parameters with default values
                    if param.default is not inspect.Parameter.empty:
                        continue

                    raise DependencyNotFoundError(param_name)
                except DependencyError as e:
                    raise DependencyError(
                        f"Failed to inject dependency for parameter '{param_name}' of "
                        f"{func.__name__} with type '{param_type.__name__}': {str(e)}"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def injectable(cls: Type[T]) -> Type[T]:
    """
    Make a class injectable by type.

    This decorator registers a class with the dependency container
    so it can be injected by type.

    Args:
        cls: The class to make injectable

    Returns:
        The decorated class

    Example:
        ```python
        @injectable
        class Logger:
            def log(self, message):
                print(message)
        ```
    """
    container = get_container()

    # Check if already registered
    if not container.has_type(cls):
        # Register factory function
        container.register_type_factory(cls, cls)

    return cls
