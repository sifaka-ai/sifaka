"""
Dependency Injection Module

This module provides standardized dependency injection patterns for Sifaka components.
It defines base classes and utilities for dependency injection, ensuring consistent
behavior across all components.

## Usage Examples

```python
from sifaka.core.dependency import DependencyProvider, inject_dependencies

# Create a dependency provider
provider = DependencyProvider()
provider.register("model", OpenAIProvider("gpt-4"))
provider.register("validator", LengthValidator())

# Create a component with injected dependencies
@inject_dependencies
class MyComponent:
    def __init__(self, model=None, validator=None):
        self.model = model
        self.validator = validator

# Create an instance with dependencies injected
component = MyComponent()  # Dependencies automatically injected
```
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from sifaka.utils.errors import DependencyError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class DependencyProvider:
    """
    Dependency provider for Sifaka components.

    This class provides a registry for dependencies, allowing components
    to request dependencies by name or type.
    """

    _instance = None
    _dependencies: Dict[str, Any] = {}

    def __new__(cls) -> "DependencyProvider":
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(DependencyProvider, cls).__new__(cls)
            cls._instance._dependencies = {}
        return cls._instance

    def register(self, name: str, dependency: Any) -> None:
        """
        Register a dependency.

        Args:
            name: The dependency name
            dependency: The dependency instance
        """
        self._dependencies[name] = dependency
        logger.debug(f"Registered dependency {name}: {dependency.__class__.__name__}")

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Get a dependency by name.

        Args:
            name: The dependency name
            default: Optional default value if dependency not found

        Returns:
            The dependency instance or default value

        Raises:
            DependencyError: If dependency not found and no default provided
        """
        if name in self._dependencies:
            return self._dependencies[name]
        elif default is not None:
            return default
        else:
            raise DependencyError(f"Dependency not found: {name}")

    def get_by_type(self, dependency_type: Type[T], default: Optional[T] = None) -> T:
        """
        Get a dependency by type.

        Args:
            dependency_type: The dependency type
            default: Optional default value if dependency not found

        Returns:
            The dependency instance or default value

        Raises:
            DependencyError: If dependency not found and no default provided
        """
        # Find dependencies of the specified type
        matching_dependencies = [
            dep for dep in self._dependencies.values() if isinstance(dep, dependency_type)
        ]

        if matching_dependencies:
            return cast(T, matching_dependencies[0])
        elif default is not None:
            return default
        else:
            raise DependencyError(f"Dependency not found for type: {dependency_type.__name__}")

    def clear(self) -> None:
        """Clear all registered dependencies."""
        self._dependencies.clear()
        logger.debug("Cleared all dependencies")


def inject_dependencies(func_or_class: F) -> F:
    """
    Decorator for injecting dependencies.

    This decorator injects dependencies into function or class constructor
    parameters, using the DependencyProvider to resolve dependencies.

    Args:
        func_or_class: The function or class to inject dependencies into

    Returns:
        The decorated function or class
    """
    if inspect.isclass(func_or_class):
        # If decorating a class, wrap the __init__ method
        original_init = func_or_class.__init__

        @functools.wraps(original_init)
        def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # Get parameter names and annotations from __init__
            sig = inspect.signature(original_init)
            provider = DependencyProvider()

            # Inject dependencies for parameters not provided in kwargs
            for param_name, param in sig.parameters.items():
                if param_name != "self" and param_name not in kwargs:
                    # Try to get dependency by name
                    try:
                        kwargs[param_name] = provider.get(param_name, None)
                    except DependencyError:
                        # If not found by name, try by type annotation
                        if param.annotation != inspect.Parameter.empty:
                            try:
                                kwargs[param_name] = provider.get_by_type(param.annotation, None)
                            except DependencyError:
                                # If not found and has default, use default
                                if param.default != inspect.Parameter.empty:
                                    kwargs[param_name] = param.default

            # Call original __init__
            original_init(self, *args, **kwargs)

        func_or_class.__init__ = wrapped_init
        return cast(F, func_or_class)
    else:
        # If decorating a function, wrap the function
        @functools.wraps(func_or_class)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get parameter names and annotations from function
            sig = inspect.signature(func_or_class)
            provider = DependencyProvider()

            # Inject dependencies for parameters not provided in kwargs
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs:
                    # Try to get dependency by name
                    try:
                        kwargs[param_name] = provider.get(param_name, None)
                    except DependencyError:
                        # If not found by name, try by type annotation
                        if param.annotation != inspect.Parameter.empty:
                            try:
                                kwargs[param_name] = provider.get_by_type(param.annotation, None)
                            except DependencyError:
                                # If not found and has default, use default
                                if param.default != inspect.Parameter.empty:
                                    kwargs[param_name] = param.default

            # Call original function
            return func_or_class(*args, **kwargs)

        return cast(F, wrapper)


class DependencyInjector:
    """
    Utility class for injecting dependencies.

    This class provides methods for injecting dependencies into
    objects and functions.
    """

    @staticmethod
    def inject(obj: Any, **dependencies: Any) -> None:
        """
        Inject dependencies into an object.

        Args:
            obj: The object to inject dependencies into
            **dependencies: The dependencies to inject
        """
        for name, dependency in dependencies.items():
            if hasattr(obj, name):
                setattr(obj, name, dependency)
                logger.debug(f"Injected dependency {name} into {obj.__class__.__name__}")

    @staticmethod
    def create_with_dependencies(cls: Type[T], **dependencies: Any) -> T:
        """
        Create an object with injected dependencies.

        Args:
            cls: The class to create an instance of
            **dependencies: The dependencies to inject

        Returns:
            An instance of the class with dependencies injected
        """
        # Create instance
        instance = cls(**dependencies)

        # Inject any remaining dependencies
        DependencyInjector.inject(instance, **dependencies)

        return instance


def provide_dependency(name: str, dependency: Any) -> None:
    """
    Register a dependency with the global provider.

    Args:
        name: The dependency name
        dependency: The dependency instance
    """
    provider = DependencyProvider()
    provider.register(name, dependency)


def get_dependency(name: str, default: Optional[Any] = None) -> Any:
    """
    Get a dependency from the global provider.

    Args:
        name: The dependency name
        default: Optional default value if dependency not found

    Returns:
        The dependency instance or default value

    Raises:
        DependencyError: If dependency not found and no default provided
    """
    provider = DependencyProvider()
    return provider.get(name, default)


def get_dependency_by_type(dependency_type: Type[T], default: Optional[T] = None) -> T:
    """
    Get a dependency by type from the global provider.

    Args:
        dependency_type: The dependency type
        default: Optional default value if dependency not found

    Returns:
        The dependency instance or default value

    Raises:
        DependencyError: If dependency not found and no default provided
    """
    provider = DependencyProvider()
    return provider.get_by_type(dependency_type, default)


def clear_dependencies() -> None:
    """Clear all registered dependencies."""
    provider = DependencyProvider()
    provider.clear()
