"""
Dependency Injector Module

This module provides utilities for dependency injection, including the
DependencyInjector class and the inject_dependencies decorator.

## Components
- **DependencyInjector**: Utility class for manual dependency injection
- **inject_dependencies**: Decorator for automatic dependency injection

## Usage Examples
```python
from sifaka.core.dependency.injector import inject_dependencies, DependencyInjector

# Use the decorator for automatic injection
@inject_dependencies
class MyComponent:
    def __init__(self, model=None, validator=None):
        self.model = model  # Injected from DependencyProvider
        self.validator = validator  # Injected from DependencyProvider

# Use the injector for manual injection
injector = DependencyInjector()
dependencies = (injector and injector.inject({"model": None, "validator": None})
```

## Error Handling
- Raises DependencyError for missing dependencies
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast
from .provider import DependencyProvider

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class DependencyInjector:
    """
    Utility class for manual dependency injection.

    This class provides methods for manually injecting dependencies into
    functions, methods, or classes. It uses the DependencyProvider to
    resolve dependencies by name or type.

    ## Architecture
    The DependencyInjector is a utility class that provides methods for
    manually injecting dependencies. It uses the DependencyProvider to
    resolve dependencies and supports injection by parameter name or
    type annotation.

    ## Lifecycle
    1. **Initialization**: Creates an instance with optional session and request IDs
    2. **Injection**: Injects dependencies into functions, methods, or classes
    3. **Resolution**: Resolves dependencies by name or type

    ## Examples
    ```python
    from sifaka.core.dependency.injector import DependencyInjector

    # Create an injector
    injector = DependencyInjector()

    # Inject dependencies into a dictionary
    dependencies = (injector and injector.inject({"model": None, "validator": None})

    # Inject dependencies into a function
    def process_data(model=None, validator=None):
        # Use injected dependencies
        pass

    injected_func = (injector and injector.inject_function(process_data)
    result = injected_func()  # Dependencies automatically injected
    ```

    Attributes:
        provider (DependencyProvider): The dependency provider
        session_id (Optional[str]): The session ID for scoped dependencies
        request_id (Optional[str]): The request ID for scoped dependencies
    """

    def __init__(
        self, session_id: Optional[Optional[str]] = None, request_id: Optional[str] = None
    ) -> None:
        """
        Initialize a dependency injector.

        Args:
            session_id: Optional session ID for scoped dependencies
            request_id: Optional request ID for scoped dependencies
        """
        self.provider = DependencyProvider()
        self.session_id = session_id
        self.request_id = request_id

    def inject(self, dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject dependencies into a dictionary.

        This method injects dependencies into a dictionary, replacing None
        values with resolved dependencies from the provider.

        Args:
            dependencies: Dictionary of dependency names and values

        Returns:
            Dictionary with dependencies injected

        Example:
            ```python
            injector = DependencyInjector()
            dependencies = (injector and injector.inject({"model": None, "validator": None})
            ```
        """
        result = {}
        for name, value in dependencies.items():
            if value is None:
                result[name] = self.provider.get(
                    name, session_id=self.session_id, request_id=self.request_id
                )
            else:
                result[name] = value
        return result

    def inject_function(self, func: F) -> F:
        """
        Inject dependencies into a function.

        This method creates a wrapper function that injects dependencies
        into the original function's parameters.

        Args:
            func: The function to inject dependencies into

        Returns:
            A wrapper function with dependencies injected

        Example:
            ```python
            injector = DependencyInjector()

            def process_data(model=None, validator=None):
                # Use injected dependencies
                pass

            injected_func = (injector and injector.inject_function(process_data)
            result = injected_func()  # Dependencies automatically injected
            ```
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            sig = inspect.signature(func)
            params = sig.parameters
            injected_kwargs = {}
            for name, param in params.items():
                if name in kwargs:
                    continue
                if param.kind == param.POSITIONAL_OR_KEYWORD:
                    if len(args) > list(params.keys()).index(name):
                        continue
                if param.default is param.empty:
                    continue
                if param.default is not None:
                    continue
                try:
                    injected_kwargs[name] = self.provider.get(
                        name, session_id=self.session_id, request_id=self.request_id
                    )
                except Exception as e:
                    logger.debug(f"Failed to inject dependency {name}: {e}")
            merged_kwargs = {**injected_kwargs, **kwargs}
            return func(*args, **merged_kwargs)

        return cast(F, wrapper)

    def inject_method(self, method: F) -> F:
        """
        Inject dependencies into a method.

        This method creates a wrapper method that injects dependencies
        into the original method's parameters.

        Args:
            method: The method to inject dependencies into

        Returns:
            A wrapper method with dependencies injected

        Example:
            ```python
            injector = DependencyInjector()

            class MyComponent:
                def process_data(self, model=None, validator=None):
                    # Use injected dependencies
                    pass

            component = MyComponent()
            component.process_data = (injector.inject_method(component.process_data)
            result = (component.process_data()  # Dependencies automatically injected
            ```
        """

        @functools.wraps(method)
        def wrapper(self_arg, *args, **kwargs) -> Any:
            sig = inspect.signature(method)
            params = sig.parameters
            injected_kwargs = {}
            for name, param in list(params.items())[1:]:
                if name in kwargs:
                    continue
                if param.kind == param.POSITIONAL_OR_KEYWORD:
                    if len(args) > list(params.keys()).index(name) - 1:
                        continue
                if param.default is param.empty:
                    continue
                if param.default is not None:
                    continue
                try:
                    injected_kwargs[name] = self.provider.get(
                        name, session_id=self.session_id, request_id=self.request_id
                    )
                except Exception as e:
                    logger.debug(f"Failed to inject dependency {name}: {e}")
            merged_kwargs = {**injected_kwargs, **kwargs}
            return method(self_arg, *args, **merged_kwargs)

        return cast(F, wrapper)

    def inject_class(self, cls: Type[T]) -> Type[T]:
        """
        Inject dependencies into a class.

        This method creates a wrapper class that injects dependencies
        into the original class's constructor.

        Args:
            cls: The class to inject dependencies into

        Returns:
            A wrapper class with dependencies injected

        Example:
            ```python
            injector = DependencyInjector()

            class MyComponent:
                def __init__(self, model=None, validator=None):
                    self.model = model
                    self.validator = validator

            InjectedComponent = (injector.inject_class(MyComponent)
            component = InjectedComponent()  # Dependencies automatically injected
            ```
        """
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def injected_init(self, *args, **kwargs) -> None:
            sig = inspect.signature(orig_init)
            params = sig.parameters
            injected_kwargs = {}
            for name, param in list(params.items())[1:]:
                if name in kwargs:
                    continue
                if param.kind == param.POSITIONAL_OR_KEYWORD:
                    if len(args) > list(params.keys()).index(name) - 1:
                        continue
                if param.default is param.empty:
                    continue
                if param.default is not None:
                    continue
                try:
                    injected_kwargs[name] = self.provider.get(
                        name, session_id=self.session_id, request_id=self.request_id
                    )
                except Exception as e:
                    logger.debug(f"Failed to inject dependency {name}: {e}")
            merged_kwargs = {**injected_kwargs, **kwargs}
            orig_init(self, *args, **merged_kwargs)

        cls.__init__ = injected_init
        return cls


def inject_dependencies(
    func_or_class: Optional[Optional[F]] = None,
    session_id: Optional[Optional[str]] = None,
    request_id: Optional[Optional[str]] = None,
) -> F:
    """
    Decorator for automatically injecting dependencies into functions or classes.

    This decorator injects dependencies into function parameters or class constructor
    parameters, using the DependencyProvider to resolve dependencies. It can be used
    with or without arguments and supports both functions and classes.

    For classes, it wraps the __init__ method to inject dependencies. For functions,
    it wraps the function itself. Dependencies are resolved by parameter name or
    type annotation.

    Args:
        func_or_class: The function or class to inject dependencies into
        session_id: Optional session ID for scoped dependencies
        request_id: Optional request ID for scoped dependencies

    Returns:
        The function or class with dependencies injected

    Examples:
        ```python
        # Basic usage with a class
        @inject_dependencies
        class MyComponent:
            def __init__(self, model=None, validator=None):
                self.model = model  # Injected from DependencyProvider
                self.validator = validator  # Injected from DependencyProvider

        # With explicit session ID
        @inject_dependencies(session_id="user_session_1")
        class MySessionComponent:
            def __init__(self, database=None):
                self.database = database  # Session-specific instance

        # With explicit request ID
        @inject_dependencies(request_id="request_123")
        class MyRequestComponent:
            def __init__(self, validator=None):
                self.validator = validator  # Request-specific instance

        # With a function
        @inject_dependencies
        def process_data(model=None, validator=None):
            # Use injected dependencies
            pass
        ```
    """
    injector = DependencyInjector(session_id=session_id, request_id=request_id)
    if func_or_class is not None:
        if inspect.isfunction(func_or_class):
            return injector.inject_function(func_or_class)
        elif inspect.isclass(func_or_class):
            return injector.inject_class(func_or_class)
        elif inspect.ismethod(func_or_class):
            return injector.inject_method(func_or_class)
        elif callable(func_or_class):
            return injector.inject_function(func_or_class)
        else:
            raise TypeError(f"Expected function or class, got {type(func_or_class)}")

    def decorator(func_or_cls: F) -> F:
        if inspect.isfunction(func_or_cls):
            return injector.inject_function(func_or_cls)
        elif inspect.isclass(func_or_cls):
            return injector.inject_class(func_or_cls)
        elif inspect.ismethod(func_or_cls):
            return injector.inject_method(func_or_cls)
        elif callable(func_or_cls):
            return injector.inject_function(func_or_cls)
        else:
            raise TypeError(f"Expected function or class, got {type(func_or_cls)}")

    return cast(F, decorator)
