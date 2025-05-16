"""
Mock container for dependency injection testing.

This module provides a mock dependency container for testing,
allowing dependencies to be mocked in tests.
"""

import contextlib
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from unittest.mock import MagicMock, patch

from sifaka.di import get_container
from sifaka.di.core import DependencyContainer, DependencyScope

T = TypeVar("T")


class MockDependencyContainer:
    """
    Mock dependency container for testing.

    This class provides a mock dependency container that can be used in tests
    to mock dependencies and isolate code from its dependencies.
    """

    def __init__(self):
        """Initialize a new mock dependency container."""
        self.reset()

    def reset(self) -> None:
        """Reset the mock container to its initial state."""
        self._dependencies: Dict[str, Any] = {}
        self._types: Dict[Type, Any] = {}

    def register(
        self, name: str, dependency: Any, scope: DependencyScope = DependencyScope.SINGLETON
    ) -> None:
        """
        Register a dependency.

        Args:
            name: The dependency name
            dependency: The dependency instance
            scope: The dependency scope (ignored in mock)
        """
        self._dependencies[name] = dependency

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a factory function.

        In the mock container, the factory is immediately executed and
        the instance is stored.

        Args:
            name: The dependency name
            factory: The factory function
            scope: The dependency scope (ignored in mock)
        """
        self._dependencies[name] = factory()

    def register_type(
        self, type_: Type[T], instance: T, scope: DependencyScope = DependencyScope.SINGLETON
    ) -> None:
        """
        Register a dependency by type.

        Args:
            type_: The dependency type
            instance: The dependency instance
            scope: The dependency scope (ignored in mock)
        """
        self._types[type_] = instance

    def resolve(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Resolve a dependency by name.

        Args:
            name: The dependency name
            default: Default value if dependency not found

        Returns:
            The resolved dependency or default value
        """
        return self._dependencies.get(name, default)

    def resolve_type(self, type_: Type[T], default: Optional[T] = None) -> T:
        """
        Resolve a dependency by type.

        Args:
            type_: The dependency type
            default: Default value if dependency not found

        Returns:
            The resolved dependency or default value
        """
        return self._types.get(type_, default)

    def has_dependency(self, name: str) -> bool:
        """
        Check if a dependency is registered.

        Args:
            name: The dependency name

        Returns:
            True if the dependency is registered, False otherwise
        """
        return name in self._dependencies

    def has_type(self, type_: Type) -> bool:
        """
        Check if a type is registered.

        Args:
            type_: The dependency type

        Returns:
            True if the type is registered, False otherwise
        """
        return type_ in self._types

    def clear(self) -> None:
        """Clear all dependencies."""
        self.reset()

    def mock_dependency(self, name: str, spec_set: Optional[Any] = None, **kwargs) -> MagicMock:
        """
        Register a mock dependency.

        Args:
            name: The dependency name
            spec_set: Optional spec_set for the mock
            **kwargs: Additional arguments passed to MagicMock

        Returns:
            The mock dependency
        """
        mock = MagicMock(spec_set=spec_set, **kwargs)
        self.register(name, mock)
        return mock

    def mock_type(self, type_: Type[T], spec_set: Optional[Any] = None, **kwargs) -> MagicMock:
        """
        Register a mock dependency by type.

        Args:
            type_: The dependency type
            spec_set: Optional spec_set for the mock
            **kwargs: Additional arguments passed to MagicMock

        Returns:
            The mock dependency
        """
        mock = MagicMock(spec_set=spec_set or type_, **kwargs)
        self.register_type(type_, mock)
        return mock


@contextlib.contextmanager
def mock_container():
    """
    Context manager for mocking the dependency container.

    This context manager replaces the global dependency container
    with a mock container for testing.

    Returns:
        A mock dependency container

    Example:
        ```python
        def test_something():
            with mock_container() as container:
                container.mock_dependency("logger")
                # Test code that uses dependencies
        ```
    """
    mock_container = MockDependencyContainer()

    with patch("sifaka.di.get_container", return_value=mock_container):
        DependencyContainer.reset_instance()
        try:
            yield mock_container
        finally:
            DependencyContainer.reset_instance()


def mock_dependency(name: str, spec_set: Optional[Any] = None, **kwargs) -> MagicMock:
    """
    Register a mock dependency.

    This function is a shortcut for:
    ```python
    get_container().mock_dependency(name, spec_set, **kwargs)
    ```

    Args:
        name: The dependency name
        spec_set: Optional spec_set for the mock
        **kwargs: Additional arguments passed to MagicMock

    Returns:
        The mock dependency
    """
    container = get_container()
    if not hasattr(container, "mock_dependency"):
        raise ValueError(
            "Cannot mock dependencies on a real container. Use mock_container context manager."
        )
    return container.mock_dependency(name, spec_set, **kwargs)
