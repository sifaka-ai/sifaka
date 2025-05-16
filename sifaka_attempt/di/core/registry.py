"""
Registry for dependency management.

The registry is responsible for storing dependency registrations,
including factory functions, scopes, and dependency relationships.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, cast

from sifaka.di.core.protocols import DependencyScope, RegistryProtocol
from sifaka.di.errors import (
    CircularDependencyError,
    DependencyAlreadyRegisteredError,
    DependencyConfigurationError,
    TypeAlreadyRegisteredError,
)
from sifaka.di.utils import build_dependency_graph, detect_cycles, find_cycle_containing

T = TypeVar("T")

logger = logging.getLogger(__name__)


class DependencyRegistry(RegistryProtocol):
    """Registry for dependency management."""

    def __init__(self):
        """Initialize a new dependency registry."""
        self._dependencies: Dict[str, Any] = {}
        self._factory_functions: Dict[str, Callable[[], Any]] = {}
        self._deferred_factories: Dict[str, Callable[[], Any]] = {}
        self._scopes: Dict[str, DependencyScope] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
        self._type_registry: Dict[Type, str] = {}
        self._type_factory_registry: Dict[Type, Callable[[], Any]] = {}
        self._type_scopes: Dict[Type, DependencyScope] = {}

    def register(
        self,
        name: str,
        dependency: Any,
        scope: DependencyScope = DependencyScope.SINGLETON,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Register a dependency.

        Args:
            name: The dependency name
            dependency: The dependency instance
            scope: The dependency scope
            dependencies: List of dependencies this dependency depends on

        Raises:
            DependencyAlreadyRegisteredError: If a dependency with the same name already exists
            CircularDependencyError: If registering this dependency would create a circular dependency
        """
        if (
            name in self._dependencies
            or name in self._factory_functions
            or name in self._deferred_factories
        ):
            raise DependencyAlreadyRegisteredError(name)

        self._dependencies[name] = dependency
        self._scopes[name] = scope
        self._register_dependencies(name, dependencies or [])

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        scope: DependencyScope = DependencyScope.SINGLETON,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Register a factory function.

        Args:
            name: The dependency name
            factory: The factory function
            scope: The dependency scope
            dependencies: List of dependencies this factory depends on

        Raises:
            DependencyAlreadyRegisteredError: If a dependency with the same name already exists
            CircularDependencyError: If registering this dependency would create a circular dependency
        """
        if (
            name in self._dependencies
            or name in self._factory_functions
            or name in self._deferred_factories
        ):
            raise DependencyAlreadyRegisteredError(name)

        self._factory_functions[name] = factory
        self._scopes[name] = scope
        self._register_dependencies(name, dependencies or [])

    def register_type(
        self,
        type_: Type[T],
        instance: T,
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a dependency by type.

        Args:
            type_: The dependency type
            instance: The dependency instance
            scope: The dependency scope

        Raises:
            TypeAlreadyRegisteredError: If a dependency with the same type already exists
        """
        if type_ in self._type_registry or type_ in self._type_factory_registry:
            raise TypeAlreadyRegisteredError(type_)

        # Generate a unique name for the type
        name = f"{type_.__module__}.{type_.__name__}"
        self._dependencies[name] = instance
        self._scopes[name] = scope
        self._type_registry[type_] = name
        self._type_scopes[type_] = scope

    def register_type_factory(
        self,
        type_: Type[T],
        factory: Callable[[], T],
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a factory function by type.

        Args:
            type_: The dependency type
            factory: The factory function
            scope: The dependency scope

        Raises:
            TypeAlreadyRegisteredError: If a dependency with the same type already exists
        """
        if type_ in self._type_registry or type_ in self._type_factory_registry:
            raise TypeAlreadyRegisteredError(type_)

        # Generate a unique name for the type
        name = f"{type_.__module__}.{type_.__name__}"
        self._type_factory_registry[type_] = factory
        self._factory_functions[name] = factory
        self._scopes[name] = scope
        self._type_registry[type_] = name
        self._type_scopes[type_] = scope

    def register_deferred(
        self,
        name: str,
        factory: Callable[[], Any],
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a deferred dependency.

        Register a factory function that will be called only when the dependency
        is requested, helping to break circular dependencies.

        Args:
            name: The dependency name
            factory: The factory function
            scope: The dependency scope

        Raises:
            DependencyAlreadyRegisteredError: If a dependency with the same name already exists
        """
        if (
            name in self._dependencies
            or name in self._factory_functions
            or name in self._deferred_factories
        ):
            raise DependencyAlreadyRegisteredError(name)

        self._deferred_factories[name] = factory
        self._scopes[name] = scope

    def has_dependency(self, name: str) -> bool:
        """
        Check if a dependency is registered.

        Args:
            name: The dependency name

        Returns:
            True if the dependency is registered, False otherwise
        """
        return (
            name in self._dependencies
            or name in self._factory_functions
            or name in self._deferred_factories
        )

    def has_type(self, type_: Type[T]) -> bool:
        """
        Check if a type is registered.

        Args:
            type_: The dependency type

        Returns:
            True if the type is registered, False otherwise
        """
        return type_ in self._type_registry or type_ in self._type_factory_registry

    def get_scope(self, name: str) -> Optional[DependencyScope]:
        """
        Get the scope of a dependency.

        Args:
            name: The dependency name

        Returns:
            The dependency scope or None if the dependency is not registered
        """
        return self._scopes.get(name)

    def get_dependency(self, name: str) -> Optional[Any]:
        """
        Get a dependency by name.

        This method retrieves the registered dependency instance.
        It does not instantiate factory functions.

        Args:
            name: The dependency name

        Returns:
            The dependency instance or None if the dependency is not found
        """
        return self._dependencies.get(name)

    def get_factory(self, name: str) -> Optional[Callable[[], Any]]:
        """
        Get a factory function by name.

        Args:
            name: The dependency name

        Returns:
            The factory function or None if the factory is not found
        """
        return self._factory_functions.get(name) or self._deferred_factories.get(name)

    def get_type_name(self, type_: Type[T]) -> Optional[str]:
        """
        Get the name of a registered type.

        Args:
            type_: The dependency type

        Returns:
            The dependency name or None if the type is not registered
        """
        return self._type_registry.get(type_)

    def get_type_factory(self, type_: Type[T]) -> Optional[Callable[[], T]]:
        """
        Get a factory function by type.

        Args:
            type_: The dependency type

        Returns:
            The factory function or None if the factory is not found
        """
        return self._type_factory_registry.get(type_)

    def get_dependency_by_type(self, type_: Type[T]) -> Optional[T]:
        """
        Get a dependency by type.

        This method retrieves the registered dependency instance.
        It does not instantiate factory functions.

        Args:
            type_: The dependency type

        Returns:
            The dependency instance or None if the dependency is not found
        """
        name = self._type_registry.get(type_)
        if name:
            return cast(T, self._dependencies.get(name))
        return None

    def get_type_scope(self, type_: Type[T]) -> Optional[DependencyScope]:
        """
        Get the scope of a type.

        Args:
            type_: The dependency type

        Returns:
            The dependency scope or None if the type is not registered
        """
        return self._type_scopes.get(type_)

    def get_all_dependencies(self) -> Dict[str, Any]:
        """
        Get all registered dependencies.

        Returns:
            Dictionary of all registered dependencies by name
        """
        return self._dependencies.copy()

    def get_all_factories(self) -> Dict[str, Callable[[], Any]]:
        """
        Get all registered factory functions.

        Returns:
            Dictionary of all registered factory functions by name
        """
        return {**self._factory_functions.copy(), **self._deferred_factories.copy()}

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph.

        Returns:
            Dictionary mapping dependency names to their dependencies
        """
        return self._dependency_graph.copy()

    def get_all_types(self) -> Dict[Type, str]:
        """
        Get all registered types.

        Returns:
            Dictionary mapping types to their names
        """
        return self._type_registry.copy()

    def clear(self) -> None:
        """
        Clear all dependencies from the registry.
        """
        self._dependencies.clear()
        self._factory_functions.clear()
        self._deferred_factories.clear()
        self._scopes.clear()
        self._dependency_graph.clear()
        self._type_registry.clear()
        self._type_factory_registry.clear()
        self._type_scopes.clear()

    def _register_dependencies(self, name: str, dependencies: List[str]) -> None:
        """
        Register dependencies for a component and check for circular dependencies.

        Args:
            name: The dependency name
            dependencies: List of dependencies this dependency depends on

        Raises:
            CircularDependencyError: If registering this dependency would create a circular dependency
            DependencyConfigurationError: If a dependency in the list is not registered
        """
        for dependency in dependencies:
            if not self.has_dependency(dependency):
                logger.warning(f"Dependency '{dependency}' is not registered")

        self._dependency_graph[name] = dependencies.copy()
        graph = build_dependency_graph(self._dependency_graph)
        cycle = find_cycle_containing(graph, name)

        if cycle:
            # Remove the dependency from the graph before raising the error
            self._dependency_graph.pop(name, None)
            raise CircularDependencyError(cycle)
