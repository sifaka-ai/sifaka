"""
Dependency container for the dependency injection system.

The container is the main entry point for the dependency injection system,
combining registry, resolver, and scope manager functionality.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from sifaka.di.core.protocols import (
    DependencyContainerProtocol,
    DependencyHealthStatus,
    DependencyScope,
    ScopeContextProtocol,
)
from sifaka.di.core.registry import DependencyRegistry
from sifaka.di.core.resolver import DependencyResolver
from sifaka.di.core.scope_manager import ScopeManager
from sifaka.di.errors import CircularDependencyError, DependencyError
from sifaka.di.utils import detect_cycles

T = TypeVar("T")

logger = logging.getLogger(__name__)


class DependencyContainer(DependencyContainerProtocol):
    """
    Container for managing dependencies.

    This class combines the functionality of registry, resolver, and scope manager
    to provide a complete dependency injection solution.
    """

    _instance = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "DependencyContainer":
        """
        Get the singleton instance of the dependency container.

        Returns:
            The singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance of the dependency container.

        This method is useful for testing.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
                cls._instance = None

    def __init__(self):
        """Initialize a new dependency container."""
        self._registry = DependencyRegistry()
        self._scope_manager = ScopeManager()
        self._resolver = DependencyResolver(self._registry, self._scope_manager)

    def register(
        self, name: str, dependency: Any, scope: DependencyScope = DependencyScope.SINGLETON
    ) -> None:
        """
        Register a dependency.

        Args:
            name: The dependency name
            dependency: The dependency instance
            scope: The dependency scope
        """
        self._registry.register(name, dependency, scope)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a factory function.

        Args:
            name: The dependency name
            factory: The factory function
            scope: The dependency scope
        """
        self._registry.register_factory(name, factory, scope)

    def register_type(
        self, type_: Type[T], instance: T, scope: DependencyScope = DependencyScope.SINGLETON
    ) -> None:
        """
        Register a dependency by type.

        Args:
            type_: The dependency type
            instance: The dependency instance
            scope: The dependency scope
        """
        self._registry.register_type(type_, instance, scope)

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
        """
        self._registry.register_deferred(name, factory, scope)

    def resolve(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Resolve a dependency by name.

        Args:
            name: The dependency name
            default: Default value if dependency not found

        Returns:
            The resolved dependency or default value
        """
        return self._resolver.resolve(name, default)

    def resolve_type(self, type_: Type[T], default: Optional[T] = None) -> T:
        """
        Resolve a dependency by type.

        Args:
            type_: The dependency type
            default: Default value if dependency not found

        Returns:
            The resolved dependency or default value
        """
        return self._resolver.resolve_type(type_, default)

    def session_scope(self, session_id: Optional[str] = None) -> ScopeContextProtocol:
        """
        Create a session scope context.

        Args:
            session_id: Optional session ID

        Returns:
            A session scope context manager
        """
        return self._scope_manager.session_scope(session_id)

    def request_scope(
        self, request_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> ScopeContextProtocol:
        """
        Create a request scope context.

        Args:
            request_id: Optional request ID
            session_id: Optional session ID

        Returns:
            A request scope context manager
        """
        return self._scope_manager.request_scope(request_id, session_id)

    def verify_dependencies(self, names: List[str]) -> bool:
        """
        Verify that dependencies are registered.

        Args:
            names: List of dependency names to verify

        Returns:
            True if all dependencies are registered, False otherwise
        """
        for name in names:
            if not self._registry.has_dependency(name):
                logger.warning(f"Dependency '{name}' is not registered")
                return False
        return True

    def verify_dependency_graph(self) -> bool:
        """
        Verify that the dependency graph is valid.

        Returns:
            True if the dependency graph is valid, False otherwise
        """
        graph = self._registry.get_dependency_graph()
        cycles = detect_cycles(graph)

        if cycles:
            for cycle in cycles:
                cycle_str = " -> ".join(cycle)
                logger.warning(f"Circular dependency detected: {cycle_str}")
            return False

        return True

    def check_health(self, name: str) -> DependencyHealthStatus:
        """
        Check the health of a dependency.

        Args:
            name: The dependency name

        Returns:
            Health status of the dependency
        """
        return self._resolver.check_health(name)

    def check_all_health(self) -> Dict[str, DependencyHealthStatus]:
        """
        Check the health of all dependencies.

        Returns:
            Dictionary of health statuses by dependency name
        """
        return self._resolver.check_all_health()

    def clear(self) -> None:
        """
        Clear all dependencies.
        """
        self._registry.clear()
        self._scope_manager.clear_all_scopes()

    def has_dependency(self, name: str) -> bool:
        """
        Check if a dependency is registered.

        Args:
            name: The dependency name

        Returns:
            True if the dependency is registered, False otherwise
        """
        return self._registry.has_dependency(name)

    def has_type(self, type_: Type) -> bool:
        """
        Check if a type is registered.

        Args:
            type_: The dependency type

        Returns:
            True if the type is registered, False otherwise
        """
        return self._registry.has_type(type_)

    def register_with_dependencies(
        self,
        name: str,
        dependency: Any,
        dependencies: List[str],
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a dependency with explicit dependencies.

        Args:
            name: The dependency name
            dependency: The dependency instance
            dependencies: List of dependencies this dependency depends on
            scope: The dependency scope

        Raises:
            CircularDependencyError: If registering this dependency would create a circular dependency
        """
        try:
            self._registry.register(name, dependency, scope, dependencies)
        except CircularDependencyError as e:
            logger.error(f"Circular dependency detected when registering '{name}': {e}")
            raise

    def register_factory_with_dependencies(
        self,
        name: str,
        factory: Callable[[], Any],
        dependencies: List[str],
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a factory function with explicit dependencies.

        Args:
            name: The dependency name
            factory: The factory function
            dependencies: List of dependencies this factory depends on
            scope: The dependency scope

        Raises:
            CircularDependencyError: If registering this dependency would create a circular dependency
        """
        try:
            self._registry.register_factory(name, factory, scope, dependencies)
        except CircularDependencyError as e:
            logger.error(f"Circular dependency detected when registering factory '{name}': {e}")
            raise

    def resolve_many(self, *names: str) -> List[Any]:
        """
        Resolve multiple dependencies by name.

        Args:
            *names: Dependency names to resolve

        Returns:
            List of resolved dependencies in the same order as the names
        """
        return [self.resolve(name) for name in names]
