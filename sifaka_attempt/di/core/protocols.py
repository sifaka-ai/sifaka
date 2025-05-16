"""
Protocol definitions for the dependency injection system.

This module defines the core interfaces for the dependency injection system,
using Protocol classes to establish clear contracts between components.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Type, runtime_checkable
from enum import Enum, auto

# Type variable for generic type annotations
T = TypeVar("T")
R = TypeVar("R")


class DependencyScope(str, Enum):
    """Enumeration of dependency scopes."""

    SINGLETON = "singleton"
    """One instance per application."""

    SESSION = "session"
    """One instance per session."""

    REQUEST = "request"
    """One instance per request."""

    TRANSIENT = "transient"
    """New instance each time."""


class ResolutionContext:
    """
    Context for dependency resolution.

    Provides contextual information for dependency resolution, such as
    session and request IDs for scoped dependencies.
    """

    def __init__(self, session_id: Optional[str] = None, request_id: Optional[str] = None):
        """
        Initialize a resolution context.

        Args:
            session_id: Optional session ID for scoped dependencies
            request_id: Optional request ID for scoped dependencies
        """
        self.session_id = session_id
        self.request_id = request_id


class DependencyHealthStatus:
    """
    Health status for a dependency.

    Represents the health status of a dependency, including whether it's
    healthy and any error information.
    """

    def __init__(
        self, healthy: bool, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a dependency health status.

        Args:
            healthy: Whether the dependency is healthy
            error: Optional error message if the dependency is unhealthy
            metadata: Optional metadata about the dependency
        """
        self.healthy = healthy
        self.error = error
        self.metadata = metadata or {}


@runtime_checkable
class ScopeContextProtocol(Protocol):
    """Protocol for scope contexts."""

    def __enter__(self) -> Any:
        """Enter the scope context."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the scope context."""
        ...

    def clear(self) -> "ScopeContextProtocol":
        """Clear dependencies on exit."""
        ...


@runtime_checkable
class RegistryProtocol(Protocol):
    """Protocol for dependency registries."""

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
        """
        ...

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
        """
        ...

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
        ...

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
        """
        ...

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
        ...

    def has_dependency(self, name: str) -> bool:
        """
        Check if a dependency is registered.

        Args:
            name: The dependency name

        Returns:
            True if the dependency is registered, False otherwise
        """
        ...

    def has_type(self, type_: Type[T]) -> bool:
        """
        Check if a type is registered.

        Args:
            type_: The dependency type

        Returns:
            True if the type is registered, False otherwise
        """
        ...

    def get_scope(self, name: str) -> Optional[DependencyScope]:
        """
        Get the scope of a dependency.

        Args:
            name: The dependency name

        Returns:
            The dependency scope or None if the dependency is not registered
        """
        ...


@runtime_checkable
class ResolverProtocol(Protocol):
    """Protocol for dependency resolvers."""

    def resolve(
        self, name: str, default: Optional[Any] = None, context: Optional[ResolutionContext] = None
    ) -> Any:
        """
        Resolve a dependency by name.

        Args:
            name: The dependency name
            default: Default value if dependency not found
            context: Optional resolution context

        Returns:
            The resolved dependency or default value
        """
        ...

    def resolve_type(
        self,
        type_: Type[T],
        default: Optional[T] = None,
        context: Optional[ResolutionContext] = None,
    ) -> T:
        """
        Resolve a dependency by type.

        Args:
            type_: The dependency type
            default: Default value if dependency not found
            context: Optional resolution context

        Returns:
            The resolved dependency or default value
        """
        ...

    def resolve_all(
        self, names: List[str], context: Optional[ResolutionContext] = None
    ) -> Dict[str, Any]:
        """
        Resolve multiple dependencies by name.

        Args:
            names: List of dependency names to resolve
            context: Optional resolution context

        Returns:
            Dictionary of resolved dependencies by name
        """
        ...

    def resolve_all_types(
        self, types: List[Type], context: Optional[ResolutionContext] = None
    ) -> Dict[Type, Any]:
        """
        Resolve multiple dependencies by type.

        Args:
            types: List of dependency types to resolve
            context: Optional resolution context

        Returns:
            Dictionary of resolved dependencies by type
        """
        ...

    def check_health(
        self, name: str, context: Optional[ResolutionContext] = None
    ) -> DependencyHealthStatus:
        """
        Check the health of a dependency.

        Args:
            name: The dependency name
            context: Optional resolution context

        Returns:
            Health status of the dependency
        """
        ...

    def check_all_health(
        self, context: Optional[ResolutionContext] = None
    ) -> Dict[str, DependencyHealthStatus]:
        """
        Check the health of all dependencies.

        Args:
            context: Optional resolution context

        Returns:
            Dictionary of health statuses by dependency name
        """
        ...


@runtime_checkable
class ScopeManagerProtocol(Protocol):
    """Protocol for scope managers."""

    def create_scope(
        self, scope_type: DependencyScope, scope_id: Optional[str] = None
    ) -> ScopeContextProtocol:
        """
        Create a scope context.

        Args:
            scope_type: The type of scope to create
            scope_id: Optional scope ID

        Returns:
            A scope context manager
        """
        ...

    def session_scope(self, session_id: Optional[str] = None) -> ScopeContextProtocol:
        """
        Create a session scope context.

        Args:
            session_id: Optional session ID

        Returns:
            A session scope context manager
        """
        ...

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
        ...

    def clear_scope(self, scope_type: DependencyScope, scope_id: Optional[str] = None) -> None:
        """
        Clear dependencies in a scope.

        Args:
            scope_type: The type of scope to clear
            scope_id: Optional scope ID
        """
        ...

    def clear_all_scopes(self) -> None:
        """Clear all dependencies in all scopes."""
        ...


@runtime_checkable
class DependencyContainerProtocol(Protocol):
    """Protocol for dependency containers."""

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
        ...

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
        ...

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
        ...

    def register_deferred(
        self,
        name: str,
        factory: Callable[[], Any],
        scope: DependencyScope = DependencyScope.SINGLETON,
    ) -> None:
        """
        Register a deferred dependency.

        Args:
            name: The dependency name
            factory: The factory function
            scope: The dependency scope
        """
        ...

    def resolve(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Resolve a dependency by name.

        Args:
            name: The dependency name
            default: Default value if dependency not found

        Returns:
            The resolved dependency or default value
        """
        ...

    def resolve_type(self, type_: Type[T], default: Optional[T] = None) -> T:
        """
        Resolve a dependency by type.

        Args:
            type_: The dependency type
            default: Default value if dependency not found

        Returns:
            The resolved dependency or default value
        """
        ...

    def session_scope(self, session_id: Optional[str] = None) -> ScopeContextProtocol:
        """
        Create a session scope context.

        Args:
            session_id: Optional session ID

        Returns:
            A session scope context manager
        """
        ...

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
        ...

    def verify_dependencies(self, names: List[str]) -> bool:
        """
        Verify that dependencies are registered.

        Args:
            names: List of dependency names to verify

        Returns:
            True if all dependencies are registered, False otherwise
        """
        ...

    def verify_dependency_graph(self) -> bool:
        """
        Verify that the dependency graph is valid.

        Returns:
            True if the dependency graph is valid, False otherwise
        """
        ...

    def check_health(self, name: str) -> DependencyHealthStatus:
        """
        Check the health of a dependency.

        Args:
            name: The dependency name

        Returns:
            Health status of the dependency
        """
        ...

    def check_all_health(self) -> Dict[str, DependencyHealthStatus]:
        """
        Check the health of all dependencies.

        Returns:
            Dictionary of health statuses by dependency name
        """
        ...
