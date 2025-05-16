"""
Dependency Injection Container

This module provides a streamlined, clean dependency injection container
that addresses the core architectural issues in the previous implementation.
"""

from typing import Any, Dict, Type, TypeVar, Callable, Optional, Set, List
import inspect
import threading
from dataclasses import dataclass, field

T = TypeVar("T")


class DIError(Exception):
    """Base class for all dependency injection errors."""

    pass


class DependencyNotFoundError(DIError):
    """Error raised when a dependency is not found."""

    pass


class CircularDependencyError(DIError):
    """Error raised when a circular dependency is detected."""

    pass


@dataclass
class Dependency:
    """Container for dependency registration information."""

    instance: Any = None
    factory: Optional[Callable[[], Any]] = None
    singleton: bool = True
    dependencies: Set[str] = field(default_factory=set)
    created: bool = False
    being_created: bool = False


class Container:
    """
    Simplified dependency injection container.

    Design goals:
    1. Clean, simple API with minimal complexity
    2. No circular dependencies by design
    3. Consistent usage patterns
    4. Static analysis friendly (type hints everywhere)
    5. No global state or singletons by default
    """

    def __init__(self):
        """Initialize a new container."""
        self._dependencies: Dict[str, Dependency] = {}
        self._type_map: Dict[Type, str] = {}
        self._lock = threading.RLock()

    def register(self, name: str, instance: Any) -> None:
        """
        Register an instance with the container.

        Args:
            name: The name to register the instance under
            instance: The instance to register
        """
        with self._lock:
            self._dependencies[name] = Dependency(instance=instance, created=True)
            # Register the type for type-based resolution
            self._type_map[type(instance)] = name

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        singleton: bool = True,
        dependencies: List[str] = None,
    ) -> None:
        """
        Register a factory function with the container.

        Args:
            name: The name to register the factory under
            factory: The factory function
            singleton: Whether to cache the factory result
            dependencies: List of dependencies this factory depends on
        """
        with self._lock:
            self._dependencies[name] = Dependency(
                factory=factory, singleton=singleton, dependencies=set(dependencies or [])
            )

            # Verify no circular dependencies
            self._check_circular_dependencies(name, set())

    def register_type(self, type_: Type[T], instance: T) -> None:
        """
        Register an instance for a specific type.

        Args:
            type_: The type to register the instance for
            instance: The instance to register
        """
        with self._lock:
            # Generate a unique name for the type
            name = f"type:{type_.__module__}.{type_.__name__}"
            self.register(name, instance)
            self._type_map[type_] = name

    def register_type_factory(
        self, type_: Type[T], factory: Callable[[], T], singleton: bool = True
    ) -> None:
        """
        Register a factory function for a specific type.

        Args:
            type_: The type to register the factory for
            factory: The factory function
            singleton: Whether to cache the factory result
        """
        with self._lock:
            # Generate a unique name for the type
            name = f"type:{type_.__module__}.{type_.__name__}"
            self.register_factory(name, factory, singleton)
            self._type_map[type_] = name

    def resolve(self, name: str) -> Any:
        """
        Resolve a dependency by name.

        Args:
            name: The name of the dependency to resolve

        Returns:
            The resolved dependency

        Raises:
            DependencyNotFoundError: If the dependency is not found
            CircularDependencyError: If a circular dependency is detected
        """
        with self._lock:
            if name not in self._dependencies:
                raise DependencyNotFoundError(f"Dependency '{name}' not found")

            dep = self._dependencies[name]

            # Return existing instance if already created
            if dep.created:
                return dep.instance

            # Detect circular dependencies
            if dep.being_created:
                raise CircularDependencyError(
                    f"Circular dependency detected while resolving '{name}'"
                )

            # Create instance from factory
            if dep.factory:
                # Mark as being created to detect cycles
                dep.being_created = True

                try:
                    # Resolve dependencies first
                    for dependency_name in dep.dependencies:
                        self.resolve(dependency_name)

                    # Create the instance
                    instance = dep.factory()

                    # Cache the instance if singleton
                    if dep.singleton:
                        dep.instance = instance
                        dep.created = True

                    return instance
                finally:
                    dep.being_created = False

            raise DependencyNotFoundError(f"Dependency '{name}' has no instance or factory")

    def resolve_type(self, type_: Type[T]) -> T:
        """
        Resolve a dependency by type.

        Args:
            type_: The type of the dependency to resolve

        Returns:
            The resolved dependency

        Raises:
            DependencyNotFoundError: If the dependency is not found
        """
        with self._lock:
            if type_ not in self._type_map:
                raise DependencyNotFoundError(f"Type '{type_.__name__}' not registered")

            name = self._type_map[type_]
            return self.resolve(name)

    def clear(self) -> None:
        """Clear all dependencies."""
        with self._lock:
            self._dependencies.clear()
            self._type_map.clear()

    def _check_circular_dependencies(self, name: str, visited: Set[str]) -> None:
        """
        Check for circular dependencies.

        Args:
            name: The name of the dependency to check
            visited: Set of already visited dependencies

        Raises:
            CircularDependencyError: If a circular dependency is detected
        """
        if name in visited:
            path = " -> ".join(visited) + " -> " + name
            raise CircularDependencyError(f"Circular dependency detected: {path}")

        visited.add(name)

        dep = self._dependencies.get(name)
        if dep and dep.dependencies:
            for dependency_name in dep.dependencies:
                if dependency_name not in self._dependencies:
                    continue
                self._check_circular_dependencies(dependency_name, visited.copy())
