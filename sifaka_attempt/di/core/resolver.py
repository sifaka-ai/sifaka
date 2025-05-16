"""
Resolver for dependency injection.

The resolver is responsible for resolving dependencies
from the registry and managing their lifecycle based on scope.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

from sifaka.di.core.protocols import (
    DependencyHealthStatus,
    DependencyScope,
    RegistryProtocol,
    ResolverProtocol,
    ResolutionContext,
    ScopeManagerProtocol,
)
from sifaka.di.errors import (
    DependencyNotFoundError,
    DependencyResolutionError,
    TypeNotFoundError,
    TypeResolutionError,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class DependencyResolver(ResolverProtocol):
    """Resolver for dependencies."""

    def __init__(self, registry: RegistryProtocol, scope_manager: ScopeManagerProtocol):
        """
        Initialize a new dependency resolver.

        Args:
            registry: The dependency registry
            scope_manager: The scope manager
        """
        self._registry = registry
        self._scope_manager = scope_manager

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

        Raises:
            DependencyNotFoundError: If the dependency is not found and no default is provided
            DependencyResolutionError: If the dependency cannot be resolved
        """
        try:
            # Get dependency info from registry
            if not self._registry.has_dependency(name):
                if default is not None:
                    return default
                raise DependencyNotFoundError(name)

            scope = self._registry.get_scope(name)
            if not scope:
                if default is not None:
                    return default
                raise DependencyNotFoundError(name)

            # Handle different scopes
            session_id = context.session_id if context else None
            request_id = context.request_id if context else None

            # Try to get instance from scope
            instance = self._scope_manager.get_instance(name, scope, session_id, request_id)
            if instance is not None:
                return instance

            # Need to create the instance
            instance = self._create_instance(name)

            # Store in appropriate scope
            if scope != DependencyScope.TRANSIENT:
                self._scope_manager.set_instance(name, instance, scope, session_id, request_id)

            return instance
        except DependencyNotFoundError:
            raise
        except Exception as e:
            if default is not None:
                logger.warning(f"Failed to resolve dependency '{name}': {str(e)}, using default")
                return default
            raise DependencyResolutionError(name, e)

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

        Raises:
            TypeNotFoundError: If the type is not found and no default is provided
            TypeResolutionError: If the type cannot be resolved
        """
        try:
            # Get dependency name from registry
            name = self._registry.get_type_name(type_)
            if not name:
                if default is not None:
                    return default
                raise TypeNotFoundError(type_)

            # Resolve by name
            instance = self.resolve(name, None, context)
            return cast(T, instance)
        except TypeNotFoundError:
            raise
        except Exception as e:
            if default is not None:
                logger.warning(
                    f"Failed to resolve type '{type_.__name__}': {str(e)}, using default"
                )
                return default
            raise TypeResolutionError(type_, e)

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
        result = {}
        for name in names:
            try:
                result[name] = self.resolve(name, context=context)
            except (DependencyNotFoundError, DependencyResolutionError) as e:
                logger.warning(f"Failed to resolve dependency '{name}': {str(e)}")
        return result

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
        result = {}
        for type_ in types:
            try:
                result[type_] = self.resolve_type(type_, context=context)
            except (TypeNotFoundError, TypeResolutionError) as e:
                logger.warning(f"Failed to resolve type '{type_.__name__}': {str(e)}")
        return result

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
        try:
            # Try to resolve the dependency
            dependency = self.resolve(name, context=context)

            # Check if the dependency has a health check method
            if hasattr(dependency, "check_health") and callable(
                getattr(dependency, "check_health")
            ):
                try:
                    return dependency.check_health()
                except Exception as e:
                    return DependencyHealthStatus(
                        healthy=False,
                        error=f"Health check failed: {str(e)}",
                        metadata={"exception": str(e)},
                    )

            # If no health check method, assume it's healthy
            return DependencyHealthStatus(
                healthy=True, metadata={"note": "No health check method available"}
            )
        except (DependencyNotFoundError, DependencyResolutionError) as e:
            return DependencyHealthStatus(
                healthy=False, error=str(e), metadata={"exception": str(e)}
            )

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
        result = {}

        # Get all dependencies and factories
        dependencies = self._registry.get_all_dependencies()
        factories = self._registry.get_all_factories()

        # Check health of all dependencies
        for name in set(list(dependencies.keys()) + list(factories.keys())):
            result[name] = self.check_health(name, context)

        return result

    def _create_instance(self, name: str) -> Any:
        """
        Create a new instance of a dependency.

        Args:
            name: The dependency name

        Returns:
            The new instance

        Raises:
            DependencyResolutionError: If the dependency cannot be created
        """
        # Check for existing instance
        instance = self._registry.get_dependency(name)
        if instance is not None:
            return instance

        # Try to create from factory
        factory = self._registry.get_factory(name)
        if factory is not None:
            try:
                return factory()
            except Exception as e:
                raise DependencyResolutionError(name, e)

        raise DependencyNotFoundError(name)
