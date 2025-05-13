"""
Dependency Utilities Module

This module provides utility functions for dependency management, including
functions for registering, retrieving, and managing dependencies.

## Components
- **provide_dependency**: Register a dependency with the global provider
- **provide_factory**: Register a factory function with the global provider
- **get_dependency**: Get a dependency from the global provider
- **get_dependency_by_type**: Get a dependency by type from the global provider
- **create_session_scope**: Create a session scope context manager
- **create_request_scope**: Create a request scope context manager
- **clear_dependencies**: Clear dependencies from the global provider

## Usage Examples
```python
from sifaka.core.dependency.utils import (
    provide_dependency,
    provide_factory,
    get_dependency,
    create_session_scope,
)
from sifaka.core.dependency.scopes import DependencyScope

# Register a dependency
provide_dependency("model", OpenAIModel())

# Register a factory function
provide_factory("database", lambda: Database.connect(), scope=DependencyScope.SESSION)

# Get a dependency
model = get_dependency("model")

# Use a session scope
with create_session_scope("user_1") as session:
    # Get a session-scoped dependency
    db = get_dependency("database")  # Session-specific instance
```

## Error Handling
- Raises DependencyError for missing dependencies
- Raises ConfigurationError for circular dependencies
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

from .provider import DependencyProvider
from .scopes import DependencyScope, RequestScope, SessionScope

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for return type
T = TypeVar("T")


def provide_dependency(
    name: str,
    dependency: Any,
    scope: DependencyScope = DependencyScope.SINGLETON,
    dependencies: Optional[Optional[List[str]]] = None,
) -> None:
    """
    Register a dependency with the global dependency provider.

    This function is a convenience wrapper around DependencyProvider.register()
    that uses the global singleton provider. It registers a dependency instance
    with the provider, making it available for injection.

    Args:
        name: The dependency name used for lookup
        dependency: The dependency instance to register
        scope: The dependency scope that controls its lifecycle
        dependencies: List of dependency names this dependency depends on

    Raises:
        ConfigurationError: If registering would create a circular dependency
        DependencyError: If dependencies are invalid

    Example:
        ```python
        from sifaka.core.dependency.utils import provide_dependency
        from sifaka.core.dependency.scopes import DependencyScope

        # Register a singleton dependency
        provide_dependency("model", OpenAIModel())

        # Register a request-scoped dependency
        provide_dependency(
            "validator",
            LengthValidator(),
            scope=DependencyScope.REQUEST
        )
        ```
    """
    provider = DependencyProvider()
    provider.register(name, dependency, scope, dependencies)


def provide_factory(
    name: str,
    factory: Callable[[], Any],
    scope: DependencyScope = DependencyScope.SINGLETON,
    dependencies: Optional[Optional[List[str]]] = None,
) -> None:
    """
    Register a factory function with the global dependency provider.

    This function is a convenience wrapper around DependencyProvider.register_factory()
    that uses the global singleton provider. It registers a factory function with
    the provider, which will be called lazily when the dependency is first requested.

    Args:
        name: The dependency name used for lookup
        factory: The factory function that creates the dependency
        scope: The dependency scope that controls its lifecycle
        dependencies: List of dependency names this factory depends on

    Raises:
        ConfigurationError: If registering would create a circular dependency
        DependencyError: If factory or dependencies are invalid

    Example:
        ```python
        from sifaka.core.dependency.utils import provide_factory
        from sifaka.core.dependency.scopes import DependencyScope

        # Register a factory for database connection
        provide_factory(
            "database",
            lambda: Database.connect(config.DB_URL),
            scope=DependencyScope.SESSION
        )

        # Register a factory with dependencies
        provide_factory(
            "user_service",
            lambda: UserService(),
            scope=DependencyScope.REQUEST,
            dependencies=["database", "auth_service"]
        )
        ```
    """
    provider = DependencyProvider()
    provider.register_factory(name, factory, scope, dependencies)


def get_dependency(
    name: str,
    default: Optional[Any] = None,
    session_id: Optional[Optional[str]] = None,
    request_id: Optional[Optional[str]] = None,
) -> Any:
    """
    Get a dependency from the global dependency provider.

    This function is a convenience wrapper around DependencyProvider.get()
    that uses the global singleton provider. It retrieves a dependency by name,
    creating it if necessary using a registered factory function.

    Args:
        name: The dependency name to look up
        default: Default value if dependency not found
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies

    Returns:
        The dependency instance or default value

    Raises:
        DependencyError: If dependency not found and no default provided

    Example:
        ```python
        from sifaka.core.dependency.utils import get_dependency

        # Get a dependency
        model = get_dependency("model")

        # Get a dependency with default
        validator = get_dependency("validator", default=DefaultValidator())

        # Get a session-scoped dependency
        db = get_dependency("database", session_id="user_1")

        # Get a request-scoped dependency
        auth = get_dependency(
            "auth_service",
            session_id="user_1",
            request_id="request_123"
        )
        ```
    """
    provider = DependencyProvider()
    return provider.get(name, default, session_id, request_id)


def get_dependency_by_type(
    dependency_type: Type[T],
    default: Optional[Optional[T]] = None,
    session_id: Optional[Optional[str]] = None,
    request_id: Optional[Optional[str]] = None,
) -> T:
    """
    Get a dependency by type from the global dependency provider.

    This function retrieves a dependency by type, looking for a dependency
    with a name that matches the type name. It is a convenience wrapper
    around get_dependency() that uses the type name as the dependency name.

    Args:
        dependency_type: The type of dependency to look up
        default: Default value if dependency not found
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies

    Returns:
        The dependency instance or default value

    Raises:
        DependencyError: If dependency not found and no default provided

    Example:
        ```python
        from sifaka.core.dependency.utils import get_dependency_by_type

        # Get a dependency by type
        model = get_dependency_by_type(OpenAIModel)

        # Get a dependency by type with default
        validator = get_dependency_by_type(LengthValidator, default=DefaultValidator())

        # Get a session-scoped dependency by type
        db = get_dependency_by_type(Database, session_id="user_1")
        ```
    """
    # Get dependency name from type
    name = dependency_type.__name__

    # Get dependency
    dependency = get_dependency(name, default, session_id, request_id)

    # Validate dependency type
    if dependency is not None and not isinstance(dependency, dependency_type):
        logger.warning(f"Dependency {name} is not an instance of {dependency_type.__name__}")

    return cast(T, dependency)


def create_session_scope(session_id: Optional[Optional[str]] = None) -> SessionScope:
    """
    Create a session scope context manager.

    This function is a convenience wrapper around DependencyProvider.session_scope()
    that uses the global singleton provider. It creates a context manager for
    session-scoped dependencies.

    Args:
        session_id: Optional session ID (generated if not provided)

    Returns:
        A context manager for session-scoped dependencies

    Example:
        ```python
        from sifaka.core.dependency.utils import create_session_scope

        # Create a session scope
        with create_session_scope("user_1") as session:
            # Get session-scoped dependency
            db = get_dependency("database")  # Session-specific instance
        ```
    """
    provider = DependencyProvider()
    return provider.session_scope(session_id)


def create_request_scope(request_id: Optional[Optional[str]] = None) -> RequestScope:
    """
    Create a request scope context manager.

    This function is a convenience wrapper around DependencyProvider.request_scope()
    that uses the global singleton provider. It creates a context manager for
    request-scoped dependencies.

    Args:
        request_id: Optional request ID (generated if not provided)

    Returns:
        A context manager for request-scoped dependencies

    Example:
        ```python
        from sifaka.core.dependency.utils import create_request_scope

        # Create a request scope
        with create_request_scope("request_123") as request:
            # Get request-scoped dependency
            validator = get_dependency("validator")  # Request-specific instance
        ```
    """
    provider = DependencyProvider()
    return provider.request_scope(request_id)


def clear_dependencies(
    session_id: Optional[Optional[str]] = None,
    request_id: Optional[Optional[str]] = None,
    clear_singletons: bool = False,
) -> None:
    """
    Clear dependencies from the global dependency provider.

    This function is a convenience wrapper around DependencyProvider.clear_dependencies()
    that uses the global singleton provider. It clears dependencies from the provider,
    optionally limited to a specific session or request.

    Args:
        session_id: Optional session ID to clear
        request_id: Optional request ID to clear
        clear_singletons: Whether to clear singleton dependencies

    Example:
        ```python
        from sifaka.core.dependency.utils import clear_dependencies

        # Clear all dependencies
        clear_dependencies(clear_singletons=True)

        # Clear session-scoped dependencies
        clear_dependencies(session_id="user_1")

        # Clear request-scoped dependencies
        clear_dependencies(session_id="user_1", request_id="request_123")
        ```
    """
    provider = DependencyProvider()
    provider.clear_dependencies(session_id, request_id, clear_singletons)
