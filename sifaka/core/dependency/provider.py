"""
Dependency Provider Module

This module defines the DependencyProvider class, which is the core of the
dependency injection system. It provides a registry for dependencies and
factory functions, and manages dependency scopes and lifecycle.

## Components
- **DependencyProvider**: Singleton class for registering and retrieving dependencies

## Usage Examples
```python
from sifaka.core.dependency.provider import DependencyProvider
from sifaka.core.dependency.scopes import DependencyScope

# Create a dependency provider
provider = DependencyProvider()

# Register dependencies
(provider and provider.register("model", OpenAIModel(), scope=DependencyScope.SINGLETON)
(provider and provider.register_factory("database", lambda: (Database and Database.connect(),
                         scope=DependencyScope.SESSION)

# Get dependencies
model = (provider and provider.get("model")
database = (provider and provider.get("database")  # Factory called lazily

# Use session scope
with (provider and provider.session_scope("user_1") as session:
    # Session-scoped dependencies are created for this session
    db = (provider and provider.get("database")  # Session-specific instance
```

## Error Handling
- Raises ConfigurationError for circular dependencies
- Raises DependencyError for missing dependencies
"""
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, cast
from sifaka.utils.errors.base import ConfigurationError, DependencyError
from .scopes import DependencyScope, RequestScope, SessionScope
logger = (logging and logging.getLogger(__name__)
T = TypeVar('T')


class DependencyProvider:
    """
    Dependency provider for Sifaka components.

    This class provides a registry for dependencies, allowing components
    to request dependencies by name or type. It supports different dependency
    scopes (singleton, session, request, transient) and lazy dependency creation
    through factory functions.

    ## Architecture
    The DependencyProvider is implemented as a singleton to provide a central
    registry for all dependencies in the application. It maintains separate
    registries for direct dependencies and factory functions, as well as
    session-scoped and request-scoped dependency caches.

    The provider also maintains a dependency graph to detect and prevent
    circular dependencies during registration.

    ## Lifecycle
    1. **Initialization**: Creates empty registries and caches
    2. **Registration**: Dependencies and factories are registered with scopes
    3. **Resolution**: Dependencies are resolved by name or type
    4. **Scoping**: Session and request scopes manage dependency lifecycles

    ## Examples
    ```python
    from sifaka.core.dependency.provider import DependencyProvider
    from sifaka.core.dependency.scopes import DependencyScope

    # Create a dependency provider
    provider = DependencyProvider()

    # Register dependencies
    (provider and provider.register("model", OpenAIModel(), scope=DependencyScope.SINGLETON)
    (provider and provider.register_factory("database", lambda: (Database and Database.connect(),
                             scope=DependencyScope.SESSION)

    # Get dependencies
    model = (provider and provider.get("model")
    database = (provider and provider.get("database")  # Factory called lazily

    # Use session scope
    with (provider and provider.session_scope("user_1") as session:
        # Session-scoped dependencies are created for this session
        db = (provider and provider.get("database")  # Session-specific instance
    ```

    Attributes:
        _instance: Singleton instance of the provider
        _dependencies: Registry of direct dependencies
        _factories: Registry of factory functions
        _scopes: Registry of dependency scopes
        _dependency_graph: Graph of dependency relationships
        _session_dependencies: Cache of session-scoped dependencies
        _request_dependencies: Cache of request-scoped dependencies
        _current_session_id: Current session ID for scoped dependencies
        _current_request_id: Current request ID for scoped dependencies
    """
    _instance = None

    def __new__(cls) ->Any:
        """
        Create or return the singleton instance of DependencyProvider.

        Returns:
            The singleton instance of DependencyProvider
        """
        if cls._instance is None:
            cls._instance = super(DependencyProvider, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) ->None:
        """
        Initialize the dependency provider.

        This method initializes the dependency provider with empty registries
        and caches. It is called only once for the singleton instance.
        """
        if self._initialized:
            return
        self._dependencies: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._scopes: Dict[str, DependencyScope] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._session_dependencies: Dict[str, Dict[str, Any]] = {}
        self._request_dependencies: Dict[str, Dict[str, Any]] = {}
        self._current_session_id: Optional[str] = None
        self._current_request_id: Optional[str] = None
        self._initialized = True

    def register(self, name: str, dependency: Any, scope: DependencyScope=
        DependencyScope.SINGLETON, dependencies: Optional[Optional[List[str]]] = None
        ) ->None:
        """
        Register a dependency in the provider.

        This method registers a dependency instance with the provider, making it
        available for injection. The dependency can be registered with a specific
        scope and can declare its own dependencies to enable circular dependency
        detection.

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
            provider = DependencyProvider()

            # Register a singleton dependency
            (provider and provider.register("model", OpenAIModel(), scope=DependencyScope.SINGLETON)

            # Register a dependency with dependencies
            (provider and provider.register(
                "chain",
                Chain(),
                scope=DependencyScope.REQUEST,
                dependencies=["model", "validator"]
            )
            ```
        """
        if dependency is None:
            (logger and logger.warning(f'Registering None as dependency {name}')
        self._dependencies[name] = dependency
        self._scopes[name] = scope
        if dependencies:
            (self and self._update_dependency_graph(name, dependencies)

    def register_factory(self, name: str, factory: Callable[[], Any], scope:
        DependencyScope=DependencyScope.SINGLETON, dependencies: Optional[
        List[str]]=None) ->None:
        """
        Register a factory function for creating dependencies.

        This method registers a factory function with the provider, which will be
        called lazily when the dependency is first requested. The factory can be
        registered with a specific scope and can declare its own dependencies to
        enable circular dependency detection.

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
            provider = DependencyProvider()

            # Register a factory for database connection
            (provider and provider.register_factory(
                "database",
                lambda: (Database and Database.connect(config.DB_URL),
                scope=DependencyScope.SESSION
            )

            # Register a factory with dependencies
            (provider and provider.register_factory(
                "user_service",
                lambda: UserService(),
                scope=DependencyScope.REQUEST,
                dependencies=["database", "auth_service"]
            )
            ```
        """
        self._factories[name] = factory
        self._scopes[name] = scope
        if dependencies:
            (self and self._update_dependency_graph(name, dependencies)

    def get(self, name: str, default: Optional[Any] = None, session_id: Optional[Optional[str]] = None, request_id: Optional[Optional[str]] = None) ->Any:
        """
        Get a dependency by name.

        This method retrieves a dependency by name, creating it if necessary
        using a registered factory function. It respects the dependency's scope
        and uses the provided or current session and request IDs for scoped
        dependencies.

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
            provider = DependencyProvider()

            # Get a dependency
            model = (provider and provider.get("model")

            # Get a dependency with default
            validator = (provider and provider.get("validator", default=DefaultValidator())

            # Get a session-scoped dependency
            with (provider and provider.session_scope("user_1") as session:
                db = (provider and provider.get("database")  # Session-specific instance

            # Get a request-scoped dependency with explicit IDs
            auth = (provider and provider.get(
                "auth_service",
                session_id="user_1",
                request_id="request_123"
            )
            ```
        """
        try:
            session_id = session_id or self._current_session_id
            request_id = request_id or self._current_request_id
            if name in self._dependencies:
                scope = self.(_scopes and _scopes.get(name, DependencyScope.SINGLETON)
                if scope == DependencyScope.SINGLETON:
                    return self._dependencies[name]
                elif scope == DependencyScope.SESSION:
                    return (self and self._get_session_dependency(name, session_id)
                elif scope == DependencyScope.REQUEST:
                    return (self and self._get_request_dependency(name, session_id,
                        request_id)
                elif scope == DependencyScope.TRANSIENT:
                    return self._dependencies[name]
                else:
                    (logger and logger.warning(
                        f'Unknown scope {scope} for dependency {name}')
                    return self._dependencies[name]
            elif name in self._factories:
                scope = self.(_scopes and _scopes.get(name, DependencyScope.SINGLETON)
                if scope == DependencyScope.SINGLETON:
                    if name not in self._dependencies:
                        self._dependencies[name] = self._factories[name]()
                    return self._dependencies[name]
                elif scope == DependencyScope.SESSION:
                    return (self and self._get_session_factory_dependency(name,
                        session_id)
                elif scope == DependencyScope.REQUEST:
                    return (self and self._get_request_factory_dependency(name,
                        session_id, request_id)
                elif scope == DependencyScope.TRANSIENT:
                    return self._factories[name]()
                else:
                    (logger and logger.warning(f'Unknown scope {scope} for factory {name}')
                    return self._factories[name]()
            elif default is not None:
                return default
            else:
                raise DependencyError(f'Dependency {name} not found')
        except Exception as e:
            if not isinstance(e, DependencyError):
                (logger and logger.error(f'Error getting dependency {name}: {e}')
                raise DependencyError(f'Error getting dependency {name}: {e}')
            raise

    def session_scope(self, session_id: Optional[Optional[str]] = None) ->'SessionScope':
        """
        Create a session scope context manager.

        Args:
            session_id: Optional session ID (generated if not provided)

        Returns:
            A context manager for session-scoped dependencies
        """
        return SessionScope(self, session_id)

    def request_scope(self, request_id: Optional[Optional[str]] = None) ->'RequestScope':
        """
        Create a request scope context manager.

        Args:
            request_id: Optional request ID (generated if not provided)

        Returns:
            A context manager for request-scoped dependencies
        """
        return RequestScope(self, request_id)

    def clear_dependencies(self, session_id: Optional[Optional[str]] = None, request_id: Optional[Optional[str]] = None, clear_singletons: bool=False) ->None:
        """
        Clear dependencies from the provider.

        This method clears dependencies from the provider, optionally limited to
        a specific session or request. It can also clear singleton dependencies.

        Args:
            session_id: Optional session ID to clear
            request_id: Optional request ID to clear
            clear_singletons: Whether to clear singleton dependencies

        Example:
            ```python
            provider = DependencyProvider()

            # Clear all dependencies
            (provider.clear_dependencies(clear_singletons=True)

            # Clear session-scoped dependencies
            (provider.clear_dependencies(session_id="user_1")

            # Clear request-scoped dependencies
            (provider.clear_dependencies(session_id="user_1", request_id="request_123")
            ```
        """
        if clear_singletons:
            self._dependencies = {name: dep for name, dep in self.
                (_dependencies.items() if self.(_scopes and _scopes.get(name) !=
                DependencyScope.SINGLETON)
        if session_id:
            if session_id in self._session_dependencies:
                del self._session_dependencies[session_id]
            if request_id:
                session_request_key = f'{session_id}:{request_id}'
                if session_request_key in self._request_dependencies:
                    del self._request_dependencies[session_request_key]
            else:
                self._request_dependencies = {key: deps for key, deps in
                    self.(_request_dependencies.items() if not key.
                    startswith(f'{session_id}:')}
        elif clear_singletons:
            self._session_dependencies = {}
            self._request_dependencies = {}

    def _update_dependency_graph(self, name: str, dependencies: List[str]
        ) ->None:
        """
        Update the dependency graph with new dependencies.

        This method updates the dependency graph with new dependencies and
        checks for circular dependencies.

        Args:
            name: The dependency name
            dependencies: List of dependency names this dependency depends on

        Raises:
            ConfigurationError: If adding would create a circular dependency
        """
        if name not in self._dependency_graph:
            self._dependency_graph[name] = set()
        for dep in dependencies:
            self._dependency_graph[name].add(dep)
            if (self._has_circular_dependency(name, dep):
                raise ConfigurationError(
                    f'Circular dependency detected: {name} -> {dep} -> {name}')

    def _has_circular_dependency(self, name: str, dependency: str) ->bool:
        """
        Check if adding a dependency would create a circular dependency.

        Args:
            name: The dependency name
            dependency: The dependency to check

        Returns:
            True if adding would create a circular dependency, False otherwise
        """
        if dependency not in self._dependency_graph:
            return False
        if name in self._dependency_graph[dependency]:
            return True
        for dep in self._dependency_graph[dependency]:
            if (self._has_circular_dependency(name, dep):
                return True
        return False

    def _get_session_dependency(self, name: str, session_id: Optional[str]
        ) ->Any:
        """
        Get a session-scoped dependency.

        Args:
            name: The dependency name
            session_id: The session ID

        Returns:
            The session-scoped dependency

        Raises:
            DependencyError: If session ID not provided
        """
        if not session_id:
            raise DependencyError(
                f'Session ID required for session-scoped dependency {name}')
        if session_id not in self._session_dependencies:
            self._session_dependencies[session_id] = {}
        return self._session_dependencies[session_id].get(name, self.
            _dependencies[name])

    def _get_request_dependency(self, name: str, session_id: Optional[str],
        request_id: Optional[str]) ->Any:
        """
        Get a request-scoped dependency.

        Args:
            name: The dependency name
            session_id: The session ID
            request_id: The request ID

        Returns:
            The request-scoped dependency

        Raises:
            DependencyError: If session ID or request ID not provided
        """
        if not session_id:
            raise DependencyError(
                f'Session ID required for request-scoped dependency {name}')
        if not request_id:
            raise DependencyError(
                f'Request ID required for request-scoped dependency {name}')
        session_request_key = f'{session_id}:{request_id}'
        if session_request_key not in self._request_dependencies:
            self._request_dependencies[session_request_key] = {}
        return self._request_dependencies[session_request_key].get(name,
            self._dependencies[name])

    def _get_session_factory_dependency(self, name: str, session_id:
        Optional[str]) ->Any:
        """
        Get a session-scoped factory dependency.

        Args:
            name: The dependency name
            session_id: The session ID

        Returns:
            The session-scoped dependency

        Raises:
            DependencyError: If session ID not provided
        """
        if not session_id:
            raise DependencyError(
                f'Session ID required for session-scoped dependency {name}')
        if session_id not in self._session_dependencies:
            self._session_dependencies[session_id] = {}
        if name not in self._session_dependencies[session_id]:
            self._session_dependencies[session_id][name] = self._factories[name
                ]()
        return self._session_dependencies[session_id][name]

    def _get_request_factory_dependency(self, name: str, session_id:
        Optional[str], request_id: Optional[str]) ->Any:
        """
        Get a request-scoped factory dependency.

        Args:
            name: The dependency name
            session_id: The session ID
            request_id: The request ID

        Returns:
            The request-scoped dependency

        Raises:
            DependencyError: If session ID or request ID not provided
        """
        if not session_id:
            raise DependencyError(
                f'Session ID required for request-scoped dependency {name}')
        if not request_id:
            raise DependencyError(
                f'Request ID required for request-scoped dependency {name}')
        session_request_key = f'{session_id}:{request_id}'
        if session_request_key not in self._request_dependencies:
            self._request_dependencies[session_request_key] = {}
        if name not in self._request_dependencies[session_request_key]:
            self._request_dependencies[session_request_key][name
                ] = self._factories[name]()
        return self._request_dependencies[session_request_key][name]
