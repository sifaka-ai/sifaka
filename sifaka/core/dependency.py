"""
Dependency Injection Module

This module provides standardized dependency injection patterns for Sifaka components.
It defines base classes and utilities for dependency injection, ensuring consistent
behavior across all components.

## Overview
The dependency injection system in Sifaka provides a flexible and standardized way to manage
dependencies between components. It supports different dependency scopes, lazy dependency
creation through factory functions, and automatic dependency resolution based on parameter
names and type annotations.

## Components
- **DependencyProvider**: Singleton class for registering and retrieving dependencies
- **DependencyScope**: Enum defining dependency lifecycles (singleton, session, request, transient)
- **SessionScope**: Context manager for session-scoped dependencies
- **RequestScope**: Context manager for request-scoped dependencies
- **DependencyInjector**: Utility class for manual dependency injection
- **inject_dependencies**: Decorator for automatic dependency injection

## Usage Examples
```python
from sifaka.core.dependency import DependencyProvider, inject_dependencies, DependencyScope

# Create a dependency provider
provider = DependencyProvider()
provider.register("model", OpenAIProvider("gpt-4"), scope=DependencyScope.SINGLETON)
provider.register("validator", LengthValidator(), scope=DependencyScope.REQUEST)

# Register a factory function for lazy dependency creation
provider.register_factory("database", lambda: Database.connect(), scope=DependencyScope.SESSION)

# Create a component with injected dependencies
@inject_dependencies
class MyComponent:
    def __init__(self, model=None, validator=None, database=None):
        self.model = model
        self.validator = validator
        self.database = database

# Create an instance with dependencies injected
component = MyComponent()  # Dependencies automatically injected

# Create a session-scoped instance
with provider.session_scope("user_session_1") as session:
    # All session-scoped dependencies will be created for this session
    component = MyComponent()  # Session-scoped dependencies are injected

# Create a request-scoped instance
with provider.request_scope("request_123") as request:
    # All request-scoped dependencies will be created for this request
    component = MyComponent()  # Request-scoped dependencies are injected
```

## Error Handling
The dependency injection system raises `DependencyError` when dependencies cannot be resolved.
It also detects and prevents circular dependencies during registration.

## Configuration
Dependencies can be configured with different scopes and dependencies, allowing for
flexible component composition and lifecycle management.
"""

import functools
import inspect
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, cast

from sifaka.utils.errors import DependencyError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    # Classes
    "DependencyScope",
    "DependencyProvider",
    "SessionScope",
    "RequestScope",
    "DependencyInjector",
    # Decorators
    "inject_dependencies",
    # Functions
    "provide_dependency",
    "provide_factory",
    "get_dependency",
    "get_dependency_by_type",
    "create_session_scope",
    "create_request_scope",
    "clear_dependencies",
]

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class DependencyScope(str, Enum):
    """
    Dependency scope enum defining the lifecycle of dependencies.

    This enum defines the different scopes that control the lifecycle of dependencies
    in the Sifaka dependency injection system. Each scope determines when dependencies
    are created and how long they live.

    ## Architecture
    The scope system is hierarchical:
    - SINGLETON: Global scope (application-wide)
    - SESSION: Session scope (per user session)
    - REQUEST: Request scope (per individual request)
    - TRANSIENT: No caching (new instance each time)

    ## Lifecycle
    - SINGLETON: Created once and reused throughout the application
    - SESSION: Created once per session and reused within that session
    - REQUEST: Created once per request and reused within that request
    - TRANSIENT: Created each time the dependency is requested

    ## Examples
    ```python
    from sifaka.core.dependency import DependencyProvider, DependencyScope

    # Register dependencies with different scopes
    provider = DependencyProvider()
    provider.register("database", Database(), scope=DependencyScope.SINGLETON)
    provider.register("user_data", UserData(), scope=DependencyScope.SESSION)
    provider.register("validator", Validator(), scope=DependencyScope.REQUEST)
    provider.register("generator", Generator(), scope=DependencyScope.TRANSIENT)
    ```

    Attributes:
        SINGLETON (str): One instance per application
        SESSION (str): One instance per session
        REQUEST (str): One instance per request
        TRANSIENT (str): New instance each time
    """

    SINGLETON = "singleton"
    SESSION = "session"
    REQUEST = "request"
    TRANSIENT = "transient"


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
    1. **Initialization**: The provider is created as a singleton
    2. **Registration**: Dependencies and factories are registered with scopes
    3. **Resolution**: Dependencies are resolved by name or type when requested
    4. **Scoping**: Session and request scopes are managed through context managers
    5. **Cleanup**: Dependencies can be cleared when no longer needed

    ## Error Handling
    - Raises `DependencyError` when dependencies cannot be resolved
    - Detects and prevents circular dependencies during registration
    - Provides fallback mechanisms for missing session/request IDs

    ## Examples
    ```python
    from sifaka.core.dependency import DependencyProvider, DependencyScope

    # Get the singleton provider
    provider = DependencyProvider()

    # Register dependencies
    provider.register("model", OpenAIModel(), scope=DependencyScope.SINGLETON)
    provider.register_factory("database", lambda: Database.connect(),
                             scope=DependencyScope.SESSION)

    # Get dependencies
    model = provider.get("model")
    database = provider.get("database")  # Factory called lazily

    # Use session scope
    with provider.session_scope("user_1") as session:
        # Session-scoped dependencies are created for this session
        db = provider.get("database")  # Session-specific instance
    ```

    Attributes:
        _instance: Singleton instance of the provider
        _dependencies (Dict[str, Any]): Registry of direct dependencies
        _scopes (Dict[str, DependencyScope]): Scope of each dependency
        _factories (Dict[str, Callable[[], Any]]): Registry of factory functions
        _session_dependencies (Dict[str, Dict[str, Any]]): Session-scoped dependency cache
        _request_dependencies (Dict[str, Dict[str, Any]]): Request-scoped dependency cache
        _current_session_id (Optional[str]): Current active session ID
        _current_request_id (Optional[str]): Current active request ID
        _dependency_graph (Dict[str, Set[str]]): Dependency relationship graph
    """

    _instance = None
    _dependencies: Dict[str, Any] = {}
    _scopes: Dict[str, DependencyScope] = {}
    _factories: Dict[str, Callable[[], Any]] = {}
    _session_dependencies: Dict[str, Dict[str, Any]] = {}
    _request_dependencies: Dict[str, Dict[str, Any]] = {}
    _current_session_id: Optional[str] = None
    _current_request_id: Optional[str] = None
    _dependency_graph: Dict[str, Set[str]] = {}

    def __new__(cls) -> "DependencyProvider":
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(DependencyProvider, cls).__new__(cls)
            cls._instance._dependencies = {}
            cls._instance._scopes = {}
            cls._instance._factories = {}
            cls._instance._session_dependencies = {}
            cls._instance._request_dependencies = {}
            cls._instance._dependency_graph = {}
            cls._instance._current_session_id = None
            cls._instance._current_request_id = None
        return cls._instance

    def register(
        self,
        name: str,
        dependency: Any,
        scope: DependencyScope = DependencyScope.SINGLETON,
        dependencies: Optional[List[str]] = None,
    ) -> None:
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
            provider.register("model", OpenAIModel(), scope=DependencyScope.SINGLETON)

            # Register a dependency with dependencies
            provider.register(
                "chain",
                Chain(),
                scope=DependencyScope.REQUEST,
                dependencies=["model", "validator"]
            )
            ```
        """
        # Validate dependency
        if dependency is None:
            logger.warning(f"Registering None as dependency {name}")

        # Register dependency
        self._dependencies[name] = dependency
        self._scopes[name] = scope

        # Update dependency graph
        if dependencies:
            self._update_dependency_graph(name, dependencies)

        logger.debug(
            f"Registered dependency {name}: {dependency.__class__.__name__} "
            f"with scope {scope.value}"
        )

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        scope: DependencyScope = DependencyScope.SINGLETON,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Register a factory function for lazy dependency creation.

        This method registers a factory function that will be called to create
        the dependency when it is first requested. This enables lazy initialization
        of dependencies and allows for dependencies that are expensive to create
        or require runtime configuration.

        Args:
            name: The dependency name used for lookup
            factory: Factory function that creates the dependency when called
            scope: The dependency scope that controls its lifecycle
            dependencies: List of dependency names this dependency depends on

        Raises:
            ConfigurationError: If registering would create a circular dependency
            DependencyError: If factory or dependencies are invalid

        Example:
            ```python
            provider = DependencyProvider()

            # Register a factory for database connection
            provider.register_factory(
                "database",
                lambda: Database.connect(config.DB_URL),
                scope=DependencyScope.SESSION
            )

            # Register a factory with dependencies
            provider.register_factory(
                "user_service",
                lambda: UserService(),
                scope=DependencyScope.REQUEST,
                dependencies=["database", "auth_service"]
            )
            ```
        """
        # Register factory
        self._factories[name] = factory
        self._scopes[name] = scope

        # Update dependency graph
        if dependencies:
            self._update_dependency_graph(name, dependencies)

        logger.debug(f"Registered factory for dependency {name} with scope {scope.value}")

    def _update_dependency_graph(self, name: str, dependencies: List[str]) -> None:
        """
        Update the dependency graph and check for circular dependencies.

        Args:
            name: The dependency name
            dependencies: List of dependencies this dependency depends on

        Raises:
            ConfigurationError: If adding the dependency would create a circular dependency
        """
        # Initialize dependency set if not exists
        if name not in self._dependency_graph:
            self._dependency_graph[name] = set()

        # Add direct dependencies
        for dep in dependencies:
            self._dependency_graph[name].add(dep)

        # Check for circular dependencies
        visited = set()
        path = []

        def check_circular(node: str) -> bool:
            if node in path:
                cycle = path[path.index(node) :] + [node]
                raise DependencyError(f"Circular dependency detected: {' -> '.join(cycle)}")
            if node in visited:
                return False

            visited.add(node)
            path.append(node)

            for dep in self._dependency_graph.get(node, set()):
                if check_circular(dep):
                    return True

            path.pop()
            return False

        check_circular(name)

    def get(
        self,
        name: str,
        default: Optional[Any] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Any:
        """
        Get a dependency by name from the provider.

        This method retrieves a dependency by its registered name. It handles
        different dependency scopes appropriately, creating new instances for
        factory-registered dependencies as needed based on their scope.

        Args:
            name: The dependency name to look up
            default: Optional default value if dependency not found
            session_id: Optional session ID for session-scoped dependencies
            request_id: Optional request ID for request-scoped dependencies

        Returns:
            The dependency instance or default value if not found

        Raises:
            DependencyError: If dependency not found and no default provided

        Example:
            ```python
            provider = DependencyProvider()

            # Get a dependency
            model = provider.get("model")

            # Get a dependency with default
            validator = provider.get("validator", default=DefaultValidator())

            # Get a session-scoped dependency
            with provider.session_scope("user_1") as session:
                db = provider.get("database")  # Session-specific instance

            # Get a request-scoped dependency with explicit IDs
            auth = provider.get(
                "auth_service",
                session_id="user_1",
                request_id="request_123"
            )
            ```
        """
        try:
            # Use current session/request ID if not provided
            session_id = session_id or self._current_session_id
            request_id = request_id or self._current_request_id

            # Check if dependency exists
            if name in self._dependencies:
                # Get dependency scope
                scope = self._scopes.get(name, DependencyScope.SINGLETON)

                # Handle scoped dependencies
                if scope == DependencyScope.SINGLETON:
                    return self._dependencies[name]
                elif scope == DependencyScope.SESSION:
                    if session_id is None:
                        logger.warning(f"Session ID required for session-scoped dependency: {name}")
                        return self._dependencies[name]
                    if session_id not in self._session_dependencies:
                        self._session_dependencies[session_id] = {}
                    if name not in self._session_dependencies[session_id]:
                        self._session_dependencies[session_id][name] = self._dependencies[name]
                    return self._session_dependencies[session_id][name]
                elif scope == DependencyScope.REQUEST:
                    if request_id is None:
                        logger.warning(f"Request ID required for request-scoped dependency: {name}")
                        return self._dependencies[name]
                    if request_id not in self._request_dependencies:
                        self._request_dependencies[request_id] = {}
                    if name not in self._request_dependencies[request_id]:
                        self._request_dependencies[request_id][name] = self._dependencies[name]
                    return self._request_dependencies[request_id][name]
                elif scope == DependencyScope.TRANSIENT:
                    # For transient dependencies, create a new instance if factory exists
                    if name in self._factories:
                        return self._factories[name]()
                    return self._dependencies[name]

            # Check if factory exists
            elif name in self._factories:
                # Get dependency scope
                scope = self._scopes.get(name, DependencyScope.SINGLETON)

                # Handle scoped dependencies
                if scope == DependencyScope.SINGLETON:
                    # Create and cache singleton instance
                    self._dependencies[name] = self._factories[name]()
                    return self._dependencies[name]
                elif scope == DependencyScope.SESSION:
                    if session_id is None:
                        logger.warning(f"Session ID required for session-scoped dependency: {name}")
                        # Create and cache singleton instance as fallback
                        self._dependencies[name] = self._factories[name]()
                        return self._dependencies[name]
                    if session_id not in self._session_dependencies:
                        self._session_dependencies[session_id] = {}
                    if name not in self._session_dependencies[session_id]:
                        self._session_dependencies[session_id][name] = self._factories[name]()
                    return self._session_dependencies[session_id][name]
                elif scope == DependencyScope.REQUEST:
                    if request_id is None:
                        logger.warning(f"Request ID required for request-scoped dependency: {name}")
                        # Create and cache singleton instance as fallback
                        self._dependencies[name] = self._factories[name]()
                        return self._dependencies[name]
                    if request_id not in self._request_dependencies:
                        self._request_dependencies[request_id] = {}
                    if name not in self._request_dependencies[request_id]:
                        self._request_dependencies[request_id][name] = self._factories[name]()
                    return self._request_dependencies[request_id][name]
                elif scope == DependencyScope.TRANSIENT:
                    # Always create a new instance for transient dependencies
                    return self._factories[name]()

            # Return default or raise error
            elif default is not None:
                logger.debug(f"Dependency not found: {name}, using default value")
                return default
            else:
                logger.error(f"Dependency not found: {name}")
                raise DependencyError(f"Dependency not found: {name}")

        except Exception as e:
            if not isinstance(e, DependencyError):
                logger.error(f"Error getting dependency {name}: {str(e)}")
                raise DependencyError(f"Error getting dependency {name}: {str(e)}") from e
            raise

    def get_by_type(
        self,
        dependency_type: Type[T],
        default: Optional[T] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> T:
        """
        Get a dependency by type.

        Args:
            dependency_type: The dependency type
            default: Optional default value if dependency not found
            session_id: Optional session ID for session-scoped dependencies
            request_id: Optional request ID for request-scoped dependencies

        Returns:
            The dependency instance or default value

        Raises:
            DependencyError: If dependency not found and no default provided
        """
        try:
            # Use current session/request ID if not provided
            session_id = session_id or self._current_session_id
            request_id = request_id or self._current_request_id

            # Check registered dependencies
            for name, dep in self._dependencies.items():
                if isinstance(dep, dependency_type):
                    # Get dependency with proper scoping
                    return self.get(name, None, session_id, request_id)

            # Check factory-created dependencies
            for name in self._factories:
                # Try to create an instance and check its type
                try:
                    instance = self.get(name, None, session_id, request_id)
                    if isinstance(instance, dependency_type):
                        return cast(T, instance)
                except Exception:
                    # Skip factories that fail
                    continue

            # Return default or raise error
            if default is not None:
                logger.debug(
                    f"Dependency not found for type: {dependency_type.__name__}, using default value"
                )
                return default
            else:
                logger.error(f"Dependency not found for type: {dependency_type.__name__}")
                raise DependencyError(f"Dependency not found for type: {dependency_type.__name__}")

        except Exception as e:
            if not isinstance(e, DependencyError):
                logger.error(
                    f"Error getting dependency for type {dependency_type.__name__}: {str(e)}"
                )
                raise DependencyError(
                    f"Error getting dependency for type {dependency_type.__name__}: {str(e)}"
                ) from e
            raise

    def session_scope(self, session_id: Optional[str] = None) -> "SessionScope":
        """
        Create a session scope context manager.

        Args:
            session_id: Optional session ID (generated if not provided)

        Returns:
            A context manager for session-scoped dependencies
        """
        return SessionScope(self, session_id)

    def request_scope(self, request_id: Optional[str] = None) -> "RequestScope":
        """
        Create a request scope context manager.

        Args:
            request_id: Optional request ID (generated if not provided)

        Returns:
            A context manager for request-scoped dependencies
        """
        return RequestScope(self, request_id)

    def clear_session(self, session_id: str) -> None:
        """
        Clear session-scoped dependencies.

        Args:
            session_id: The session ID to clear
        """
        if session_id in self._session_dependencies:
            self._session_dependencies.pop(session_id)
            logger.debug(f"Cleared session dependencies for session {session_id}")

    def clear_request(self, request_id: str) -> None:
        """
        Clear request-scoped dependencies.

        Args:
            request_id: The request ID to clear
        """
        if request_id in self._request_dependencies:
            self._request_dependencies.pop(request_id)
            logger.debug(f"Cleared request dependencies for request {request_id}")

    def clear(self) -> None:
        """Clear all registered dependencies."""
        self._dependencies.clear()
        self._scopes.clear()
        self._factories.clear()
        self._session_dependencies.clear()
        self._request_dependencies.clear()
        self._dependency_graph.clear()
        self._current_session_id = None
        self._current_request_id = None
        logger.debug("Cleared all dependencies")


def inject_dependencies(
    func_or_class: Optional[F] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
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
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies

    Returns:
        The decorated function or class with automatic dependency injection

    Raises:
        DependencyError: If required dependencies cannot be resolved

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
        def process_data(data, model=None, validator=None):
            # model and validator are injected
            validated_data = validator.validate(data)
            return model.process(validated_data)
        ```
    """
    # Handle case where decorator is called with arguments
    if func_or_class is None:
        return lambda f: _inject_dependencies(f, session_id, request_id)

    # Handle case where decorator is called without arguments
    return _inject_dependencies(func_or_class, session_id, request_id)


def _inject_dependencies(
    func_or_class: F, session_id: Optional[str] = None, request_id: Optional[str] = None
) -> F:
    """
    Internal function for injecting dependencies.

    Args:
        func_or_class: The function or class to inject dependencies into
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies

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

            # Use current session/request ID if available
            current_session_id = session_id or provider._current_session_id
            current_request_id = request_id or provider._current_request_id

            # Inject dependencies for parameters not provided in kwargs
            for param_name, param in sig.parameters.items():
                if param_name != "self" and param_name not in kwargs:
                    # Try to get dependency by name
                    try:
                        kwargs[param_name] = provider.get(
                            param_name, None, current_session_id, current_request_id
                        )
                    except DependencyError:
                        # If not found by name, try by type annotation
                        if param.annotation != inspect.Parameter.empty:
                            try:
                                kwargs[param_name] = provider.get_by_type(
                                    param.annotation, None, current_session_id, current_request_id
                                )
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

            # Use current session/request ID if available
            current_session_id = session_id or provider._current_session_id
            current_request_id = request_id or provider._current_request_id

            # Inject dependencies for parameters not provided in kwargs
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs:
                    # Try to get dependency by name
                    try:
                        kwargs[param_name] = provider.get(
                            param_name, None, current_session_id, current_request_id
                        )
                    except DependencyError:
                        # If not found by name, try by type annotation
                        if param.annotation != inspect.Parameter.empty:
                            try:
                                kwargs[param_name] = provider.get_by_type(
                                    param.annotation, None, current_session_id, current_request_id
                                )
                            except DependencyError:
                                # If not found and has default, use default
                                if param.default != inspect.Parameter.empty:
                                    kwargs[param_name] = param.default

            # Call original function
            return func_or_class(*args, **kwargs)

        return cast(F, wrapper)


class SessionScope:
    """
    Context manager for session-scoped dependencies.

    This class provides a context manager for managing session-scoped
    dependencies. It sets the current session ID for the duration of
    the context and optionally clears session-scoped dependencies when
    the context exits.

    ## Architecture
    The SessionScope works with the DependencyProvider to manage the lifecycle
    of session-scoped dependencies. It temporarily sets the current session ID
    in the provider, allowing session-scoped dependencies to be created and
    retrieved for that session.

    ## Lifecycle
    1. **Enter**: Sets the current session ID in the provider
    2. **Usage**: Session-scoped dependencies are created and cached for this session
    3. **Exit**: Restores the previous session ID and optionally clears dependencies

    ## Examples
    ```python
    from sifaka.core.dependency import DependencyProvider

    provider = DependencyProvider()

    # Register session-scoped dependencies
    provider.register_factory(
        "database",
        lambda: Database.connect(),
        scope=DependencyScope.SESSION
    )

    # Use session scope
    with provider.session_scope("user_1") as session:
        # Get session-scoped dependency
        db = provider.get("database")  # Session-specific instance

        # All dependencies requested in this context will use this session
        result = process_data(input_data)  # Uses session-scoped dependencies

    # Session is now over, provider reverts to previous session (if any)
    ```

    Attributes:
        provider (DependencyProvider): The dependency provider
        session_id (str): The session ID for this scope
        previous_session_id (Optional[str]): The previous session ID to restore
        clear_on_exit (bool): Whether to clear session dependencies on exit
    """

    def __init__(self, provider: "DependencyProvider", session_id: Optional[str] = None):
        """
        Initialize a session scope.

        Args:
            provider: The dependency provider
            session_id: Optional session ID (generated if not provided)
        """
        self.provider = provider
        self.session_id = session_id or str(uuid.uuid4())
        self.previous_session_id = None

    def __enter__(self) -> str:
        """
        Enter the session scope.

        Returns:
            The session ID
        """
        # Save previous session ID
        self.previous_session_id = self.provider._current_session_id

        # Set current session ID
        self.provider._current_session_id = self.session_id

        logger.debug(f"Entered session scope: {self.session_id}")
        return self.session_id

    def __exit__(self, *_: Any) -> None:
        """Exit the session scope."""
        # Restore previous session ID
        self.provider._current_session_id = self.previous_session_id

        logger.debug(f"Exited session scope: {self.session_id}")


class RequestScope:
    """
    Context manager for request-scoped dependencies.

    This class provides a context manager for managing request-scoped
    dependencies. It sets the current request ID for the duration of
    the context and optionally clears request-scoped dependencies when
    the context exits.

    ## Architecture
    The RequestScope works with the DependencyProvider to manage the lifecycle
    of request-scoped dependencies. It temporarily sets the current request ID
    in the provider, allowing request-scoped dependencies to be created and
    retrieved for that request.

    ## Lifecycle
    1. **Enter**: Sets the current request ID in the provider
    2. **Usage**: Request-scoped dependencies are created and cached for this request
    3. **Exit**: Restores the previous request ID and optionally clears dependencies

    ## Examples
    ```python
    from sifaka.core.dependency import DependencyProvider

    provider = DependencyProvider()

    # Register request-scoped dependencies
    provider.register_factory(
        "validator",
        lambda: RequestValidator(),
        scope=DependencyScope.REQUEST
    )

    # Use request scope
    with provider.request_scope("request_123") as request:
        # Get request-scoped dependency
        validator = provider.get("validator")  # Request-specific instance

        # All dependencies requested in this context will use this request
        result = process_data(input_data)  # Uses request-scoped dependencies

    # Request is now over, provider reverts to previous request (if any)
    ```

    Attributes:
        provider (DependencyProvider): The dependency provider
        request_id (str): The request ID for this scope
        previous_request_id (Optional[str]): The previous request ID to restore
        clear_on_exit (bool): Whether to clear request dependencies on exit
    """

    def __init__(self, provider: "DependencyProvider", request_id: Optional[str] = None):
        """
        Initialize a request scope.

        Args:
            provider: The dependency provider
            request_id: Optional request ID (generated if not provided)
        """
        self.provider = provider
        self.request_id = request_id or str(uuid.uuid4())
        self.previous_request_id = None

    def __enter__(self) -> str:
        """
        Enter the request scope.

        Returns:
            The request ID
        """
        # Save previous request ID
        self.previous_request_id = self.provider._current_request_id

        # Set current request ID
        self.provider._current_request_id = self.request_id

        logger.debug(f"Entered request scope: {self.request_id}")
        return self.request_id

    def __exit__(self, *_: Any) -> None:
        """Exit the request scope."""
        # Restore previous request ID
        self.provider._current_request_id = self.previous_request_id

        # Clear request dependencies
        self.provider.clear_request(self.request_id)

        logger.debug(f"Exited request scope: {self.request_id}")


class DependencyInjector:
    """
    Utility class for manually injecting dependencies into objects.

    This class provides methods for manually injecting dependencies into
    objects and creating objects with injected dependencies. It's useful
    when automatic dependency injection via the decorator is not suitable.

    ## Architecture
    The DependencyInjector provides static methods for manual dependency
    injection, complementing the automatic injection provided by the
    `inject_dependencies` decorator. It works by setting attributes on
    objects or passing dependencies to constructors.

    ## Examples
    ```python
    from sifaka.core.dependency import DependencyInjector

    # Inject dependencies into an existing object
    component = MyComponent()
    DependencyInjector.inject(
        component,
        model=OpenAIModel(),
        validator=LengthValidator()
    )

    # Create an object with injected dependencies
    component = DependencyInjector.create_with_dependencies(
        MyComponent,
        model=OpenAIModel(),
        validator=LengthValidator()
    )
    ```
    """

    @staticmethod
    def inject(obj: Any, **dependencies: Any) -> None:
        """
        Inject dependencies into an existing object.

        This method injects dependencies into an existing object by setting
        attributes on the object. It only sets attributes that already exist
        on the object, ignoring dependencies that don't match existing attributes.

        Args:
            obj: The object to inject dependencies into
            **dependencies: The dependencies to inject as keyword arguments

        Example:
            ```python
            # Create an object
            component = MyComponent()

            # Inject dependencies
            DependencyInjector.inject(
                component,
                model=OpenAIModel(),
                validator=LengthValidator()
            )

            # Now component.model and component.validator are set
            ```
        """
        for name, dependency in dependencies.items():
            if hasattr(obj, name):
                setattr(obj, name, dependency)
                logger.debug(f"Injected dependency {name} into {obj.__class__.__name__}")

    @staticmethod
    def create_with_dependencies(cls: Type[T], **dependencies: Any) -> T:
        """
        Create an object with injected dependencies.

        This method creates a new instance of the specified class and injects
        dependencies into it. Dependencies are passed to the constructor and
        also set as attributes if they exist on the object.

        Args:
            cls: The class to create an instance of
            **dependencies: The dependencies to inject as keyword arguments

        Returns:
            An instance of the class with dependencies injected

        Example:
            ```python
            # Create an object with injected dependencies
            component = DependencyInjector.create_with_dependencies(
                MyComponent,
                model=OpenAIModel(),
                validator=LengthValidator()
            )

            # Dependencies are passed to constructor and set as attributes
            ```
        """
        # Create instance
        instance = cls(**dependencies)

        # Inject any remaining dependencies
        DependencyInjector.inject(instance, **dependencies)

        return instance


def provide_dependency(
    name: str,
    dependency: Any,
    scope: DependencyScope = DependencyScope.SINGLETON,
    dependencies: Optional[List[str]] = None,
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
        from sifaka.core.dependency import provide_dependency, DependencyScope

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
    dependencies: Optional[List[str]] = None,
) -> None:
    """
    Register a factory function with the global dependency provider.

    This function is a convenience wrapper around DependencyProvider.register_factory()
    that uses the global singleton provider. It registers a factory function that will
    be called to create the dependency when it is first requested.

    Args:
        name: The dependency name used for lookup
        factory: Factory function that creates the dependency when called
        scope: The dependency scope that controls its lifecycle
        dependencies: List of dependency names this dependency depends on

    Raises:
        ConfigurationError: If registering would create a circular dependency
        DependencyError: If factory or dependencies are invalid

    Example:
        ```python
        from sifaka.core.dependency import provide_factory, DependencyScope

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
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Any:
    """
    Get a dependency by name from the global dependency provider.

    This function is a convenience wrapper around DependencyProvider.get()
    that uses the global singleton provider. It retrieves a dependency by
    its registered name, handling different dependency scopes appropriately.

    Args:
        name: The dependency name to look up
        default: Optional default value if dependency not found
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies

    Returns:
        The dependency instance or default value if not found

    Raises:
        DependencyError: If dependency not found and no default provided

    Example:
        ```python
        from sifaka.core.dependency import get_dependency

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
    default: Optional[T] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> T:
    """
    Get a dependency by type from the global dependency provider.

    This function is a convenience wrapper around DependencyProvider.get_by_type()
    that uses the global singleton provider. It retrieves a dependency by
    its type, searching through all registered dependencies for one that
    matches the specified type.

    Args:
        dependency_type: The dependency type to look up
        default: Optional default value if dependency not found
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies

    Returns:
        The dependency instance or default value if not found

    Raises:
        DependencyError: If dependency not found and no default provided

    Example:
        ```python
        from sifaka.core.dependency import get_dependency_by_type
        from sifaka.models.interfaces import ModelProvider
        from sifaka.validators.interfaces import Validator

        # Get a dependency by type
        model = get_dependency_by_type(ModelProvider)

        # Get a dependency by type with default
        validator = get_dependency_by_type(Validator, default=DefaultValidator())

        # Get a session-scoped dependency by type
        db = get_dependency_by_type(Database, session_id="user_1")
        ```
    """
    provider = DependencyProvider()
    return provider.get_by_type(dependency_type, default, session_id, request_id)


def create_session_scope(session_id: Optional[str] = None) -> SessionScope:
    """
    Create a session scope context manager.

    Args:
        session_id: Optional session ID (generated if not provided)

    Returns:
        A context manager for session-scoped dependencies
    """
    provider = DependencyProvider()
    return provider.session_scope(session_id)


def create_request_scope(request_id: Optional[str] = None) -> RequestScope:
    """
    Create a request scope context manager.

    Args:
        request_id: Optional request ID (generated if not provided)

    Returns:
        A context manager for request-scoped dependencies
    """
    provider = DependencyProvider()
    return provider.request_scope(request_id)


def clear_dependencies() -> None:
    """Clear all registered dependencies."""
    provider = DependencyProvider()
    provider.clear()
