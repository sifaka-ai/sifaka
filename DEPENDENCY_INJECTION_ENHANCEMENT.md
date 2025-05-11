# Dependency Injection Enhancement Plan

This document outlines the plan for enhancing the dependency injection system in the Sifaka codebase.

## Current State

The Sifaka codebase currently has a dependency injection system in `sifaka/core/dependency.py` that provides:

1. A `DependencyProvider` singleton class for registering and retrieving dependencies
2. An `inject_dependencies` decorator for injecting dependencies into classes and functions
3. A `DependencyInjector` utility class for manual dependency injection
4. Helper functions like `provide_dependency`, `get_dependency`, etc.

However, the current implementation has some limitations:

1. No support for scoped dependencies (request, session, singleton)
2. Limited error handling and logging
3. No validation for registered dependencies
4. No dependency resolution strategies
5. Inconsistent usage across the codebase

## Enhancement Goals

1. Add support for scoped dependencies
2. Improve error handling and logging
3. Add validation for registered dependencies
4. Implement dependency resolution strategies
5. Standardize usage across the codebase

## Implementation Plan

### 1. Enhance DependencyProvider Implementation

#### 1.1 Add Support for Scoped Dependencies

```python
class DependencyScope(Enum):
    """Dependency scope enumeration."""
    SINGLETON = "singleton"  # One instance per application
    SESSION = "session"      # One instance per session
    REQUEST = "request"      # One instance per request
    TRANSIENT = "transient"  # New instance each time

class DependencyProvider:
    """Dependency provider for Sifaka components."""

    _instance = None
    _dependencies: Dict[str, Any] = {}
    _scopes: Dict[str, DependencyScope] = {}
    _factories: Dict[str, Callable[[], Any]] = {}
    _session_dependencies: Dict[str, Dict[str, Any]] = {}
    _request_dependencies: Dict[str, Dict[str, Any]] = {}

    def __new__(cls) -> "DependencyProvider":
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(DependencyProvider, cls).__new__(cls)
            cls._instance._dependencies = {}
            cls._instance._scopes = {}
            cls._instance._factories = {}
            cls._instance._session_dependencies = {}
            cls._instance._request_dependencies = {}
        return cls._instance

    def register(
        self, 
        name: str, 
        dependency: Any, 
        scope: DependencyScope = DependencyScope.SINGLETON
    ) -> None:
        """
        Register a dependency.

        Args:
            name: The dependency name
            dependency: The dependency instance
            scope: The dependency scope
        """
        self._dependencies[name] = dependency
        self._scopes[name] = scope
        logger.debug(f"Registered dependency {name}: {dependency.__class__.__name__} with scope {scope.value}")

    def register_factory(
        self, 
        name: str, 
        factory: Callable[[], Any], 
        scope: DependencyScope = DependencyScope.SINGLETON
    ) -> None:
        """
        Register a dependency factory.

        Args:
            name: The dependency name
            factory: The factory function
            scope: The dependency scope
        """
        self._factories[name] = factory
        self._scopes[name] = scope
        logger.debug(f"Registered dependency factory {name} with scope {scope.value}")

    def get(
        self, 
        name: str, 
        default: Optional[Any] = None, 
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Any:
        """
        Get a dependency by name.

        Args:
            name: The dependency name
            default: Optional default value if dependency not found
            session_id: Optional session ID for session-scoped dependencies
            request_id: Optional request ID for request-scoped dependencies

        Returns:
            The dependency instance or default value

        Raises:
            DependencyError: If dependency not found and no default provided
        """
        # Check if dependency exists
        if name in self._dependencies:
            # Get dependency scope
            scope = self._scopes.get(name, DependencyScope.SINGLETON)

            # Handle scoped dependencies
            if scope == DependencyScope.SINGLETON:
                return self._dependencies[name]
            elif scope == DependencyScope.SESSION:
                if session_id is None:
                    raise DependencyError(f"Session ID required for session-scoped dependency: {name}")
                if session_id not in self._session_dependencies:
                    self._session_dependencies[session_id] = {}
                if name not in self._session_dependencies[session_id]:
                    if name in self._factories:
                        self._session_dependencies[session_id][name] = self._factories[name]()
                    else:
                        self._session_dependencies[session_id][name] = self._dependencies[name]
                return self._session_dependencies[session_id][name]
            elif scope == DependencyScope.REQUEST:
                if request_id is None:
                    raise DependencyError(f"Request ID required for request-scoped dependency: {name}")
                if request_id not in self._request_dependencies:
                    self._request_dependencies[request_id] = {}
                if name not in self._request_dependencies[request_id]:
                    if name in self._factories:
                        self._request_dependencies[request_id][name] = self._factories[name]()
                    else:
                        self._request_dependencies[request_id][name] = self._dependencies[name]
                return self._request_dependencies[request_id][name]
            elif scope == DependencyScope.TRANSIENT:
                if name in self._factories:
                    return self._factories[name]()
                else:
                    return self._dependencies[name]
        # Check if factory exists
        elif name in self._factories:
            # Get dependency scope
            scope = self._scopes.get(name, DependencyScope.SINGLETON)

            # Handle scoped dependencies
            if scope == DependencyScope.SINGLETON:
                self._dependencies[name] = self._factories[name]()
                return self._dependencies[name]
            elif scope == DependencyScope.SESSION:
                if session_id is None:
                    raise DependencyError(f"Session ID required for session-scoped dependency: {name}")
                if session_id not in self._session_dependencies:
                    self._session_dependencies[session_id] = {}
                if name not in self._session_dependencies[session_id]:
                    self._session_dependencies[session_id][name] = self._factories[name]()
                return self._session_dependencies[session_id][name]
            elif scope == DependencyScope.REQUEST:
                if request_id is None:
                    raise DependencyError(f"Request ID required for request-scoped dependency: {name}")
                if request_id not in self._request_dependencies:
                    self._request_dependencies[request_id] = {}
                if name not in self._request_dependencies[request_id]:
                    self._request_dependencies[request_id][name] = self._factories[name]()
                return self._request_dependencies[request_id][name]
            elif scope == DependencyScope.TRANSIENT:
                return self._factories[name]()
        # Return default or raise error
        elif default is not None:
            return default
        else:
            raise DependencyError(f"Dependency not found: {name}")
```

#### 1.2 Improve Error Handling and Logging

```python
def get(
    self, 
    name: str, 
    default: Optional[Any] = None, 
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Any:
    """
    Get a dependency by name.

    Args:
        name: The dependency name
        default: Optional default value if dependency not found
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies

    Returns:
        The dependency instance or default value

    Raises:
        DependencyError: If dependency not found and no default provided
    """
    try:
        # Check if dependency exists
        if name in self._dependencies:
            # ... (existing code)
        # Check if factory exists
        elif name in self._factories:
            # ... (existing code)
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
```

#### 1.3 Add Validation for Registered Dependencies

```python
def register(
    self, 
    name: str, 
    dependency: Any, 
    scope: DependencyScope = DependencyScope.SINGLETON,
    validate: bool = True
) -> None:
    """
    Register a dependency.

    Args:
        name: The dependency name
        dependency: The dependency instance
        scope: The dependency scope
        validate: Whether to validate the dependency

    Raises:
        DependencyError: If validation fails
    """
    # Validate dependency
    if validate:
        self._validate_dependency(name, dependency)

    # Register dependency
    self._dependencies[name] = dependency
    self._scopes[name] = scope
    logger.debug(f"Registered dependency {name}: {dependency.__class__.__name__} with scope {scope.value}")

def _validate_dependency(self, name: str, dependency: Any) -> None:
    """
    Validate a dependency.

    Args:
        name: The dependency name
        dependency: The dependency instance

    Raises:
        DependencyError: If validation fails
    """
    # Check if dependency is None
    if dependency is None:
        raise DependencyError(f"Dependency {name} cannot be None")

    # Check if dependency is already registered
    if name in self._dependencies:
        logger.warning(f"Dependency {name} is already registered, overwriting")

    # Check if dependency has required methods
    if hasattr(dependency, "__required_methods__"):
        for method_name in dependency.__required_methods__:
            if not hasattr(dependency, method_name) or not callable(getattr(dependency, method_name)):
                raise DependencyError(f"Dependency {name} missing required method: {method_name}")

    # Check if dependency has required attributes
    if hasattr(dependency, "__required_attributes__"):
        for attr_name in dependency.__required_attributes__:
            if not hasattr(dependency, attr_name):
                raise DependencyError(f"Dependency {name} missing required attribute: {attr_name}")
```

#### 1.4 Implement Dependency Resolution Strategies

```python
def resolve_dependencies(
    self, 
    cls: Type[T], 
    **kwargs: Any
) -> T:
    """
    Resolve dependencies for a class.

    Args:
        cls: The class to resolve dependencies for
        **kwargs: Additional keyword arguments

    Returns:
        An instance of the class with dependencies injected

    Raises:
        DependencyError: If dependency resolution fails
    """
    # Get constructor signature
    sig = inspect.signature(cls.__init__)

    # Resolve dependencies
    resolved_kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name in kwargs:
            resolved_kwargs[param_name] = kwargs[param_name]
        else:
            # Try to get dependency by name
            try:
                resolved_kwargs[param_name] = self.get(param_name)
            except DependencyError:
                # If not found by name, try by type annotation
                if param.annotation != inspect.Parameter.empty:
                    try:
                        resolved_kwargs[param_name] = self.get_by_type(param.annotation)
                    except DependencyError:
                        # If not found and has default, use default
                        if param.default != inspect.Parameter.empty:
                            resolved_kwargs[param_name] = param.default
                        else:
                            raise DependencyError(f"Could not resolve dependency {param_name} for {cls.__name__}")

    # Create instance
    return cls(**resolved_kwargs)
```

### 2. Standardize Usage Across the Codebase

#### 2.1 Update Factory Functions

```python
def create_model_provider(provider_type: str, **kwargs: Any) -> Any:
    """
    Create a model provider.

    Args:
        provider_type: The provider type
        **kwargs: Additional keyword arguments

    Returns:
        A model provider instance

    Raises:
        ValueError: If the provider type is invalid
    """
    # Get dependency provider
    provider = DependencyProvider()

    # Check if provider is registered
    try:
        return provider.get(f"model_provider_{provider_type}")
    except DependencyError:
        pass

    # Create provider based on type
    if provider_type == "openai":
        from sifaka.models.providers.openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        from sifaka.models.providers.anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)
    elif provider_type == "gemini":
        from sifaka.models.providers.gemini import GeminiProvider
        return GeminiProvider(**kwargs)
    elif provider_type == "mock":
        from sifaka.models.providers.mock import MockProvider
        return MockProvider(**kwargs)
    else:
        raise ValueError(f"Invalid provider type: {provider_type}")
```

#### 2.2 Update Component Constructors

```python
class Critic:
    """Base critic implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        prompt_factory: Optional[Any] = None,
        config: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Initialize the critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: The language model provider to use
            prompt_factory: Optional prompt factory to use
            config: Optional configuration for the critic
            **kwargs: Additional configuration parameters
        """
        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        # Store dependencies in state
        self._state_manager.update("model", llm_provider)
        self._state_manager.update("prompt_factory", prompt_factory or self._create_default_prompt_factory())
```

#### 2.3 Use the inject_dependencies Decorator

```python
@inject_dependencies
class Critic:
    """Base critic implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any = None,
        prompt_factory: Optional[Any] = None,
        config: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Initialize the critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: The language model provider to use
            prompt_factory: Optional prompt factory to use
            config: Optional configuration for the critic
            **kwargs: Additional configuration parameters
        """
        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        # Store dependencies in state
        self._state_manager.update("model", llm_provider)
        self._state_manager.update("prompt_factory", prompt_factory or self._create_default_prompt_factory())
```

## Success Criteria

1. Enhanced DependencyProvider implementation with support for scoped dependencies
2. Improved error handling and logging
3. Validation for registered dependencies
4. Dependency resolution strategies
5. Standardized usage across the codebase
6. Comprehensive documentation for dependency management
7. All components use explicit dependency injection
8. Factory functions follow standardized patterns
9. Component initialization is standardized
10. Tests validate proper dependency injection
