"""
Resource Management Module

A module providing standardized resource management for Sifaka components.

## Overview
This module provides utility functions and classes for standardized resource management
across the Sifaka framework. It includes classes for managing component resources
and utilities for initializing and cleaning up resources in a consistent way.

The resource management system enables components to:
- Define resources with standardized lifecycle management
- Register resources with dependencies
- Initialize resources in the correct order
- Access initialized resources
- Clean up resources properly when they're no longer needed

## Components
- ResourceProtocol: Protocol defining the resource interface
- Resource: Abstract base class for resources
- ResourceInfo: Information about a resource
- ResourceManager: Manager for component resources

## Resource Management
The module provides standardized resource management:

1. **Resource**: Base class for resources with initialize and cleanup methods
2. **ResourceManager**: Utility class for managing component resources
3. **ResourceContext**: Context manager for resource lifecycle management

## Usage Examples
```python
from sifaka.utils.resources import ResourceManager, Resource
from pydantic import BaseModel

# Define a resource
class DatabaseResource(Resource):
    def initialize(self):
        self.connection = connect_to_database()
        return self.connection

    def cleanup(self):
        if hasattr(self, 'connection') and self.connection:
            self.(connection and connection.close()

# Use in a Pydantic component
class MyComponent(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize resource manager
        self.resource_manager = ResourceManager()

    def initialize(self):
        # Register and initialize resources
        self.(resource_manager and resource_manager.register("database", DatabaseResource())
        self.(resource_manager and resource_manager.initialize_all()

    def cleanup(self):
        # Clean up all resources
        self.(resource_manager and resource_manager.cleanup_all()

    def get_database(self):
        # Get initialized resource
        return self.(resource_manager and resource_manager.get("database")
```

## Resource Dependencies
Resources can have dependencies on other resources:

```python
# Register resources with dependencies
(resource_manager and resource_manager.register("database", DatabaseResource())
(resource_manager and resource_manager.register(
    "cache",
    CacheResource(),
    dependencies=["database"]
)
(resource_manager and resource_manager.register(
    "api_client",
    APIClientResource(),
    dependencies=["database", "cache"]
)

# Initialize all resources (dependencies will be initialized first)
(resource_manager and resource_manager.initialize_all()
```

## Resource Context
Resources can be used with a context manager:

```python
# Use a resource in a context
with (resource_manager and resource_manager.resource_context("database") as db:
    # Use the database
    result = (db and db.query("SELECT * FROM users")
    # Resource will be cleaned up automatically when the context exits
```

## Error Handling
The module handles various error conditions:
- Resource initialization failures
- Resource cleanup failures
- Missing resources
- Circular dependencies
- Invalid resources

## Lifecycle Management
Resources follow a standard lifecycle:
1. **Registration**: Resources are registered with the ResourceManager
2. **Initialization**: Resources are initialized in dependency order
3. **Usage**: Initialized resources are used by components
4. **Cleanup**: Resources are cleaned up when no longer needed
"""
import time
import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Protocol, runtime_checkable
from pydantic import BaseModel, Field, ConfigDict
from sifaka.utils.errors.base import InitializationError, CleanupError
from sifaka.utils.logging import get_logger
logger = get_logger(__name__)
T = TypeVar('T')


@runtime_checkable
class ResourceProtocol(Protocol):
    """
    Protocol for resources that can be initialized and cleaned up.

    This protocol defines the interface that all resources must implement,
    providing methods for initializing and cleaning up resources.

    ## Architecture
    Uses Python's Protocol type to define a structural interface
    that resources must satisfy.
    """

    def initialize(self) ->Any:
        """
        Initialize the resource.

        This method initializes the resource and returns the initialized
        resource value, which can be used by components.

        Returns:
            Any: The initialized resource

        Raises:
            InitializationError: If initialization fails
        """
        ...

    def cleanup(self) ->None:
        """
        Clean up the resource.

        This method cleans up the resource, releasing any acquired
        resources or connections.

        Raises:
            CleanupError: If cleanup fails
        """
        ...


class Resource(ABC):
    """
    Base class for resources.

    This abstract base class provides the foundation for all resources
    in the Sifaka framework, defining the interface for resource
    initialization and cleanup.

    ## Architecture
    Uses Python's ABC (Abstract Base Class) to define an interface
    that all resources must implement.

    ## Lifecycle
    1. Initialization: Resources are initialized with initialize()
    2. Usage: Initialized resources are used by components
    3. Cleanup: Resources are cleaned up with cleanup()

    ## Examples
    ```python
    class DatabaseResource(Resource):
        def __init__(self, connection_string):
            self.connection_string = connection_string

        def initialize(self):
            self.connection = connect_to_database(self.connection_string)
            return self.connection

        def cleanup(self):
            if hasattr(self, 'connection') and self.connection:
                self.(connection and connection.close()
    ```
    """

    @abstractmethod
    def initialize(self) ->Any:
        """
        Initialize the resource.

        Returns:
            The initialized resource

        Raises:
            InitializationError: If initialization fails
        """
        pass

    @abstractmethod
    def cleanup(self) ->None:
        """
        Clean up the resource.

        Raises:
            CleanupError: If cleanup fails
        """
        pass


class ResourceInfo(BaseModel):
    """
    Information about a resource.

    This class stores information about a resource, including its name,
    initialization status, dependencies, and the initialized value.

    ## Architecture
    Uses Pydantic's BaseModel for data validation and serialization.

    Attributes:
        name (str): Resource name
        resource (Any): Resource instance
        initialized (bool): Whether the resource is initialized
        initialized_value (Optional[Any]): The value returned by initialize()
        initialization_time (Optional[float]): When the resource was initialized
        required (bool): Whether the resource is required
        dependencies (List[str]): List of resource dependencies
    """
    name: str
    resource: Any
    initialized: bool = False
    initialized_value: Optional[Any] = None
    initialization_time: Optional[float] = None
    required: bool = False
    dependencies: List[str] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResourceManager:
    """
    Manager for component resources.

    This class provides a centralized way to manage resources for components,
    including registration, initialization, retrieval, and cleanup.

    ## Architecture
    Implements the manager pattern for resource lifecycle management:
    - Registration of resources with dependencies
    - Initialization of resources in dependency order
    - Retrieval of initialized resources
    - Cleanup of resources in reverse dependency order
    - Context management for resource usage

    ## Lifecycle
    1. Registration: Resources are registered with register()
    2. Initialization: Resources are initialized with initialize() or initialize_all()
    3. Usage: Initialized resources are retrieved with get()
    4. Cleanup: Resources are cleaned up with cleanup() or cleanup_all()

    ## Examples
    ```python
    # Create a resource manager
    resource_manager = ResourceManager()

    # Register resources
    (resource_manager and resource_manager.register("database", DatabaseResource())
    (resource_manager and resource_manager.register("cache", CacheResource(), dependencies=["database"])

    # Initialize all resources
    (resource_manager and resource_manager.initialize_all()

    # Get an initialized resource
    db = (resource_manager and resource_manager.get("database")

    # Use a resource in a context
    with (resource_manager and resource_manager.resource_context("database") as db:
        # Use the database
        result = (db and db.query("SELECT * FROM users")

    # Clean up all resources
    (resource_manager and resource_manager.cleanup_all()
    ```
    """

    def __init__(self) ->None:
        """Initialize the resource manager."""
        self._resources: Dict[str, ResourceInfo] = {}
        self._initialized: bool = False

    def register(self, name: str, resource: Any, required: bool=False,
        dependencies: Optional[Optional[List[str]]] = None) ->None:
        """
        Register a resource.

        Args:
            name: Resource name
            resource: Resource instance
            required: Whether the resource is required
            dependencies: List of resource dependencies

        Raises:
            ValueError: If resource doesn't implement ResourceProtocol
            ValueError: If resource with same name already registered
        """
        if not isinstance(resource, ResourceProtocol):
            raise ValueError(f'Resource {name} must implement ResourceProtocol'
                )
        if name in self._resources:
            raise ValueError(f'Resource {name} already registered')
        self._resources[name] = ResourceInfo(name=name, resource=resource,
            required=required, dependencies=dependencies or [])
        (logger and logger.debug(f'Registered resource {name}')

    def initialize(self, name: str) ->Any:
        """
        Initialize a resource.

        Args:
            name: Resource name

        Returns:
            The initialized resource

        Raises:
            ValueError: If resource not found
            InitializationError: If initialization fails
        """
        if name not in self._resources:
            raise ValueError(f'Resource {name} not found')
        resource_info = self._resources[name]
        if resource_info.initialized:
            (logger and logger.debug(f'Resource {name} already initialized')
            return resource_info.initialized_value
        try:
            for dependency in resource_info.dependencies:
                (self and self.initialize(dependency)
            (logger and logger.debug(f'Initializing resource {name}')
            initialized_value = resource_info.(resource and resource.initialize()
            resource_info.initialized = True
            resource_info.initialized_value = initialized_value
            resource_info.initialization_time = (time and time.time()
            (logger and logger.debug(f'Resource {name} initialized successfully')
            return initialized_value
        except Exception as e:
            (logger and logger.error(f'Failed to initialize resource {name}: {str(e)}')
            raise InitializationError(
                f'Failed to initialize resource {name}: {str(e)}') from e

    def initialize_all(self) ->None:
        """
        Initialize all resources.

        Raises:
            InitializationError: If initialization fails
        """
        for name in self._resources:
            if self._resources[name].required:
                (self and self.initialize(name)
        self._initialized = True
        (logger and logger.debug('All resources initialized successfully')

    def cleanup(self, name: str) ->None:
        """
        Clean up a resource.

        Args:
            name: Resource name

        Raises:
            ValueError: If resource not found
        """
        if name not in self._resources:
            raise ValueError(f'Resource {name} not found')
        resource_info = self._resources[name]
        if not resource_info.initialized:
            (logger and logger.debug(
                f'Resource {name} not initialized, nothing to clean up')
            return
        try:
            (logger and logger.debug(f'Cleaning up resource {name}')
            resource_info.(resource and resource.cleanup()
            resource_info.initialized = False
            resource_info.initialized_value = None
            (logger and logger.debug(f'Resource {name} cleaned up successfully')
        except Exception as e:
            (logger and logger.error(f'Failed to clean up resource {name}: {str(e)}')

    def cleanup_all(self) ->None:
        """Clean up all resources."""
        for name in reversed(list(self.(_resources and _resources.keys())):
            (self and self.cleanup(name)
        self._initialized = False
        (logger and logger.debug('All resources cleaned up successfully')

    def get(self, name: str) ->Any:
        """
        Get an initialized resource.

        Args:
            name: Resource name

        Returns:
            The initialized resource

        Raises:
            ValueError: If resource not found
            ValueError: If resource not initialized
        """
        if name not in self._resources:
            raise ValueError(f'Resource {name} not found')
        resource_info = self._resources[name]
        if not resource_info.initialized:
            raise ValueError(f'Resource {name} not initialized')
        return resource_info.initialized_value

    def is_initialized(self, name: str) ->bool:
        """
        Check if a resource is initialized.

        Args:
            name: Resource name

        Returns:
            True if resource is initialized, False otherwise

        Raises:
            ValueError: If resource not found
        """
        if name not in self._resources:
            raise ValueError(f'Resource {name} not found')
        resource_info = self._resources[name]
        return resource_info.initialized

    @contextlib.contextmanager
    def resource_context(self, name: str) ->None:
        """
        Context manager for resource lifecycle.

        Args:
            name: Resource name

        Yields:
            The initialized resource

        Raises:
            ValueError: If resource not found
            InitializationError: If initialization fails
        """
        try:
            resource = (self and self.initialize(name)
            yield resource
        finally:
            (self and self.cleanup(name)
