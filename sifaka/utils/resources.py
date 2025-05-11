"""
Resource Management Utilities for Sifaka.

This module provides utility functions and classes for standardized resource management
across the Sifaka framework. It includes classes for managing component resources
and utilities for initializing and cleaning up resources in a consistent way.

## Resource Management

The module provides standardized resource management:

1. **Resource**: Base class for resources
2. **ResourceManager**: Utility class for managing component resources
3. **ResourceContext**: Context manager for resource lifecycle management

## Usage Examples

```python
from sifaka.utils.resources import ResourceManager, Resource
from pydantic import BaseModel

class DatabaseResource(Resource):
    def initialize(self):
        self.connection = connect_to_database()
        return self.connection
        
    def cleanup(self):
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

class MyComponent(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize resource manager
        self.resource_manager = ResourceManager()
        
    def initialize(self):
        # Register and initialize resources
        self.resource_manager.register("database", DatabaseResource())
        self.resource_manager.initialize_all()
        
    def cleanup(self):
        # Clean up all resources
        self.resource_manager.cleanup_all()
        
    def get_database(self):
        # Get initialized resource
        return self.resource_manager.get("database")
```

## Resource Management Pattern

The recommended pattern is to use the ResourceManager directly:

```python
from sifaka.utils.resources import ResourceManager, Resource

class APIClientResource(Resource):
    def __init__(self, api_key):
        self.api_key = api_key
        
    def initialize(self):
        self.client = APIClient(self.api_key)
        return self.client
        
    def cleanup(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

class MyComponent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.resource_manager = ResourceManager()
        
    def initialize(self):
        # Register and initialize resources
        self.resource_manager.register("api_client", APIClientResource(self.api_key))
        self.resource_manager.initialize_all()
        
    def process(self, data):
        # Get initialized resource
        client = self.resource_manager.get("api_client")
        return client.process(data)
        
    def cleanup(self):
        # Clean up all resources
        self.resource_manager.cleanup_all()
```
"""

import time
import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ConfigDict

from sifaka.utils.errors import InitializationError, CleanupError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # Resource type


@runtime_checkable
class ResourceProtocol(Protocol):
    """Protocol for resources that can be initialized and cleaned up."""

    def initialize(self) -> Any:
        """Initialize the resource."""
        ...

    def cleanup(self) -> None:
        """Clean up the resource."""
        ...


class Resource(ABC):
    """Base class for resources."""

    @abstractmethod
    def initialize(self) -> Any:
        """
        Initialize the resource.
        
        Returns:
            The initialized resource
            
        Raises:
            InitializationError: If initialization fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up the resource.
        
        Raises:
            CleanupError: If cleanup fails
        """
        pass


class ResourceInfo(BaseModel):
    """Information about a resource."""

    name: str
    resource: Any
    initialized: bool = False
    initialized_value: Optional[Any] = None
    initialization_time: Optional[float] = None
    required: bool = False
    dependencies: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResourceManager:
    """Manager for component resources."""

    def __init__(self):
        """Initialize the resource manager."""
        self._resources: Dict[str, ResourceInfo] = {}
        self._initialized: bool = False

    def register(
        self, 
        name: str, 
        resource: Any, 
        required: bool = False,
        dependencies: Optional[List[str]] = None
    ) -> None:
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
        # Validate resource
        if not isinstance(resource, ResourceProtocol):
            raise ValueError(f"Resource {name} must implement ResourceProtocol")
            
        # Check if resource already registered
        if name in self._resources:
            raise ValueError(f"Resource {name} already registered")
            
        # Register resource
        self._resources[name] = ResourceInfo(
            name=name,
            resource=resource,
            required=required,
            dependencies=dependencies or []
        )
        
        logger.debug(f"Registered resource {name}")

    def initialize(self, name: str) -> Any:
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
        # Check if resource exists
        if name not in self._resources:
            raise ValueError(f"Resource {name} not found")
            
        # Get resource info
        resource_info = self._resources[name]
        
        # Check if already initialized
        if resource_info.initialized:
            logger.debug(f"Resource {name} already initialized")
            return resource_info.initialized_value
            
        try:
            # Initialize dependencies first
            for dependency in resource_info.dependencies:
                self.initialize(dependency)
                
            # Initialize resource
            logger.debug(f"Initializing resource {name}")
            initialized_value = resource_info.resource.initialize()
            
            # Update resource info
            resource_info.initialized = True
            resource_info.initialized_value = initialized_value
            resource_info.initialization_time = time.time()
            
            logger.debug(f"Resource {name} initialized successfully")
            return initialized_value
            
        except Exception as e:
            logger.error(f"Failed to initialize resource {name}: {str(e)}")
            raise InitializationError(f"Failed to initialize resource {name}: {str(e)}") from e

    def initialize_all(self) -> None:
        """
        Initialize all resources.
        
        Raises:
            InitializationError: If initialization fails
        """
        # Initialize all resources
        for name in self._resources:
            if self._resources[name].required:
                self.initialize(name)
                
        self._initialized = True
        logger.debug("All resources initialized successfully")

    def cleanup(self, name: str) -> None:
        """
        Clean up a resource.
        
        Args:
            name: Resource name
            
        Raises:
            ValueError: If resource not found
        """
        # Check if resource exists
        if name not in self._resources:
            raise ValueError(f"Resource {name} not found")
            
        # Get resource info
        resource_info = self._resources[name]
        
        # Check if initialized
        if not resource_info.initialized:
            logger.debug(f"Resource {name} not initialized, nothing to clean up")
            return
            
        try:
            # Clean up resource
            logger.debug(f"Cleaning up resource {name}")
            resource_info.resource.cleanup()
            
            # Update resource info
            resource_info.initialized = False
            resource_info.initialized_value = None
            
            logger.debug(f"Resource {name} cleaned up successfully")
            
        except Exception as e:
            # Log but don't raise
            logger.error(f"Failed to clean up resource {name}: {str(e)}")

    def cleanup_all(self) -> None:
        """Clean up all resources."""
        # Clean up all resources in reverse order of initialization
        for name in reversed(list(self._resources.keys())):
            self.cleanup(name)
            
        self._initialized = False
        logger.debug("All resources cleaned up successfully")

    def get(self, name: str) -> Any:
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
        # Check if resource exists
        if name not in self._resources:
            raise ValueError(f"Resource {name} not found")
            
        # Get resource info
        resource_info = self._resources[name]
        
        # Check if initialized
        if not resource_info.initialized:
            raise ValueError(f"Resource {name} not initialized")
            
        return resource_info.initialized_value

    def is_initialized(self, name: str) -> bool:
        """
        Check if a resource is initialized.
        
        Args:
            name: Resource name
            
        Returns:
            True if resource is initialized, False otherwise
            
        Raises:
            ValueError: If resource not found
        """
        # Check if resource exists
        if name not in self._resources:
            raise ValueError(f"Resource {name} not found")
            
        # Get resource info
        resource_info = self._resources[name]
        
        return resource_info.initialized

    @contextlib.contextmanager
    def resource_context(self, name: str):
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
            # Initialize resource
            resource = self.initialize(name)
            
            # Yield resource
            yield resource
            
        finally:
            # Clean up resource
            self.cleanup(name)
