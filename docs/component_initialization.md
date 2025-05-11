# Component Initialization Guidelines

This document provides guidelines for component initialization in the Sifaka codebase.

## Overview

Component initialization is a critical aspect of software engineering that affects the reliability, maintainability, and extensibility of a codebase. The Sifaka framework uses a standardized approach to component initialization that ensures consistent behavior across all components.

## Key Concepts

### 1. Initialization Lifecycle

The initialization lifecycle of a Sifaka component consists of the following phases:

1. **Creation**: The component is instantiated with its basic parameters
2. **Configuration Validation**: The component's configuration is validated
3. **Dependency Validation**: The component's dependencies are validated
4. **Resource Registration**: The component's resources are registered
5. **Resource Initialization**: The component's resources are initialized
6. **Warm-up**: The component is prepared for use
7. **Cleanup**: The component's resources are released

### 2. InitializableMixin

The `InitializableMixin` class in `sifaka/core/initialization.py` provides a standardized implementation of the initialization lifecycle. It includes methods for:

- Validating configuration
- Validating dependencies
- Registering resources
- Initializing resources
- Warming up resources
- Cleaning up resources

### 3. Resource Management

The `ResourceManager` class in `sifaka/utils/resources.py` provides a standardized way to manage component resources. It includes methods for:

- Registering resources
- Initializing resources
- Getting initialized resources
- Cleaning up resources

### 4. State Management

The `StateManager` class in `sifaka/utils/state.py` provides a standardized way to manage component state. It includes methods for:

- Updating state
- Getting state
- Setting metadata
- Getting metadata
- Resetting state

## Guidelines

### 1. Use InitializableMixin

Use the `InitializableMixin` class to ensure consistent initialization behavior:

```python
from sifaka.core.initialization import InitializableMixin

class MyComponent(InitializableMixin):
    def __init__(self, name, description, config=None):
        super().__init__(name, description, config)
```

### 2. Use BaseInitializable

For Pydantic-based components, use the `BaseInitializable` class:

```python
from sifaka.core.initialization import BaseInitializable
from pydantic import Field

class MyComponent(BaseInitializable):
    # Configuration
    config: MyConfig = Field(description="Component configuration")
```

### 3. Override Lifecycle Methods

Override the lifecycle methods to implement component-specific behavior:

```python
def _validate_configuration(self) -> None:
    # Validate component-specific configuration
    if not self.config.required_param:
        raise ValueError("Required parameter missing")

def _validate_dependencies(self) -> None:
    # Validate component-specific dependencies
    if not hasattr(self, "_required_dependency"):
        raise ValueError("Required dependency missing")

def _register_resources(self) -> None:
    # Register component-specific resources
    self._resource_manager.register(
        "database",
        DatabaseResource(self.config.connection_string),
        required=True
    )
```

### 4. Use Resource Manager

Use the `ResourceManager` to manage component resources:

```python
from sifaka.utils.resources import Resource

class DatabaseResource(Resource):
    def initialize(self):
        # Initialize the resource
        self.connection = connect_to_database()
        return self.connection
        
    def cleanup(self):
        # Clean up the resource
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

# In your component
def _register_resources(self) -> None:
    self._resource_manager.register(
        "database",
        DatabaseResource(self.config.connection_string),
        required=True
    )
```

### 5. Use State Manager

Use the `StateManager` to manage component state:

```python
# Update state
self._state_manager.update("processing", True)

# Get state
is_processing = self._state_manager.get("processing", False)

# Set metadata
self._state_manager.set_metadata("last_processed", time.time())

# Get metadata
last_processed = self._state_manager.get_metadata("last_processed")
```

## Examples

### Example 1: Basic Component

```python
from sifaka.core.initialization import InitializableMixin
from sifaka.utils.resources import Resource

class APIClientResource(Resource):
    def __init__(self, api_key):
        self.api_key = api_key
        
    def initialize(self):
        self.client = APIClient(self.api_key)
        return self.client
        
    def cleanup(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

class MyComponent(InitializableMixin):
    def __init__(self, name, description, config=None):
        super().__init__(name, description, config)
        
    def _validate_configuration(self):
        if not self.config.get("api_key"):
            raise ValueError("API key is required")
            
    def _register_resources(self):
        self._resource_manager.register(
            "api_client",
            APIClientResource(self.config["api_key"]),
            required=True
        )
        
    def process(self, data):
        # Get initialized resource
        client = self._resource_manager.get("api_client")
        return client.process(data)
```

### Example 2: Pydantic-based Component

```python
from pydantic import BaseModel, Field
from sifaka.core.initialization import BaseInitializable
from sifaka.utils.resources import Resource

class DatabaseResource(Resource):
    def __init__(self, connection_string):
        self.connection_string = connection_string
        
    def initialize(self):
        self.connection = connect_to_database(self.connection_string)
        return self.connection
        
    def cleanup(self):
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

class MyConfig(BaseModel):
    connection_string: str = Field(description="Database connection string")
    timeout: int = Field(default=30, description="Timeout in seconds")

class MyComponent(BaseInitializable):
    # Configuration
    config: MyConfig = Field(description="Component configuration")
    
    def _validate_configuration(self):
        # Configuration is already validated by Pydantic
        pass
        
    def _register_resources(self):
        self._resource_manager.register(
            "database",
            DatabaseResource(self.config.connection_string),
            required=True
        )
        
    def query(self, sql):
        # Get initialized resource
        db = self._resource_manager.get("database")
        return db.execute(sql)
```

## Best Practices

1. **Make Dependencies Explicit**: Always make dependencies explicit in constructors.
2. **Validate Configuration**: Always validate configuration in `_validate_configuration`.
3. **Validate Dependencies**: Always validate dependencies in `_validate_dependencies`.
4. **Register Resources**: Always register resources in `_register_resources`.
5. **Clean Up Resources**: Always clean up resources in `cleanup`.
6. **Use Resource Manager**: Use the resource manager for resource lifecycle management.
7. **Use State Manager**: Use the state manager for state management.
8. **Document Lifecycle Methods**: Document the lifecycle methods in docstrings.
9. **Handle Errors Gracefully**: Handle initialization and cleanup errors gracefully.
10. **Test Lifecycle Methods**: Test the lifecycle methods to ensure proper behavior.
