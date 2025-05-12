from typing import Any, List
"""
Dependency Injection Module

This module provides a dependency injection system for Sifaka components.
It allows components to request dependencies by name or type, and supports
different dependency scopes (singleton, session, request, transient).

## Components
- **DependencyProvider**: Singleton class for registering and retrieving dependencies
- **DependencyScope**: Enum defining dependency lifecycles
- **SessionScope**: Context manager for session-scoped dependencies
- **RequestScope**: Context manager for request-scoped dependencies
- **DependencyInjector**: Utility class for manual dependency injection
- **inject_dependencies**: Decorator for automatic dependency injection

## Usage Examples
```python
from sifaka.core.dependency.provider import DependencyProvider
from sifaka.core.dependency.scopes import DependencyScope
from sifaka.core.dependency.injector import inject_dependencies
from sifaka.core.dependency.utils import provide_dependency, get_dependency

# Register dependencies
provide_dependency("model", OpenAIModel(), scope=DependencyScope.SINGLETON)
provide_dependency("validator", LengthValidator(), scope=DependencyScope.REQUEST)

# Get dependencies
model = get_dependency("model")
validator = get_dependency("validator")

# Use dependency injection
@inject_dependencies
class MyComponent:
    def __init__(self, model=None, validator=None):
        self.model = model  # Injected from DependencyProvider
        self.validator = validator  # Injected from DependencyProvider

# Create an instance with dependencies injected
component = MyComponent()  # Dependencies automatically injected
```

## Error Handling
- Raises DependencyError for missing dependencies
- Raises ConfigurationError for circular dependencies
"""
from .provider import DependencyProvider
from .scopes import DependencyScope, SessionScope, RequestScope
from .injector import DependencyInjector, inject_dependencies
from .utils import provide_dependency, provide_factory, get_dependency, get_dependency_by_type, create_session_scope, create_request_scope, clear_dependencies
from sifaka.utils.errors.base import DependencyError, ConfigurationError
__all__: List[Any] = ['DependencyScope', 'DependencyProvider',
    'SessionScope', 'RequestScope', 'DependencyInjector',
    'inject_dependencies', 'provide_dependency', 'provide_factory',
    'get_dependency', 'get_dependency_by_type', 'create_session_scope',
    'create_request_scope', 'clear_dependencies', 'DependencyError',
    'ConfigurationError']
