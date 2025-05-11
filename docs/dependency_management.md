# Dependency Management Guidelines

This document provides guidelines for dependency management in the Sifaka codebase.

## Overview

Dependency management is a critical aspect of software engineering that affects the maintainability, testability, and extensibility of a codebase. The Sifaka framework uses a standardized approach to dependency management that ensures consistent behavior across all components.

## Key Concepts

### 1. Dependency Injection

Dependency injection is a design pattern that allows a class to receive its dependencies from external sources rather than creating them itself. This promotes loose coupling, making the code more modular and easier to test.

### 2. Dependency Provider

The `DependencyProvider` class in `sifaka/core/dependency.py` is a singleton that serves as a central registry for dependencies. It allows components to register and retrieve dependencies by name or type.

### 3. Scoped Dependencies

Dependencies can have different scopes, which determine their lifecycle:

- **Singleton**: One instance per application
- **Session**: One instance per session
- **Request**: One instance per request
- **Transient**: New instance each time

### 4. Factory Functions

Factory functions are used to create components with their dependencies. They provide a consistent way to create and configure components.

## Guidelines

### 1. Use Dependency Injection

Always use dependency injection to make dependencies explicit and avoid hard-coded dependencies:

```python
# Bad
class Critic:
    def __init__(self, config):
        self.config = config
        self.model = OpenAIProvider()  # Hard-coded dependency

# Good
class Critic:
    def __init__(self, config, model_provider):
        self.config = config
        self.model = model_provider  # Injected dependency
```

### 2. Use the DependencyProvider

Use the `DependencyProvider` to register and retrieve dependencies:

```python
from sifaka.core.dependency import DependencyProvider

# Register a dependency
provider = DependencyProvider()
provider.register("model", OpenAIProvider("gpt-4"))

# Retrieve a dependency
model = provider.get("model")
```

### 3. Use the inject_dependencies Decorator

Use the `inject_dependencies` decorator to automatically inject dependencies into classes and functions:

```python
from sifaka.core.dependency import inject_dependencies

@inject_dependencies
class Critic:
    def __init__(self, model=None, validator=None):
        self.model = model
        self.validator = validator

# Create an instance with dependencies injected
critic = Critic()  # Dependencies automatically injected
```

### 4. Use Factory Functions

Use factory functions to create components with their dependencies:

```python
def create_critic(
    critic_type: str,
    model_provider: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a critic.

    Args:
        critic_type: The critic type
        model_provider: The model provider to use
        **kwargs: Additional keyword arguments

    Returns:
        A critic instance
    """
    # Get dependency provider
    provider = DependencyProvider()

    # Resolve dependencies
    if model_provider is None:
        try:
            model_provider = provider.get("model_provider")
        except DependencyError:
            raise ValueError("Model provider is required")

    # Create critic based on type
    if critic_type == "prompt":
        from sifaka.critics.implementations.prompt import PromptCritic
        return PromptCritic(
            model_provider=model_provider,
            **kwargs,
        )
    # ...
```

### 5. Use Type Annotations

Use type annotations to make dependencies explicit and enable static type checking:

```python
from typing import Optional
from sifaka.interfaces import Model, Validator

class Critic:
    def __init__(
        self,
        model_provider: Model,
        validator: Optional[Validator] = None,
    ):
        self.model = model_provider
        self.validator = validator
```

### 6. Use String Type Annotations for Forward References

Use string type annotations for forward references to avoid circular imports:

```python
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sifaka.interfaces import Model, Validator

class Critic:
    def __init__(
        self,
        model_provider: "Model",
        validator: Optional["Validator"] = None,
    ):
        self.model = model_provider
        self.validator = validator
```

### 7. Use Lazy Loading

Use lazy loading for imports in factory functions to avoid circular imports:

```python
def create_model_provider(provider_type: str, **kwargs):
    if provider_type == "openai":
        from sifaka.models.providers.openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    # ...
```

### 8. Validate Dependencies

Validate dependencies to ensure they are present and valid:

```python
def _validate_dependencies(self) -> None:
    """
    Validate the critic dependencies.

    This method validates the critic dependencies, ensuring that
    all required dependencies are present and valid.

    Raises:
        DependencyError: If a required dependency is missing or invalid
    """
    # Validate model
    model = self._state_manager.get("model")
    if model is None:
        raise DependencyError("Model provider is required")

    # Validate prompt factory
    prompt_factory = self._state_manager.get("prompt_factory")
    if prompt_factory is None:
        raise DependencyError("Prompt factory is required")
```

### 9. Use State Management

Use state management to store and retrieve dependencies:

```python
class Critic(InitializableMixin):
    """Base critic implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        prompt_factory: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the critic.

        Args:
            name: The critic name
            description: The critic description
            llm_provider: The language model provider
            prompt_factory: Optional prompt factory
            config: Optional critic configuration
            **kwargs: Additional critic parameters
        """
        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        # Store dependencies in state
        self._state_manager.update("model", llm_provider)
        self._state_manager.update("prompt_factory", prompt_factory or self._create_default_prompt_factory())
```

### 10. Use Resource Management

Use resource management to initialize and release resources:

```python
def _initialize_resources(self) -> None:
    """
    Initialize the provider resources.

    This method initializes the provider resources, preparing them for use.
    It creates a client for the provider.

    Raises:
        InitializationError: If resource initialization fails
    """
    try:
        # Create client
        self._state_manager.update("client", self._create_client())
        self._state_manager.update("resources_initialized", True)
    except Exception as e:
        logger.error(f"Error initializing resources for {self.__class__.__name__}: {str(e)}")
        raise InitializationError(f"Error initializing resources for {self.__class__.__name__}: {str(e)}") from e

def _release_resources(self) -> None:
    """
    Release the provider resources.

    This method releases the provider resources, cleaning up after use.
    It closes the client for the provider.

    Raises:
        CleanupError: If resource cleanup fails
    """
    try:
        # Close client
        client = self._state_manager.get("client")
        if client is not None and hasattr(client, "close"):
            client.close()
        self._state_manager.update("client", None)
        self._state_manager.update("resources_initialized", False)
    except Exception as e:
        logger.error(f"Error releasing resources for {self.__class__.__name__}: {str(e)}")
        raise CleanupError(f"Error releasing resources for {self.__class__.__name__}: {str(e)}") from e
```

## Best Practices

1. **Make Dependencies Explicit**: Always make dependencies explicit in constructors.
2. **Use Dependency Injection**: Inject dependencies rather than creating them inside components.
3. **Use Factory Functions**: Use factory functions to create components with their dependencies.
4. **Validate Dependencies**: Validate dependencies to ensure they are present and valid.
5. **Use Type Annotations**: Use type annotations to make dependencies explicit.
6. **Use String Type Annotations**: Use string type annotations for forward references.
7. **Use Lazy Loading**: Use lazy loading for imports in factory functions.
8. **Use State Management**: Use state management to store and retrieve dependencies.
9. **Use Resource Management**: Use resource management to initialize and release resources.
10. **Document Dependencies**: Document dependencies in docstrings.

## Examples

### Example 1: Creating a Component with Dependencies

```python
from sifaka.core.dependency import DependencyProvider
from sifaka.models.providers.openai import OpenAIProvider
from sifaka.critics.implementations.prompt import PromptCritic

# Register dependencies
provider = DependencyProvider()
provider.register("model", OpenAIProvider("gpt-4"))

# Create component with dependencies
critic = PromptCritic(
    name="my_critic",
    description="My critic",
    llm_provider=provider.get("model"),
)
```

### Example 2: Using the inject_dependencies Decorator

```python
from sifaka.core.dependency import DependencyProvider, inject_dependencies
from sifaka.models.providers.openai import OpenAIProvider

# Register dependencies
provider = DependencyProvider()
provider.register("model", OpenAIProvider("gpt-4"))

# Create component with dependencies injected
@inject_dependencies
class MyCritic:
    def __init__(self, model=None):
        self.model = model

critic = MyCritic()  # model is automatically injected
```

### Example 3: Using Factory Functions

```python
from sifaka.core.factories import create_critic
from sifaka.models.providers.openai import OpenAIProvider

# Create model provider
model = OpenAIProvider("gpt-4")

# Create critic using factory function
critic = create_critic(
    critic_type="prompt",
    model_provider=model,
    system_prompt="You are an expert editor.",
)
```

## Conclusion

Following these guidelines will help ensure consistent dependency management across the Sifaka codebase, making it more maintainable, testable, and extensible.
