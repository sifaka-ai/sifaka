# Circular Dependencies Resolution Plan

This document outlines specific recommendations for resolving circular dependencies in the Sifaka codebase.

## Priority Circular Dependencies

Based on the dependency analysis, the following circular dependencies should be addressed first:

### 1. Model Component Circular Dependencies

The model component has several circular dependencies:

```
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.core -> sifaka.models.managers.client -> sifaka.models.base
sifaka.models.base -> sifaka.models.providers.anthropic -> sifaka.models.core -> sifaka.models.services.generation -> sifaka.models.base
sifaka.models.base -> sifaka.models.providers.openai -> sifaka.models.base
```

**Resolution Strategy:**
1. Move all interfaces to `sifaka.interfaces.model`
2. Use string type annotations for forward references
3. Implement lazy loading for provider imports in factory functions
4. Restructure the model component to have a cleaner dependency hierarchy

### 2. Configuration Circular Dependencies

The configuration system has circular dependencies:

```
sifaka.utils.config -> sifaka.models.config -> sifaka.utils.config
sifaka.utils.config -> sifaka.chain.config -> sifaka.utils.config
sifaka.utils.config -> sifaka.classifiers.config -> sifaka.utils.config
sifaka.utils.config -> sifaka.critics.models -> sifaka.utils.config
```

**Resolution Strategy:**
1. Consolidate all configuration classes in `sifaka.utils.config`
2. Use string type annotations for forward references
3. Move component-specific configuration to their respective modules
4. Use composition instead of inheritance for configuration classes

### 3. Rules Component Circular Dependencies

The rules component has circular dependencies:

```
sifaka.utils.text -> sifaka.rules.base -> sifaka.utils.text
sifaka.rules.base -> sifaka.rules.formatting.length -> sifaka.rules.base
```

**Resolution Strategy:**
1. Move all interfaces to `sifaka.interfaces.rule`
2. Use string type annotations for forward references
3. Implement lazy loading for rule imports in factory functions
4. Restructure the rules component to have a cleaner dependency hierarchy

## General Resolution Strategies

### 1. Use Type Hints with String Literals

Replace direct imports in type annotations with string literals:

```python
# Before
from sifaka.models.base import ModelProvider

class ModelService:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

# After
class ModelService:
    def __init__(self, provider: "sifaka.interfaces.model.ModelProviderProtocol"):
        self.provider = provider
```

### 2. Use TYPE_CHECKING for Forward References

Use the `TYPE_CHECKING` constant from typing to conditionally import types:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sifaka.models.base import ModelProvider

class ModelService:
    def __init__(self, provider: "ModelProvider"):
        self.provider = provider
```

### 3. Implement Lazy Loading

Use lazy loading for imports in factory functions:

```python
# Before
from sifaka.models.providers.openai import OpenAIProvider

def create_model_provider(provider_type: str, **kwargs):
    if provider_type == "openai":
        return OpenAIProvider(**kwargs)
    # ...

# After
def create_model_provider(provider_type: str, **kwargs):
    if provider_type == "openai":
        from sifaka.models.providers.openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    # ...
```

### 4. Move Interface Definitions

Move interface definitions to dedicated interface modules:

```python
# Before (in sifaka/models/base.py)
class ModelProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...

# After (in sifaka/interfaces/model.py)
class ModelProviderProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        ...
```

### 5. Use Dependency Injection

Use dependency injection to avoid hard-coded dependencies:

```python
# Before
class Critic:
    def __init__(self, config):
        self.config = config
        self.model = OpenAIProvider()  # Hard-coded dependency

# After
class Critic:
    def __init__(self, config, model_provider):
        self.config = config
        self.model = model_provider  # Injected dependency
```

## Implementation Plan

### Phase 1: Interface Consolidation

1. Ensure all interfaces are in the `sifaka.interfaces` package
2. Update imports to use the consolidated interfaces
3. Use string type annotations for forward references

### Phase 2: Factory Function Refactoring

1. Implement lazy loading in factory functions
2. Standardize factory function patterns
3. Add validation for required dependencies

### Phase 3: Component Restructuring

1. Restructure the model component
2. Restructure the rules component
3. Restructure the critics component
4. Restructure the chain component

### Phase 4: Dependency Injection Enhancement

1. Enhance the DependencyProvider implementation
2. Add support for scoped dependencies
3. Improve error handling and logging
4. Implement dependency resolution strategies

## Specific File Changes

### sifaka/models/base.py

```python
# Before
from sifaka.models.providers.openai import OpenAIProvider

# After
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sifaka.models.providers.openai import OpenAIProvider
```

### sifaka/models/factories.py

```python
# Before
from sifaka.models.providers.openai import OpenAIProvider
from sifaka.models.providers.anthropic import AnthropicProvider

def create_model_provider(provider_type: str, **kwargs):
    if provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        return AnthropicProvider(**kwargs)
    # ...

# After
def create_model_provider(provider_type: str, **kwargs):
    if provider_type == "openai":
        from sifaka.models.providers.openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        from sifaka.models.providers.anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)
    # ...
```

### sifaka/utils/config.py

```python
# Before
from sifaka.models.config import ModelConfig

# After
# Define ModelConfig here or use string type annotations
```

## Success Criteria

1. No circular dependencies in the codebase
2. Consistent dependency injection patterns across all components
3. Improved error handling for missing dependencies
4. Comprehensive documentation for dependency management
5. All components use explicit dependency injection
6. Factory functions follow standardized patterns
7. Component initialization is standardized
8. Tests validate proper dependency injection
