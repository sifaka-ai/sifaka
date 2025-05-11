# Comprehensive Plan to Eliminate Circular Dependencies in Sifaka

This plan outlines a strategy to completely eliminate circular dependencies in the Sifaka codebase without maintaining backward compatibility.

## Benefits for REVIEW.md Scores

Implementing this plan will significantly improve scores in REVIEW.md:

- **Maintainability (↑15-20 points)**: Cleaner dependency structure, reduced complexity
- **Consistency (↑10-15 points)**: Standardized interfaces and import patterns
- **Software Engineering Practices (↑15-20 points)**: Proper use of interfaces and dependency management
- **Extensibility (↑10-15 points)**: Cleaner interfaces for easier extension
- **Documentation (↑5-10 points)**: Clearer code structure improves documentation

Total potential improvement: 55-80 points across all categories.

## Implementation Progress

### Phase 1: Interface Consolidation (No Backward Compatibility) - PARTIALLY COMPLETED

1. **Move ALL Interface Definitions to the interfaces Package** - PARTIALLY COMPLETED
   - ✅ Moved LanguageModel protocol from `models/base.py` to `interfaces/model.py`
   - ⏳ Need to move remaining protocol definitions from `rules/base.py` to `interfaces/rule.py`
   - ⏳ Need to move protocol definitions from other components to their respective interface files

2. **Standardize Interface Naming** - PARTIALLY COMPLETED
   - ✅ Using consistent naming pattern: `{Component}Protocol` (e.g., `ModelProviderProtocol`)
   - ⏳ Need to remove any remaining legacy interface names or aliases

### Phase 2: Component Restructuring (No Backward Compatibility) - IN PROGRESS

1. **Restructure the Models Component** - IN PROGRESS
   - ✅ Created clean hierarchy: interfaces → base → core → providers
   - ✅ Updated models/base.py to use interfaces from interfaces/model.py
   - ✅ Updated models/core.py to use interfaces from interfaces/model.py
   - ✅ Updated models/providers/openai.py to use interfaces directly
   - ✅ Updated models/providers/anthropic.py to use interfaces directly
   - ⏳ Need to fix remaining self-referential imports within modules

2. **Restructure the Rules Component** - NOT STARTED
   - ⏳ Need to create a clean hierarchy: interfaces → base → implementations
   - ⏳ Need to move utility functions from `rules/base.py` to appropriate utility modules
   - ⏳ Need to ensure rule implementations only import from base and interfaces

3. **Restructure the Configuration System** - NOT STARTED
   - ⏳ Need to consolidate all configuration classes in `utils/config.py`
   - ⏳ Need to remove component-specific config modules that create circular dependencies
   - ⏳ Need to use composition instead of inheritance for configuration classes

## Remaining Circular Dependencies in Models Component

1. **Self-referential Imports**:
   - models/providers/openai.py -> models/providers/openai.py
   - models/providers/anthropic.py -> models/providers/anthropic.py
   - models/base.py -> models/base.py

2. **Cross-component Circular Dependencies**:
   - models/base.py -> models/providers/anthropic.py -> models/base.py

## Next Steps for Models Component

1. **Fix Self-referential Imports**:
   - Identify and remove self-imports in models/providers/openai.py
   - Identify and remove self-imports in models/providers/anthropic.py
   - Identify and remove self-imports in models/base.py

2. **Fix Cross-component Circular Dependencies**:
   - Ensure models/base.py doesn't import from models/providers/anthropic.py
   - Ensure models/providers/anthropic.py doesn't import from models/base.py
   - Use interfaces and dependency injection to break circular dependencies

3. **Implement Factory Functions**:
   - Create a dedicated models/factories.py module
   - Move all model provider creation logic to factory functions
   - Use lazy loading in factory functions to avoid circular dependencies

## Next Steps for Rules Component

1. **Move Interface Definitions**:
   - Move all protocol definitions from rules/base.py to interfaces/rule.py
   - Update rules/base.py to use interfaces from interfaces/rule.py

2. **Fix Circular Dependencies**:
   - Fix circular dependency between rules/base.py and rules/formatting/length.py
   - Fix circular dependency between rules/base.py and utils/text.py

3. **Implement Factory Functions**:
   - Create a dedicated rules/factories.py module
   - Move all rule creation logic to factory functions
   - Use lazy loading in factory functions to avoid circular dependencies

## Next Steps for Configuration System

1. **Consolidate Configuration Classes**:
   - Move all configuration classes to utils/config.py
   - Remove component-specific config modules

2. **Fix Circular Dependencies**:
   - Fix circular dependency between utils/config.py and models/config.py
   - Fix circular dependency between utils/config.py and chain/config.py
   - Fix circular dependency between utils/config.py and classifiers/config.py
   - Fix circular dependency between utils/config.py and critics/models.py

3. **Use Composition**:
   - Use composition instead of inheritance for configuration classes
   - Implement factory functions for creating configuration objects

## Implementation Details for Key Files

### 1. models/providers/openai.py - Fix Self-referential Imports

```python
"""
OpenAI model provider implementation.
"""

# Import interfaces directly
from sifaka.interfaces.client import APIClientProtocol
from sifaka.interfaces.counter import TokenCounterProtocol
from sifaka.interfaces.model import ModelProviderProtocol
from sifaka.utils.config import ModelConfig

# Implementation classes
class OpenAIClient(APIClientProtocol):
    """OpenAI API client implementation."""
    # Implementation

class OpenAITokenCounter(TokenCounterProtocol):
    """Token counter for OpenAI models."""
    # Implementation

class OpenAIProvider(ModelProviderProtocol):
    """OpenAI model provider implementation."""
    # Implementation
```

### 2. models/providers/anthropic.py - Fix Self-referential Imports

```python
"""
Anthropic model provider implementation.
"""

# Import interfaces directly
from sifaka.interfaces.client import APIClientProtocol
from sifaka.interfaces.counter import TokenCounterProtocol
from sifaka.interfaces.model import ModelProviderProtocol
from sifaka.utils.config import ModelConfig

# Implementation classes
class AnthropicClient(APIClientProtocol):
    """Anthropic API client implementation."""
    # Implementation

class AnthropicTokenCounter(TokenCounterProtocol):
    """Token counter for Anthropic models."""
    # Implementation

class AnthropicProvider(ModelProviderProtocol):
    """Anthropic model provider implementation."""
    # Implementation
```

### 3. models/factories.py - Implement Factory Functions

```python
"""
Factory functions for creating model providers.
"""

from typing import Optional, Type, TypeVar, cast

from sifaka.interfaces.client import APIClientProtocol
from sifaka.interfaces.counter import TokenCounterProtocol
from sifaka.interfaces.model import ModelProviderProtocol
from sifaka.utils.config import ModelConfig

T = TypeVar("T", bound=ModelProviderProtocol)

def create_model_provider(
    provider_class: Type[T],
    model_name: str,
    config: Optional[ModelConfig] = None,
    api_client: Optional[APIClientProtocol] = None,
    token_counter: Optional[TokenCounterProtocol] = None,
) -> T:
    """
    Create a model provider instance.

    This factory function creates a model provider instance with the specified
    parameters. It uses lazy loading to avoid circular dependencies.

    Args:
        provider_class: The model provider class to instantiate
        model_name: The name of the model to use
        config: Optional model configuration
        api_client: Optional API client to use
        token_counter: Optional token counter to use

    Returns:
        An instance of the specified model provider class
    """
    # Create the provider instance
    provider = provider_class(
        model_name=model_name,
        config=config,
        api_client=api_client,
        token_counter=token_counter,
    )

    return cast(T, provider)
```
