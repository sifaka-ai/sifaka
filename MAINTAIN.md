# Sifaka Maintainability Improvement Plan

This document outlines recommendations for improving the maintainability of the Sifaka codebase, providing concrete examples and implementation strategies for each recommendation.

## Recommendation 1: Simplify the Dependency Injection System
**Priority: 1/5 | Estimated Effort: High | Status: Implemented ✅**

The current dependency injection system is spread across multiple files and uses several different approaches to component registration. This complexity makes the codebase harder to understand and maintain. We should simplify the system while preserving its benefits.

### Current Issues:
1. The registry system is split between `registry.py` and `initialize_registry.py`
2. Multiple approaches to component registration (decorators, direct registration, imports)
3. Circular dependencies are being worked around rather than eliminated
4. Redundant factory functions in different modules

### Implementation:
- Consolidated registry functionality into a single `registry.py` module
- Standardized component registration using a decorator pattern
- Implemented proper lazy loading to eliminate circular dependencies
- Created a comprehensive documentation file at `docs/architecture/dependency_injection_new.md`

### Implementation Steps:

#### 1. Consolidate Registry Functionality

Merge `registry.py` and `initialize_registry.py` into a single `registry.py` module with a cleaner API:

```python
# sifaka/registry.py
"""
Registry system for Sifaka components.

This module provides a central registry for component registration and retrieval,
eliminating circular import issues through lazy loading.
"""

import importlib
import logging
from typing import Dict, Any, Callable, Optional, TypeVar, List, Set

# Type variables
T = TypeVar("T")
ComponentFactory = Callable[..., T]

# Logger
logger = logging.getLogger(__name__)

class Registry:
    """Singleton registry for component management."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Registry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the registry."""
        self._components: Dict[str, Dict[str, ComponentFactory]] = {}
        self._initialized_types: Set[str] = set()

        # Define lazy imports
        self._lazy_imports: Dict[str, List[str]] = {
            "model": ["sifaka.models.openai", "sifaka.models.anthropic", "sifaka.models.gemini"],
            "validator": ["sifaka.validators.length", "sifaka.validators.content"],
            "improver": ["sifaka.improvers.clarity", "sifaka.improvers.factual"]
        }

    def register(self, component_type: str, name: str, factory: ComponentFactory) -> ComponentFactory:
        """Register a component factory."""
        if component_type not in self._components:
            self._components[component_type] = {}

        self._components[component_type][name] = factory
        return factory

    def get(self, component_type: str, name: str) -> Optional[ComponentFactory]:
        """Get a component factory."""
        self._ensure_initialized(component_type)

        if component_type not in self._components:
            return None

        return self._components[component_type].get(name)

    def _ensure_initialized(self, component_type: str) -> None:
        """Ensure component type is initialized."""
        if component_type in self._initialized_types:
            return

        if component_type in self._lazy_imports:
            for module_name in self._lazy_imports[component_type]:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    logger.debug(f"Could not import {module_name}")

        self._initialized_types.add(component_type)

# Singleton instance
_registry = Registry()

# Public API
def register(component_type: str, name: str, factory: ComponentFactory = None):
    """Register a component factory."""
    if factory is None:
        # Used as a decorator
        def decorator(f):
            return _registry.register(component_type, name, f)
        return decorator

    # Used as a function
    return _registry.register(component_type, name, factory)

def get(component_type: str, name: str) -> Optional[ComponentFactory]:
    """Get a component factory."""
    return _registry.get(component_type, name)

# Convenience functions for common component types
def register_model(name: str, factory: ComponentFactory = None):
    """Register a model factory."""
    return register("model", name, factory)

def get_model(name: str) -> Optional[ComponentFactory]:
    """Get a model factory."""
    return get("model", name)

def register_validator(name: str, factory: ComponentFactory = None):
    """Register a validator factory."""
    return register("validator", name, factory)

def get_validator(name: str) -> Optional[ComponentFactory]:
    """Get a validator factory."""
    return get("validator", name)

def register_improver(name: str, factory: ComponentFactory = None):
    """Register an improver factory."""
    return register("improver", name, factory)

def get_improver(name: str) -> Optional[ComponentFactory]:
    """Get an improver factory."""
    return get("improver", name)
```

#### 2. Standardize Component Registration

Use a single, consistent approach to component registration using decorators:

```python
# Example: Registering a model
from sifaka.registry import register_model

@register_model("openai")
def create_openai_model(model_name, **options):
    return OpenAIModel(model_name, **options)

# Example: Registering a validator
from sifaka.registry import register_validator

@register_validator("length")
def create_length_validator(min_length=None, max_length=None, **options):
    return LengthValidator(min_length, max_length, **options)
```

#### 3. Eliminate Redundant Factory Functions

Replace redundant factory functions with a single implementation in the factories module:

```python
# sifaka/factories.py
"""
Factory functions for creating Sifaka components.
"""

from typing import Any, Optional

from sifaka.interfaces import Model, Validator, Improver
from sifaka.registry import get_model, get_validator, get_improver

class FactoryError(Exception):
    """Error raised when a factory function fails."""
    pass

def create_model(provider: str, model_name: str, **options: Any) -> Model:
    """Create a model instance."""
    factory = get_model(provider)
    if factory is None:
        raise FactoryError(f"Model provider '{provider}' not found")

    try:
        return factory(model_name, **options)
    except Exception as e:
        raise FactoryError(f"Error creating model: {str(e)}") from e

def create_validator(name: str, **options: Any) -> Validator:
    """Create a validator instance."""
    factory = get_validator(name)
    if factory is None:
        raise FactoryError(f"Validator '{name}' not found")

    try:
        return factory(**options)
    except Exception as e:
        raise FactoryError(f"Error creating validator: {str(e)}") from e

def create_improver(name: str, model, **options: Any) -> Improver:
    """Create an improver instance."""
    factory = get_improver(name)
    if factory is None:
        raise FactoryError(f"Improver '{name}' not found")

    try:
        return factory(model, **options)
    except Exception as e:
        raise FactoryError(f"Error creating improver: {str(e)}") from e
```

#### 4. Redesign to Eliminate Circular Dependencies

Instead of working around circular dependencies, redesign the architecture to eliminate them:

1. **Use Interface-Based Design**: Define all interfaces in a single `interfaces.py` module
2. **Implement Dependency Inversion**: High-level modules should not depend on low-level modules
3. **Use Composition Over Inheritance**: Favor composition to reduce tight coupling

Example of improved Chain class using dependency inversion:

```python
# sifaka/chain.py
from typing import Optional, List, Dict, Any

from sifaka.interfaces import Model, Validator, Improver
from sifaka.factories import create_model

class Chain:
    """Main orchestrator for generation, validation, and improvement."""

    def __init__(self, model_factory=None):
        """Initialize a chain.

        Args:
            model_factory: Factory function for creating models.
                If None, the default factory will be used.
        """
        self._model: Optional[Model] = None
        self._prompt: Optional[str] = None
        self._validators: List[Validator] = []
        self._improvers: List[Improver] = []
        self._options: Dict[str, Any] = {}
        self._model_factory = model_factory or create_model

    def with_model(self, model):
        """Set the model to use."""
        if isinstance(model, str):
            if ":" in model:
                provider, model_name = model.split(":", 1)
                self._model = self._model_factory(provider, model_name)
            else:
                raise ValueError(f"Invalid model string: {model}")
        else:
            self._model = model
        return self

    # Rest of the implementation...
```

#### 5. Add Comprehensive Tests

Create tests specifically for the registry and dependency injection system:

```python
# tests/test_registry.py
import pytest
from sifaka.registry import register, get, register_model, get_model

def test_register_and_get():
    """Test registering and retrieving a component."""
    # Register a test component
    @register("test", "test_component")
    def create_test_component():
        return "test_component"

    # Get the component
    factory = get("test", "test_component")
    assert factory is not None
    assert factory() == "test_component"

def test_model_registration():
    """Test model registration."""
    # Register a test model
    @register_model("test")
    def create_test_model(model_name, **options):
        return f"Model: {model_name}"

    # Get the model factory
    factory = get_model("test")
    assert factory is not None
    assert factory("test_model") == "Model: test_model"
```

### Expected Benefits:

1. **Simplified Codebase**: Fewer files and a more straightforward API
2. **Improved Maintainability**: Consistent approach to component registration
3. **Better Testability**: Cleaner separation of concerns makes testing easier
4. **Reduced Complexity**: Elimination of circular dependencies
5. **Enhanced Extensibility**: Clearer extension points for new components

### Implementation Timeline:

1. **Week 1**: Consolidate registry functionality
2. **Week 2**: Standardize component registration
3. **Week 3**: Eliminate redundant factory functions
4. **Week 4**: Redesign to eliminate circular dependencies
5. **Week 5**: Add comprehensive tests and documentation

## Recommendation 2: Standardize Error Handling
**Priority: 2/5 | Estimated Effort: Medium**

The current error handling is inconsistent across the codebase. We should standardize error handling to improve maintainability and provide better error messages for users.

### Current Issues:
1. Inconsistent error types across modules
2. Some error messages lack context or actionable information
3. Error handling is sometimes mixed with business logic
4. Lack of centralized error documentation

### Implementation Steps:

#### 1. Create a Centralized Error Module

Create a single `errors.py` module with a hierarchy of error types:

```python
# sifaka/errors.py
"""
Error types for Sifaka.
"""

class SifakaError(Exception):
    """Base class for all Sifaka errors."""
    pass

# Registry errors
class RegistryError(SifakaError):
    """Error raised by the registry system."""
    pass

class ComponentNotFoundError(RegistryError):
    """Error raised when a component is not found."""
    pass

# Factory errors
class FactoryError(SifakaError):
    """Error raised by factory functions."""
    pass

# Model errors
class ModelError(SifakaError):
    """Error raised by model operations."""
    pass

class ModelNotFoundError(ModelError):
    """Error raised when a model is not found."""
    pass

class ModelAPIError(ModelError):
    """Error raised when a model API call fails."""
    pass

# Chain errors
class ChainError(SifakaError):
    """Error raised by chain operations."""
    pass

class ValidationError(ChainError):
    """Error raised during validation."""
    pass

class ImprovementError(ChainError):
    """Error raised during improvement."""
    pass
```

#### 2. Standardize Error Handling Patterns

Use consistent error handling patterns throughout the codebase:

```python
# Example: Consistent error handling in factory functions
def create_model(provider: str, model_name: str, **options: Any) -> Model:
    """Create a model instance."""
    try:
        factory = get_model(provider)
        if factory is None:
            raise ComponentNotFoundError(f"Model provider '{provider}' not found")

        return factory(model_name, **options)
    except ComponentNotFoundError:
        # Re-raise component not found errors
        raise
    except Exception as e:
        # Wrap other errors
        raise FactoryError(f"Error creating model: {str(e)}") from e
```

#### 3. Improve Error Messages

Make error messages more informative and actionable:

```python
# Before
raise ValueError("Invalid model string")

# After
raise ValueError(
    f"Invalid model string: '{model_string}'. "
    f"Expected format: 'provider:model_name' (e.g., 'openai:gpt-4')"
)
```

#### 4. Document Error Types

Add comprehensive documentation for error types:

```python
# In errors.py docstrings
class ModelNotFoundError(ModelError):
    """Error raised when a model is not found.

    This error is raised when:
    1. The specified model provider is not registered
    2. The specified model name is not supported by the provider

    Examples:
        ```python
        try:
            model = create_model("unknown", "model")
        except ModelNotFoundError as e:
            print(f"Model not found: {e}")
        ```
    """
    pass
```

### Expected Benefits:

1. **Improved Debugging**: Clearer error messages make debugging easier
2. **Better User Experience**: More actionable error messages for users
3. **Consistent Error Handling**: Standardized approach across the codebase
4. **Enhanced Documentation**: Better documentation of error types and handling

## Recommendation 3: Reduce Unnecessary Abstractions
**Priority: 3/5 | Estimated Effort: Medium**

The codebase contains some unnecessary layers of abstraction that add complexity without providing significant benefits. We should simplify these abstractions to improve maintainability.

### Current Issues:
1. Too many layers of abstraction in some areas
2. Redundant wrapper functions
3. Overly complex class hierarchies
4. Unnecessary indirection in some modules

### Implementation Steps:

#### 1. Flatten Class Hierarchies

Simplify class hierarchies by removing unnecessary intermediate classes:

```python
# Before
class BaseValidator:
    """Base class for validators."""
    pass

class TextValidator(BaseValidator):
    """Base class for text validators."""
    pass

class LengthValidator(TextValidator):
    """Validator for text length."""
    pass

# After
class LengthValidator:
    """Validator for text length."""
    pass
```

#### 2. Eliminate Redundant Wrapper Functions

Remove unnecessary wrapper functions that don't add value:

```python
# Before
def create_length_validator(min_length=None, max_length=None):
    """Create a length validator."""
    return _create_length_validator_impl(min_length, max_length)

def _create_length_validator_impl(min_length, max_length):
    """Implementation of create_length_validator."""
    return LengthValidator(min_length, max_length)

# After
def create_length_validator(min_length=None, max_length=None):
    """Create a length validator."""
    return LengthValidator(min_length, max_length)
```

#### 3. Simplify Module Structure

Consolidate related functionality into fewer modules:

```
# Before
sifaka/
├── validators/
│   ├── base.py
│   ├── text/
│   │   ├── base.py
│   │   ├── length.py
│   │   └── content.py
│   └── factory.py

# After
sifaka/
├── validators/
│   ├── length.py
│   ├── content.py
│   └── __init__.py
```

### Expected Benefits:

1. **Reduced Complexity**: Fewer layers of abstraction make the code easier to understand
2. **Improved Performance**: Less indirection can lead to better performance
3. **Enhanced Maintainability**: Simpler code is easier to maintain
4. **Better Developer Experience**: Easier to navigate and understand the codebase

## Next Steps

1. **Create Detailed Implementation Plan**: Develop a detailed plan for each recommendation
2. **Prioritize Changes**: Focus on high-impact, low-risk changes first
3. **Implement Changes Incrementally**: Make small, focused changes rather than large refactorings
4. **Add Tests**: Ensure all changes are covered by tests
5. **Update Documentation**: Keep documentation in sync with code changes