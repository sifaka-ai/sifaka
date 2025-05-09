# Sifaka Implementation Plan

This document outlines a structured approach to address the improvement areas identified in the code review, with specific actionable tasks, priorities, and timelines.

## 1. Eliminate Component Duplication

### High-Priority Tasks (1-2 weeks)
- ✅ Replace duplicate `StateManager` in classifiers with unified implementation from utils
- Audit the codebase for other duplicated components by running static analysis tools
- Create centralized component registry for common interfaces like validators, processors, and handlers

### Medium-Priority Tasks (2-4 weeks)
- Refactor critics and rules implementations to share common code where appropriate
- Create shared base classes for similar components like `TextProcessor` and `TextTransformer`
- Standardize error handling patterns across component types

### Implementation Approach
```python
# Example: Centralized component registry
from typing import Dict, Type, TypeVar, Generic

T = TypeVar('T')

class ComponentRegistry(Generic[T]):
    """Centralized registry for component types."""

    def __init__(self):
        self._components: Dict[str, Type[T]] = {}

    def register(self, name: str, component_type: Type[T]) -> None:
        """Register a component type."""
        self._components[name] = component_type

    def get(self, name: str) -> Type[T]:
        """Get a component type by name."""
        if name not in self._components:
            raise KeyError(f"Component '{name}' not registered")
        return self._components[name]
```

## 2. Standardize Factory Function Patterns

### High-Priority Tasks (1-2 weeks)
- Audit all factory functions and standardize naming convention (e.g., `create_*` vs `standardize_*_config`)
- Create consistent parameter ordering and default handling across all factory functions
- Implement typed factory pattern with clear return types

### Medium-Priority Tasks (2-4 weeks)
- Add builder pattern alternatives for complex object construction
- Document factory pattern guidelines for contributors
- Create test cases to verify factory function behavior

### Implementation Approach
```python
# Example: Standardized factory function template
from typing import Dict, Any, Type, TypeVar, Optional, cast

T = TypeVar('T', bound='BaseComponent')

def create_component(
    component_type: Type[T],
    name: str,
    description: str = "",
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> T:
    """
    Standardized factory function for component creation.

    Args:
        component_type: The type of component to create
        name: Component name
        description: Component description
        config: Optional configuration dictionary
        **kwargs: Additional component-specific parameters

    Returns:
        Instantiated component
    """
    merged_config = standardize_config(config, **kwargs)
    return cast(T, component_type(name=name, description=description, config=merged_config))
```

## 3. Reduce Complexity and Improve Learning Curve

### High-Priority Tasks (1-2 weeks)
- Create getting started guides for each major component type
- Develop interactive examples that demonstrate key concepts
- Simplify public API by hiding implementation details

### Medium-Priority Tasks (2-4 weeks)
- Create a simplified facade layer for common operations
- Develop cookbooks for common use cases
- Add progressive complexity in documentation (basic → advanced)

### Implementation Approach
```python
# Example: Simplified facade for common operations
from sifaka.facade import SifakaAPI

# Simple high-level API
api = SifakaAPI()

# Instead of complex setup:
# rule = ToxicityRule.create(name="toxicity", config=RuleConfig(...))
# critic = PromptCritic.create(name="refine", config=CriticConfig(...))
# chain = Chain.create(rules=[rule], critics=[critic], ...)

# Simple API:
result = api.validate_and_improve(
    text="Your text here",
    rules=["toxicity", "length"],
    improve_if_failed=True
)
```

## 4. Improve Test Coverage

### High-Priority Tasks (1-2 weeks)
- Add unit test templates for each component type
- Implement critical path integration tests
- Set up CI/CD pipeline with coverage reporting

### Medium-Priority Tasks (2-4 weeks)
- Add property-based testing for complex components
- Create test fixtures for common test scenarios
- Implement regression test suite for bug fixes

### Implementation Approach
```python
# Example: Test template for validators
import pytest
from sifaka.rules.base import BaseValidator, RuleResult

class TestBaseValidator:
    def test_validate_input_validation(self):
        # Test that validation properly checks input types
        validator = BaseValidator()
        with pytest.raises(ValueError):
            validator.validate(123)  # Non-string input should fail

    def test_empty_text_handling(self):
        # Test empty text handling
        validator = BaseValidator()
        result = validator.validate("")
        assert not result.passed
        assert "empty text" in result.message.lower()

    # Additional tests for specific validator behaviors
```

## 5. Standardize Pydantic Usage

### High-Priority Tasks (2-3 weeks)
- Audit all Pydantic model usage and document current version
- Create migration plan to Pydantic 2 if not already using it
- Update all models to use consistent Pydantic configuration

### Medium-Priority Tasks (3-5 weeks)
- Standardize validation error handling
- Create custom Pydantic types for common patterns
- Implement consistent serialization/deserialization patterns

### Implementation Approach
```python
# Example: Standardized Pydantic model configuration
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, Optional

class SifakaBaseModel(BaseModel):
    """Base model for all Sifaka models with standardized configuration."""

    model_config = ConfigDict(
        extra="forbid",  # Prevent extra attributes
        frozen=False,    # Allow mutation unless explicitly frozen
        validate_assignment=True,  # Validate on attribute assignment
        arbitrary_types_allowed=True,  # Allow complex types like callbacks
        populate_by_name=True,  # Allow populating by alias
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def with_metadata(self, **kwargs: Any) -> "SifakaBaseModel":
        """Create a new instance with updated metadata."""
        return self.model_copy(update={"metadata": {**self.metadata, **kwargs}})
```

## 6. Standardize State Management

### High-Priority Tasks (1-2 weeks)
- ✅ Consolidate on single StateManager implementation
- Create state management guidelines for developers
- Implement state introspection tools for debugging

### Medium-Priority Tasks (2-4 weeks)
- Add observability hooks to state changes
- Create state persistence patterns for long-running components
- Implement state sharing patterns for related components

### Implementation Approach
```python
# Example: Enhanced state management with observability
from typing import Any, Dict, List, Callable, Optional
from sifaka.utils.state import State

class ObservableStateManager:
    """State manager with change observers."""

    def __init__(self):
        self._state = State()
        self._history: List[State] = []
        self._observers: Dict[str, List[Callable[[str, Any, Any], None]]] = {}

    def update(self, key: str, value: Any) -> None:
        """Update state with history tracking and observer notification."""
        old_value = self._state.data.get(key)
        self._history.append(self._state)
        self._state = self._state.model_copy(update={"data": {**self._state.data, key: value}})

        # Notify observers
        if key in self._observers:
            for observer in self._observers[key]:
                observer(key, old_value, value)

    def observe(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """Add observer for a specific key."""
        if key not in self._observers:
            self._observers[key] = []
        self._observers[key].append(callback)
```

## 7. Standardize Factory Functions Implementation

### High-Priority Tasks (1-2 weeks)
- Create unified factory function implementation guidelines
- Standardize error handling in factory functions
- Implement consistent validation in factory functions

### Medium-Priority Tasks (2-4 weeks)
- Create factory function decorators for common patterns
- Implement factory registration system for component discovery
- Add factory function test helpers

### Implementation Approach
```python
# Example: Factory function decorator for standardization
from functools import wraps
from typing import TypeVar, Any, Callable, Dict, Optional, Type, cast

T = TypeVar('T')

def standardized_factory(config_class: Type[Any]):
    """Decorator for standardizing factory functions."""

    def decorator(factory_func: Callable[..., T]) -> Callable[..., T]:
        @wraps(factory_func)
        def wrapper(
            config: Optional[Dict[str, Any]] = None,
            params: Optional[Dict[str, Any]] = None,
            **kwargs: Any
        ) -> T:
            # Standardize config handling
            final_params = {}
            if params:
                final_params.update(params)

            if isinstance(config, dict):
                dict_params = config.pop("params", {}) if config else {}
                final_params.update(dict_params)
                return factory_func(
                    config=config_class(**(config or {})),
                    params=final_params,
                    **kwargs
                )
            elif isinstance(config, config_class):
                final_params.update(config.params)
                return factory_func(
                    config=config,
                    params=final_params,
                    **kwargs
                )
            else:
                return factory_func(
                    config=config_class(),
                    params=final_params,
                    **kwargs
                )

        return wrapper

    return decorator
```

## Timeline and Prioritization

### Phase 1: Foundation (Weeks 1-4)
- Complete high-priority tasks for component duplication
- Complete high-priority tasks for state management
- Begin standardizing factory function patterns
- Set up testing infrastructure

### Phase 2: Standardization (Weeks 5-8)
- Complete Pydantic standardization
- Finalize factory function implementation standards
- Complete complexity reduction high-priority tasks
- Implement core integration tests

### Phase 3: Refinement (Weeks 9-12)
- Complete all remaining medium-priority tasks
- Add comprehensive documentation
- Perform usability testing
- Create contribution guidelines

### Phase 4: Validation (Weeks 13-16)
- Conduct internal code reviews
- Complete test coverage goals
- Create example projects
- Prepare for community feedback

## Success Metrics

1. **Code Duplication**: < 5% duplication as measured by static analysis tools
2. **Test Coverage**: > 80% line coverage for core components
3. **Documentation**: 100% of public APIs documented with examples
4. **Consistency**: 100% compliance with established patterns for new code
5. **Learning Curve**: New contributor onboarding time reduced by 50%

## Conclusion

This implementation plan provides a structured approach to address the identified improvement areas in Sifaka. By following this plan, we can significantly enhance the codebase's maintainability, consistency, and usability while preserving its powerful capabilities.