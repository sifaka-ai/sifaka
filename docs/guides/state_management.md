# State Management in Sifaka

This guide explains how to properly manage state in Sifaka components. Following these guidelines ensures consistency across the codebase and helps prevent bugs related to state management.

## Overview

Sifaka uses a centralized state management approach through the `StateManager` class. This provides:

- Consistent state access patterns
- History tracking for undo/rollback functionality
- Clear separation of state data and metadata
- Type-safe state operations

## StateManager Basics

### Key Concepts

1. **State**: Core data that components need to operate
2. **Metadata**: Information about the component itself
3. **History**: Record of state changes for rollback

### Core Methods

```python
# Update state with history tracking
self._state_manager.update("key", value)

# Get state value with default fallback
value = self._state_manager.get("key", default_value)

# Update metadata (no history tracking)
self._state_manager.set_metadata("key", value)

# Get metadata with default fallback
value = self._state_manager.get_metadata("key", default_value)

# Rollback to previous state
self._state_manager.rollback()

# Reset state to initial values
self._state_manager.reset()
```

## Initializing StateManager

### Components Extending BaseComponent

For components that extend `BaseComponent`, use `PrivateAttr` with the appropriate factory function:

```python
from pydantic import PrivateAttr
from sifaka.core.base import BaseComponent
from sifaka.utils.state import create_critic_state, StateManager

class MyCritic(BaseComponent[str, Result]):
    _state_manager: StateManager = PrivateAttr(default_factory=create_critic_state)

    def _initialize_state(self, config: Optional[BaseConfig] = None) -> None:
        """Initialize component state."""
        super()._initialize_state()

        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {
            "template": "Default template",
            "temperature": 0.7,
        })
```

### Components Requiring Shared State

For components that need to share state, use dependency injection:

```python
class MyEngine:
    def __init__(self, state_manager: StateManager, name: str = "engine"):
        self._state_manager = state_manager
        self._state_manager.update("name", name)
        self._state_manager.update("initialized", True)
```

## Common Patterns

### Initialization Pattern

Always follow this pattern when initializing state:

1. Call parent's initialization if applicable
2. Set `initialized` to `True`
3. Initialize cache and other state values
4. Set component metadata

```python
def _initialize_components(self) -> None:
    """Initialize component state."""
    # Mark as initialized
    self._state_manager.update("initialized", True)

    # Initialize cache with defaults
    self._state_manager.update("cache", {
        "template": DEFAULT_TEMPLATE,
        "temperature": DEFAULT_TEMPERATURE,
    })

    # Set component metadata
    self._state_manager.set_metadata("component_type", self.__class__.__name__)
    self._state_manager.set_metadata("initialization_time", time.time())
```

### Checking Initialization

Always check if a component is initialized before using it:

```python
def my_method(self) -> None:
    """Do something with the component."""
    if not self._state_manager.get("initialized", False):
        raise RuntimeError("Component not properly initialized")

    # Rest of the method
```

### Error Handling

Record errors consistently:

```python
def record_error(self, error: Exception) -> None:
    """Record an error in the state manager."""
    if hasattr(self, "config") and self.config.track_errors:
        error_count = self._state_manager.get_metadata("error_count", 0)
        self._state_manager.set_metadata("error_count", error_count + 1)
        self._state_manager.set_metadata("last_error", str(error))
        self._state_manager.set_metadata("last_error_time", time.time())
```

### Performance Tracking

Track performance consistently:

```python
start_time = time.time()

# Method logic here

if self.config.track_performance:
    total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
    execution_time_ms = (time.time() - start_time) * 1000
    self._state_manager.set_metadata(
        "total_processing_time_ms",
        total_time + execution_time_ms
    )
```

## Best Practices

1. **Be explicit with defaults**: Always provide default values when getting state
2. **Use type annotations**: Add type hints for all state operations
3. **Document state fields**: Comment what each state field is used for
4. **Separate concerns**: Keep state logically organized in the state manager
5. **Be consistent with naming**: Use clear, consistent names for state keys
6. **Handle state errors**: Gracefully handle missing or invalid state
7. **Test state management**: Include tests for state operations

## Common Mistakes to Avoid

1. ❌ **Using `set()` instead of `update()`**: Always use `update()` for state changes
2. ❌ **Direct instantiation**: Don't use `self._state_manager = StateManager()`
3. ❌ **Skipping initialization check**: Always check if a component is initialized
4. ❌ **Not calling parent methods**: Call `super()._initialize_state()` first
5. ❌ **Using wrong state type**: Ensure correct types for state values
6. ❌ **Missing error handling**: Always handle state-related errors

## State Management Checklist

- [ ] Using correct initialization pattern
- [ ] Using `update()` for state changes
- [ ] Using `set_metadata()` for metadata
- [ ] Providing appropriate defaults with `get()`
- [ ] Checking initialization status
- [ ] Handling errors and edge cases
- [ ] Documenting state fields
- [ ] Including performance tracking if needed
- [ ] Following consistent naming conventions

## Further Reading

- [ADR-001: State Management Patterns](../adr/001-state-management-patterns.md)
- [State Management Consistency Report](../../CON.md)
- [BaseComponent Documentation](../../sifaka/core/base.py)
- [StateManager Implementation](../../sifaka/utils/state.py)