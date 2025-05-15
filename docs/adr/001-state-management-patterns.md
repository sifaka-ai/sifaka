# ADR 001: State Management Patterns

## Status

Proposed

## Context

The Sifaka codebase uses a StateManager abstraction to maintain component state across the application. However, an analysis of the codebase revealed inconsistencies in how state is managed:

1. **Method inconsistencies**: Some components incorrectly used `set()` instead of the correct `update()` method
2. **Inconsistent initialization patterns**:
   - Direct instantiation: `self._state_manager = StateManager()`
   - Factory functions: `self._state_manager = create_critic_state()`
   - Dependency injection: Receiving StateManager via constructor
3. **Unclear guidelines** on when to use each approach

These inconsistencies increase the risk of bugs, make code harder to understand, and complicate maintenance as the codebase grows.

## Decision

We will standardize state management patterns across the codebase using the following principles:

### 1. State Manager Initialization

| Component Type | Initialization Pattern | Example |
|----------------|------------------------|---------|
| Components extending BaseComponent | Use `PrivateAttr` with factory functions | `_state_manager: StateManager = PrivateAttr(default_factory=create_critic_state)` |
| Adapters | Use adapter-specific factory | `self._state_manager = create_adapter_state()` |
| Critics | Use critic-specific factory | `self._state_manager = create_critic_state()` |
| Classifiers | Use classifier-specific factory | `self._state_manager = create_classifier_state()` |
| Components requiring shared state | Use dependency injection | `def __init__(self, state_manager: StateManager)` |

### 2. State Modification

| Operation | Method | Purpose |
|-----------|--------|---------|
| Update state data | `update(key, value)` | For core state data with history tracking |
| Get state data | `get(key, default)` | Retrieve state with optional default |
| Set metadata | `set_metadata(key, value)` | For component metadata |
| Get metadata | `get_metadata(key, default)` | Retrieve metadata with optional default |
| Reset state | `reset()` | Clear all state to initial values |
| Rollback state | `rollback()` | Return to previous state |

The `set()` method should never be used as it doesn't exist in the StateManager interface.

### 3. Component Inheritance

Components that extend `BaseComponent` should leverage the standard lifecycle methods:

```python
def _initialize_state(self, config: Optional[BaseConfig] = None) -> None:
    """Initialize component state with standardized patterns."""
    super()._initialize_state()

    # Component-specific state initialization goes here
    self._state_manager.update("initialized", True)
    self._state_manager.update("cache", { ... })
```

### 4. Documentation Requirements

All components using state management must document:
- State dependencies between components
- Purpose of each state field
- Distinction between transient cache and persistent state

## Consequences

### Positive

1. **Improved consistency**: Standardized patterns make the code more predictable and easier to understand
2. **Reduced bugs**: Proper state management reduces the risk of state-related bugs
3. **Better testability**: Consistent patterns enable automated testing of state management
4. **Clearer boundaries**: Well-defined patterns clarify component relationships
5. **Easier onboarding**: New developers can learn one pattern that applies throughout the codebase

### Negative

1. **Refactoring effort**: Existing code needs to be updated to follow the new patterns
2. **Learning curve**: Developers need to learn and apply the standardized patterns
3. **Potential rigidity**: Standardization may make some edge cases more difficult to implement

### Neutral

1. **Code review changes**: Reviews must include verification of state management patterns
2. **Documentation updates**: Component docs must specify state management approach

## Implementation Plan

1. Create static analysis rules to enforce the patterns
2. Gradually refactor components to follow standardized patterns
3. Add state management sections to component documentation
4. Create developer guide section on state management
5. Implement automated tests to verify state management consistency

## References

- [Sifaka StateManager implementation](../sifaka/utils/state.py)
- [State Management Consistency Report](../CON.md)
- [Pydantic Private Attributes](https://docs.pydantic.dev/latest/usage/models/#private-model-attributes)