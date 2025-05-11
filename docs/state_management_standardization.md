# State Management Standardization

This document describes the standardized state management pattern used across all Sifaka components.

## Overview

Sifaka uses a standardized approach to state management across all components to ensure consistency, maintainability, and reliability. The state management pattern is implemented using the `StateManager` class from `sifaka.utils.state`.

## Key Components

1. **State**: Immutable state container
2. **StateManager**: Utility class for managing component state
3. **Factory Functions**: Functions for creating state managers for specific component types

## Standardized Pattern

All components should follow this standardized pattern:

1. Use `_state_manager` as the attribute name for state management
2. Initialize state during component construction
3. Access state through the state manager
4. Use clear state update and access patterns
5. Separate configuration from state

## Implementation Example

```python
from pydantic import BaseModel, PrivateAttr
from sifaka.utils.state import create_rule_state

class MyComponent(BaseModel):
    # Configuration (immutable)
    name: str
    description: str = "A component with standardized state management"
    
    # State management (mutable)
    _state_manager = PrivateAttr(default_factory=create_rule_state)
    
    def __init__(self, **data):
        """Initialize the component with standardized state management."""
        super().__init__(**data)
        
        # Initialize state
        self._state_manager.update("initialized", False)
        self._state_manager.update("cache", {})
        
        # Set metadata
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())
    
    def process(self, data):
        """Process data using the component."""
        # Get state values
        cache = self._state_manager.get("cache", {})
        
        # Check cache
        if data in cache:
            self._state_manager.set_metadata("cache_hit", True)
            return cache[data]
        
        # Process data
        result = self._process_impl(data)
        
        # Update cache
        cache[data] = result
        self._state_manager.update("cache", cache)
        
        return result
```

## State Manager API

The `StateManager` class provides the following methods:

- `update(key, value)`: Update state with history tracking
- `get(key, default=None)`: Get state value
- `set_metadata(key, value)`: Set metadata value
- `get_metadata(key, default=None)`: Get metadata value
- `reset()`: Reset state to initial values
- `rollback()`: Rollback to previous state

## Factory Functions

The `utils/state.py` module provides factory functions for creating state managers for specific component types:

- `create_rule_state()`: Create a state manager for a rule
- `create_classifier_state()`: Create a state manager for a classifier
- `create_critic_state()`: Create a state manager for a critic
- `create_model_state()`: Create a state manager for a model provider
- `create_chain_state()`: Create a state manager for a chain
- `create_adapter_state()`: Create a state manager for an adapter

## Best Practices

1. **Initialization**: Always initialize state in the constructor
2. **Access**: Always access state through the state manager
3. **Updates**: Always update state through the state manager
4. **Metadata**: Use metadata for component-level information
5. **Caching**: Use the state manager for caching results
6. **Error Handling**: Store errors in state for debugging
7. **Statistics**: Track component statistics in state

## Testing

When testing components with standardized state management:

1. Verify that state is initialized correctly
2. Verify that state is updated correctly during operations
3. Verify that state is accessed correctly
4. Verify that metadata is set and retrieved correctly
5. Verify that caching works correctly

## Example: Rules Component

The rules component uses standardized state management:

```python
from sifaka.utils.state import create_rule_state

class MyValidator(BaseValidator[str]):
    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)
    
    def __init__(self, config):
        super().__init__(validation_type=str)
        
        # Store configuration in state
        self._state_manager.update("config", config)
        
        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())
    
    @property
    def config(self):
        """Get the validator configuration."""
        return self._state_manager.get("config")
    
    def validate(self, text):
        # Check cache if enabled
        cache_size = self.config.cache_size
        if cache_size > 0:
            cache = self._state_manager.get("cache", {})
            if text in cache:
                self._state_manager.set_metadata("cache_hit", True)
                return cache[text]
        
        # Perform validation
        result = self._validate_impl(text)
        
        # Update statistics
        self.update_statistics(result)
        
        # Cache result if caching is enabled
        if cache_size > 0:
            cache = self._state_manager.get("cache", {})
            if len(cache) >= cache_size:
                cache = {}
            cache[text] = result
            self._state_manager.update("cache", cache)
        
        return result
```
