"""
State management utilities for Sifaka.

This module provides utility functions and classes for standardized state management
across the Sifaka framework. It includes classes for representing component state
and utilities for managing state in a consistent way.

## State Management

The module provides standardized state management:

1. **State**: Immutable state container
2. **StateManager**: Utility class for managing component state
3. **ComponentState**: Base class for component state
4. **Specialized State Classes**: State classes for specific component types
   - **ClassifierState**: State for classifiers
   - **RuleState**: State for rules
   - **CriticState**: State for critics
   - **ModelState**: State for model providers
   - **ChainState**: State for chains
   - **AdapterState**: State for adapters

## Usage Examples

```python
from sifaka.utils.state import State, StateManager, ClassifierState
from pydantic import BaseModel, ConfigDict

class MyClassifier(BaseModel):
    # Configuration
    name: str
    threshold: float = 0.5

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize state manager
        self.state = StateManager()

    def warm_up(self) -> None:
        # Initialize the component
        if not self.state.get("initialized"):
            self.state.update("model", self._load_model())
            self.state.update("cache", {})
            self.state.update("initialized", True)
```

## State Management Pattern

The recommended pattern is to use the StateManager directly:

```python
from sifaka.utils.state import StateManager

class MyComponent(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.state = StateManager()

    def process(self, data: Any) -> Any:
        # Update state
        self.state.update("last_processed", data)

        # Get state
        cache = self.state.get("cache", {})

        # Set metadata
        self.state.set_metadata("component_type", self.__class__.__name__)
```
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T")


class State(BaseModel):
    """Immutable state container."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateManager:
    """Unified state management for all components."""

    def __init__(self):
        self._state: State = State()
        self._history: List[State] = []

    def update(self, key: str, value: Any) -> None:
        """Update state with history tracking."""
        self._history.append(self._state)
        self._state = self._state.model_copy(update={"data": {**self._state.data, key: value}})

    def rollback(self) -> None:
        """Rollback to previous state."""
        if self._history:
            self._state = self._history.pop()

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self._state.data.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self._state = self._state.model_copy(
            update={"metadata": {**self._state.metadata, key: value}}
        )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._state.metadata.get(key, default)

    def reset(self) -> None:
        """Reset state to initial values."""
        self._state = State()
        self._history.clear()


class ComponentState(BaseModel):
    """Base class for component state."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    initialized: bool = False
    error: Optional[str] = None


class ClassifierState(ComponentState):
    """State for classifiers."""

    model: Optional[Any] = None
    vectorizer: Optional[Any] = None
    pipeline: Optional[Any] = None
    feature_names: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)
    dependencies_loaded: bool = False


class RuleState(ComponentState):
    """State for rules."""

    validator: Optional[Any] = None
    handler: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)
    compiled_patterns: Dict[str, Any] = Field(default_factory=dict)


class CriticState(ComponentState):
    """State for critics."""

    model: Optional[Any] = None
    prompt_manager: Optional[Any] = None
    response_parser: Optional[Any] = None
    memory_manager: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)


class ModelState(ComponentState):
    """State for model providers."""

    client: Optional[Any] = None
    token_counter: Optional[Any] = None
    tracer: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)


class ChainState(ComponentState):
    """State for chains."""

    model: Optional[Any] = None
    generator: Optional[Any] = None
    validation_manager: Optional[Any] = None
    prompt_manager: Optional[Any] = None
    retry_strategy: Optional[Any] = None
    result_formatter: Optional[Any] = None
    critic: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)


class AdapterState(ComponentState):
    """State for adapters."""

    adaptee: Optional[Any] = None
    adaptee_cache: Dict[str, Any] = Field(default_factory=dict)
    config_cache: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)


# Factory functions for creating state managers with specific state types
def create_state_manager(state_class: type[T], **kwargs: Any) -> StateManager:
    """Create a state manager for a specific state class."""
    manager = StateManager()
    state = state_class(**kwargs)
    for key, value in state.model_dump().items():
        manager.update(key, value)
    return manager


def create_classifier_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a classifier."""
    return create_state_manager(ClassifierState, **kwargs)


def create_rule_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a rule."""
    return create_state_manager(RuleState, **kwargs)


def create_critic_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a critic."""
    return create_state_manager(CriticState, **kwargs)


def create_model_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a model provider."""
    return create_state_manager(ModelState, **kwargs)


def create_chain_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a chain."""
    return create_state_manager(ChainState, **kwargs)


def create_adapter_state(**kwargs: Any) -> StateManager:
    """Create a state manager for an adapter."""
    return create_state_manager(AdapterState, **kwargs)
