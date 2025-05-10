"""
State Management Module

This module provides centralized state management for the Sifaka classifiers system.
It implements a StateTracker class that manages state across all components,
providing a consistent interface for state access and modification.

## Components
1. **StateTracker**: Centralized state management for all components
2. **State**: Immutable state container

## Usage Examples
```python
from sifaka.classifiers.v2.state import StateTracker

# Create state tracker
state = StateTracker()

# Update state
state.update("implementation", implementation)
state.update("config", config)

# Get state
implementation = state.get("implementation")
config = state.get("config")

# Set metadata
state.set_metadata("execution_count", 1)
state.set_metadata("last_execution_time", time.time())

# Get metadata
execution_count = state.get_metadata("execution_count")
last_execution_time = state.get_metadata("last_execution_time")

# Create snapshot
snapshot = state.create_snapshot()

# Restore snapshot
state.restore_snapshot(snapshot)
```
"""

from typing import Any, Dict, List, Optional, TypeVar
import time
from pydantic import BaseModel, Field

T = TypeVar("T")


class State(BaseModel):
    """Immutable state container."""
    
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class StateTracker:
    """Centralized state management for all components."""
    
    def __init__(self):
        """Initialize the state tracker."""
        self._state = State()
        self._history: List[State] = []
        self._snapshots: Dict[str, State] = {}
    
    def update(self, key: str, value: Any) -> None:
        """
        Update state with history tracking.
        
        Args:
            key: The state key to update
            value: The value to set
        """
        self._history.append(self._state)
        self._state = self._state.model_copy(
            update={"data": {**self._state.data, key: value}, "timestamp": time.time()}
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value.
        
        Args:
            key: The state key to get
            default: Default value if key not found
            
        Returns:
            The state value or default
        """
        return self._state.data.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata value.
        
        Args:
            key: The metadata key to set
            value: The value to set
        """
        self._state = self._state.model_copy(
            update={"metadata": {**self._state.metadata, key: value}, "timestamp": time.time()}
        )
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value.
        
        Args:
            key: The metadata key to get
            default: Default value if key not found
            
        Returns:
            The metadata value or default
        """
        return self._state.metadata.get(key, default)
    
    def rollback(self) -> bool:
        """
        Rollback to previous state.
        
        Returns:
            True if rollback succeeded, False if no history
        """
        if not self._history:
            return False
        self._state = self._history.pop()
        return True
    
    def create_snapshot(self, name: str = "") -> str:
        """
        Create a named snapshot of the current state.
        
        Args:
            name: Optional name for the snapshot
            
        Returns:
            The snapshot ID
        """
        snapshot_id = name or f"snapshot_{time.time()}"
        self._snapshots[snapshot_id] = self._state
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore state from a snapshot.
        
        Args:
            snapshot_id: The snapshot ID to restore
            
        Returns:
            True if restore succeeded, False if snapshot not found
        """
        if snapshot_id not in self._snapshots:
            return False
        self._history.append(self._state)
        self._state = self._snapshots[snapshot_id]
        return True
    
    def clear_history(self) -> None:
        """Clear state history."""
        self._history = []
    
    def clear_snapshots(self) -> None:
        """Clear state snapshots."""
        self._snapshots = {}
    
    def reset(self) -> None:
        """Reset state to empty state."""
        self._history.append(self._state)
        self._state = State()
    
    def get_state(self) -> State:
        """
        Get the current state.
        
        Returns:
            The current state
        """
        return self._state
    
    def get_history(self) -> List[State]:
        """
        Get the state history.
        
        Returns:
            The state history
        """
        return self._history.copy()
    
    def get_snapshots(self) -> Dict[str, State]:
        """
        Get the state snapshots.
        
        Returns:
            The state snapshots
        """
        return self._snapshots.copy()
