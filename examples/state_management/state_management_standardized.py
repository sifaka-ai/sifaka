"""
State Management Standardized Example

This example demonstrates the standardized state management pattern using _state_manager
instead of _state. This is the recommended approach for all Sifaka components.

Key concepts:
1. Use _state_manager as the attribute name for state management
2. Initialize state during component construction
3. Access state through the state manager
4. Use clear state update and access patterns
5. Separate configuration from state
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from sifaka.utils.state import StateManager, create_classifier_state


class StandardizedComponent(BaseModel):
    """Example component using standardized state management."""

    # Configuration (immutable)
    name: str
    description: str = "A component with standardized state management"
    threshold: float = 0.5

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management (mutable)
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    def __init__(self, **data):
        """Initialize the component with standardized state management."""
        super().__init__(**data)
        
        # Initialize state
        self._initialize_state()
        
        # Set metadata
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("created_at", "2023-05-09")

    def _initialize_state(self) -> None:
        """Initialize component state."""
        # Update state with initial values
        self._state_manager.update("initialized", False)
        self._state_manager.update("cache", {})
        self._state_manager.update("model", None)
        self._state_manager.update("feature_names", {})
        self._state_manager.update("dependencies_loaded", False)

    def warm_up(self) -> None:
        """Initialize the component if not already initialized."""
        if not self._state_manager.get("initialized"):
            # Load model and update state
            model = self._load_model()
            self._state_manager.update("model", model)
            self._state_manager.update("initialized", True)
            print(f"Component '{self.name}' initialized with model: {model}")

    def _load_model(self) -> str:
        """Simulate loading a model."""
        return "MockModel"

    def process(self, data: str) -> Dict[str, Any]:
        """Process data using the component."""
        # Ensure component is initialized
        if not self._state_manager.get("initialized"):
            self.warm_up()

        # Get state values
        model = self._state_manager.get("model")
        cache = self._state_manager.get("cache", {})

        # Check cache
        if data in cache:
            print(f"Using cached result for '{data}'")
            return cache[data]

        # Process data
        result = {
            "input": data,
            "model": model,
            "score": len(data) / 10 * self.threshold,
            "processed_by": self.name
        }

        # Update cache
        cache[data] = result
        self._state_manager.update("cache", cache)
        
        # Update metadata
        self._state_manager.set_metadata("last_processed", data)

        return result

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "initialized": self._state_manager.get("initialized"),
            "cache_size": len(self._state_manager.get("cache", {})),
            "model": self._state_manager.get("model"),
            "metadata": {
                "component_type": self._state_manager.get_metadata("component_type"),
                "created_at": self._state_manager.get_metadata("created_at"),
                "last_processed": self._state_manager.get_metadata("last_processed")
            }
        }

    def reset(self) -> None:
        """Reset the component state."""
        self._state_manager.reset()
        self._initialize_state()
        print(f"Component '{self.name}' state has been reset")


# Example usage
if __name__ == "__main__":
    # Create component
    component = StandardizedComponent(name="ExampleComponent")
    
    # Check initial state
    print("Initial state:", component.get_state_summary())
    
    # Process data
    result1 = component.process("Hello, world!")
    print("Result 1:", result1)
    
    # Process same data (should use cache)
    result2 = component.process("Hello, world!")
    print("Result 2:", result2)
    
    # Process different data
    result3 = component.process("Different input")
    print("Result 3:", result3)
    
    # Check state after processing
    print("State after processing:", component.get_state_summary())
    
    # Reset state
    component.reset()
    
    # Check state after reset
    print("State after reset:", component.get_state_summary())
