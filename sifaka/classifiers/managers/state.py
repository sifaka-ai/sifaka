"""
State management for classifiers.

This module provides state management components for classifiers in the Sifaka framework.
"""

from typing import Any, Dict, Generic, Optional, Type, TypeVar

# Type variables for generic state management
T = TypeVar("T")  # State type


class StateManager(Generic[T]):
    """
    Manages state for classifiers.

    This class provides a standardized way to manage state for classifiers,
    including initialization, access, and cleanup of resources.

    ## Lifecycle

    1. **Creation**: Instantiate with optional initial state
       - Provide initial state if available
       - Initialize with empty state otherwise

    2. **Access**: Get and set state properties
       - Use get() to retrieve state properties
       - Use set() to update state properties
       - Use has() to check if a property exists

    3. **Cleanup**: Release resources when done
       - Call cleanup() to release any resources
       - Automatically called when the manager is garbage collected

    ## Error Handling

    The class implements these error handling patterns:
    - Safe property access with get()
    - Type checking for state properties
    - Resource cleanup in cleanup()

    ## Examples

    Basic usage:

    ```python
    from sifaka.classifiers.managers import StateManager

    # Create a state manager
    state = StateManager()

    # Set state properties
    state.set("model", load_model())
    state.set("tokenizer", load_tokenizer())

    # Get state properties
    model = state.get("model")
    if model is not None:
        result = model.predict(text)

    # Check if a property exists
    if state.has("cache"):
        cache = state.get("cache")
        # Use cache...

    # Cleanup resources
    state.cleanup()
    ```

    Using with a classifier:

    ```python
    from sifaka.classifiers.base import BaseClassifier
    from sifaka.classifiers.managers import StateManager

    class MyClassifier(BaseClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._state = StateManager()

        def warm_up(self):
            # Initialize resources
            self._state.set("model", load_model())
            self._state.set("tokenizer", load_tokenizer())

        def _classify_impl_uncached(self, text):
            # Use state resources
            model = self._state.get("model")
            tokenizer = self._state.get("tokenizer")
            
            # Process text
            tokens = tokenizer.tokenize(text)
            prediction = model.predict(tokens)
            
            # Return result
            return ClassificationResult(
                label=prediction.label,
                confidence=prediction.confidence
            )
    ```
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the state manager.

        Args:
            initial_state: Optional initial state dictionary
        """
        self._state: Dict[str, Any] = initial_state or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a state property.

        Args:
            key: Property key
            default: Default value if the key doesn't exist

        Returns:
            The property value or default
        """
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a state property.

        Args:
            key: Property key
            value: Property value
        """
        self._state[key] = value

    def has(self, key: str) -> bool:
        """
        Check if a property exists.

        Args:
            key: Property key

        Returns:
            True if the property exists, False otherwise
        """
        return key in self._state

    def cleanup(self) -> None:
        """
        Clean up resources.

        This method should be called when the state manager is no longer needed
        to release any resources held by the state.
        """
        # Clean up any resources that need explicit cleanup
        for key, value in self._state.items():
            if hasattr(value, "close") and callable(value.close):
                try:
                    value.close()
                except Exception:
                    pass  # Ignore cleanup errors
            elif hasattr(value, "cleanup") and callable(value.cleanup):
                try:
                    value.cleanup()
                except Exception:
                    pass  # Ignore cleanup errors

        # Clear the state
        self._state.clear()

    def __del__(self) -> None:
        """
        Clean up resources when the object is garbage collected.
        """
        self.cleanup()
