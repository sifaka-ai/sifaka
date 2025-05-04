"""
Classifier state management utilities.

This module provides utility functions and classes for managing classifier state
in the Sifaka framework.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ComponentState(BaseModel):
    """
    Base class for component state.

    This class provides a standardized way to represent component state.
    It can be extended to represent the state of specific component types.
    """

    initialized: bool = False
    error: Optional[str] = None

    def reset(self) -> None:
        """Reset the state to its initial values."""
        self.initialized = False
        self.error = None


class ClassifierState(ComponentState):
    """
    State for classifiers.

    This class represents the state of a classifier component.
    It includes common state variables used by classifiers.
    """

    model: Optional[Any] = None
    vectorizer: Optional[Any] = None
    pipeline: Optional[Any] = None
    feature_names: Dict[str, Any] = {}
    cache: Dict[str, Any] = {}
    dependencies_loaded: bool = False


class StateManager:
    """
    Utility class for standardized state management.

    This class provides a standardized way to manage state in Sifaka components.
    It handles initialization, access, and modification of state in a consistent way.
    """

    def __init__(
        self,
        initializer: callable,
        initialized: bool = False,
        state: Optional[Any] = None,
    ) -> None:
        """
        Initialize the state manager.

        Args:
            initializer: Function that initializes the state
            initialized: Whether the state is already initialized
            state: Initial state (if already initialized)
        """
        self._initializer = initializer
        self._initialized = initialized
        self._state = state

    def initialize(self) -> Any:
        """
        Initialize the state if not already initialized.

        Returns:
            The initialized state
        """
        if not self._initialized:
            self._state = self._initializer()
            self._initialized = True
        return self._state

    def get_state(self) -> Any:
        """
        Get the state, initializing it if necessary.

        Returns:
            The state
        """
        return self.initialize()

    def set_state(self, state: Any) -> None:
        """
        Set the state.

        Args:
            state: The new state
        """
        self._state = state
        self._initialized = True

    def update_state(self, updater: callable) -> Any:
        """
        Update the state using an updater function.

        Args:
            updater: Function that takes the current state and returns the new state

        Returns:
            The updated state
        """
        self._state = updater(self.get_state())
        return self._state

    def reset(self) -> None:
        """Reset the state manager."""
        self._initialized = False
        self._state = None

    @property
    def is_initialized(self) -> bool:
        """
        Check if the state is initialized.

        Returns:
            True if the state is initialized, False otherwise
        """
        return self._initialized


def create_classifier_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a classifier.

    Args:
        **kwargs: Additional keyword arguments to pass to the state

    Returns:
        A state manager for a classifier
    """
    return StateManager(
        initializer=lambda: ClassifierState(**kwargs),
        initialized=False,
        state=None,
    )
