"""
Component Protocol Module

This module defines the base protocol for components in the Sifaka framework.
All components should implement this protocol, either directly or through inheritance.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ComponentProtocol(Protocol):
    """
    Base protocol for all components in Sifaka.

    This protocol defines the common methods that all components should implement.
    It serves as the foundation for all specialized component protocols.
    """

    @property
    def name(self) -> str:
        """Get the name of the component."""
        ...

    @property
    def description(self) -> str:
        """Get the description of the component."""
        ...

    def initialize(self) -> None:
        """Initialize the component. Called when the component is created."""
        ...

    def warm_up(self) -> None:
        """Warm up the component. Called before the first use."""
        ...

    def cleanup(self) -> None:
        """Clean up resources. Called when the component is no longer needed."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the component."""
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the component."""
        ...
