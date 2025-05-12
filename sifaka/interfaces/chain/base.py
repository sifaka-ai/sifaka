"""
Base interfaces for chain components.

This module defines the base interfaces for chain components in the Sifaka framework.
These interfaces establish a common contract for component behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **ChainComponent**: Base interface for all chain components

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChainComponent(Protocol):
    """
    Base interface for all chain components.

    This interface defines the common properties that all chain components
    must implement. It serves as the foundation for the component hierarchy
    in the chain system, ensuring consistent identification and description
    of components.

    ## Architecture
    The ChainComponent interface uses Python's Protocol class to define
    a structural interface that components can implement without explicit
    inheritance. This enables better flexibility and composition in the
    component architecture.

    ## Lifecycle
    Components implementing this interface should maintain consistent
    name and description properties throughout their lifecycle.

    ## Examples
    ```python
    class MyComponent:
        @property
        def name(self) -> str:
            return "my_component"

        @property
        def description(self) -> str:
            return "A custom chain component"
    ```
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            str: A unique identifier for the component
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            str: A human-readable description of the component
        """
        ...
