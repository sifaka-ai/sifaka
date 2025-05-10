"""
Adapter interfaces for Sifaka.

This module defines the interfaces for adapters in the Sifaka framework.
These interfaces establish a common contract for adapter behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Adaptable**: Protocol for components that can be adapted
2. **BaseAdapter**: Base class for implementing adapters

## Usage Examples

```python
from sifaka.interfaces.adapter import Adaptable

# Create a component that can be adapted
class MyComponent:
    @property
    def name(self) -> str:
        return "my_component"

    @property
    def description(self) -> str:
        return "A custom component"

# Check if the component implements the Adaptable protocol
assert isinstance(MyComponent(), Adaptable)
```

## Error Handling

- ValueError: Raised for invalid inputs
- RuntimeError: Raised for execution failures
- TypeError: Raised for type mismatches
- AdapterError: Raised for adapter-specific errors
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Adaptable(Protocol):
    """
    Protocol for components that can be adapted to rules.

    ## Overview
    Any component that can be adapted to a Sifaka rule must implement
    this protocol, which requires a name and description.

    ## Architecture
    The protocol defines a minimal interface that components must implement
    to be compatible with Sifaka's adapter system.

    ## Lifecycle
    1. **Implementation**: Component implements the required properties
       - Provide a name for identification
       - Provide a description of functionality

    2. **Adaptation**: Component is adapted using a compatible adapter
       - Adapter receives component instance
       - Adapter validates component compatibility

    3. **Usage**: Adapted component is used as a Sifaka rule validator
       - Adapter translates between component and rule interfaces
       - Component's functionality is leveraged in validation

    ## Error Handling
    - TypeError: Raised if required properties are not implemented
    - ValueError: Raised if property values are invalid
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            str: A string name for the component

        Raises:
            NotImplementedError: If the property is not implemented
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            str: A string description of the component's purpose

        Raises:
            NotImplementedError: If the property is not implemented
        """
        ...
