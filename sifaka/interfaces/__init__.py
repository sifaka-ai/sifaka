"""
Interfaces module for Sifaka.

This module provides the core interfaces and protocols that define the contracts
for all components in the Sifaka framework. These interfaces establish clear
boundaries between components and enable better modularity and extensibility.

## Module Structure

1. **Core Interfaces**
   - `Component`: Base interface for all components
   - `Configurable`: Interface for components with configuration
   - `Stateful`: Interface for components with state management
   - `Identifiable`: Interface for components with identity

2. **Component-Specific Interfaces**
   - `models`: Interfaces for model providers
   - `rules`: Interfaces for rules and validators
   - `critics`: Interfaces for critics
   - `chain`: Interfaces for chains
   - `retrieval`: Interfaces for retrieval components

## Usage

Interfaces in this module are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.

Example:
```python
from sifaka.interfaces import Component
from sifaka.interfaces.models import ModelProvider

# Check if a class implements an interface
if isinstance(my_object, Component):
    print("Object implements Component interface")

# Create a class that implements an interface
class MyModelProvider:
    def __init__(self, name: str):
        self.name = name
        
    def get_name(self) -> str:
        return self.name
        
    # Implement other required methods...

# Check implementation
provider = MyModelProvider("my-provider")
assert isinstance(provider, ModelProvider)
```
"""

from sifaka.interfaces.core import (
    Component,
    Configurable,
    Stateful,
    Identifiable,
    Loggable,
    Traceable,
)

__all__ = [
    # Core interfaces
    "Component",
    "Configurable",
    "Stateful",
    "Identifiable",
    "Loggable",
    "Traceable",
]
