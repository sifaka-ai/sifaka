"""
Plugin interfaces for chain components.

This module defines the plugin interfaces for chain components in the Sifaka framework.
These interfaces establish a common contract for plugin behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **ChainPlugin**: Interface for chain plugins

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from typing import Protocol, runtime_checkable

from sifaka.core.interfaces import Plugin as CorePlugin

from .base import ChainComponent


@runtime_checkable
class ChainPlugin(ChainComponent, CorePlugin, Protocol):
    """
    Interface for chain plugins.

    This interface extends the core Plugin interface with chain-specific
    functionality. It ensures that chain plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """
