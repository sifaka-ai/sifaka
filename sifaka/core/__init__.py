"""
Core Module

A module that provides core functionality and interfaces for the Sifaka framework.

## Overview
This module serves as the foundation of the Sifaka framework, providing essential
interfaces and functionality that other components build upon. It establishes the
core patterns and contracts that ensure consistency across the framework.

## Components
1. Core Interfaces:
   - Component: Base interface for all components
   - Configurable: Interface for components with configuration
   - Stateful: Interface for components with state management
   - Identifiable: Interface for components with identity
   - Loggable: Interface for components with logging capabilities
   - Traceable: Interface for components with tracing capabilities
   - Plugin: Interface for plugins that extend component functionality

2. Plugin System:
   - PluginRegistry: Discovers and registers plugins
   - PluginLoader: Dynamically loads plugins at runtime

## Usage Examples
```python
from sifaka.core import Component, Configurable

class MyComponent(Component, Configurable):
    def initialize(self) -> None:
        # Initialize resources
        pass

    def cleanup(self) -> None:
        # Clean up resources
        pass

    @property
    def config(self):
        return self._config

    def update_config(self, config):
        # Update configuration
        pass
```

## Error Handling
The core module defines error handling patterns:
- RuntimeError for initialization and cleanup failures
- ValueError for invalid configuration
- Type checking for configuration objects
- Resource cleanup in cleanup methods

## Configuration
The core module supports various configuration options:
- Component initialization parameters
- Configuration object structure
- State management patterns
- Logging and tracing capabilities
"""

from .interfaces import (
    Component,
    Configurable,
    Stateful,
    Identifiable,
    Loggable,
    Traceable,
    Plugin,
)

from .plugins import (
    PluginRegistry,
    PluginLoader,
)

__all__ = [
    # Core interfaces
    "Component",
    "Configurable",
    "Stateful",
    "Identifiable",
    "Loggable",
    "Traceable",
    "Plugin",
    # Plugin system
    "PluginRegistry",
    "PluginLoader",
]
