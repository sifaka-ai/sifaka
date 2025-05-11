"""
Chain Plugins Module

This module provides the plugin system for the Sifaka chain system.
It enables dynamic discovery and loading of plugins for extending the system.

This module re-exports the core plugin system components with chain-specific
customizations to ensure consistent plugin behavior across all components.

## Usage Examples
```python
from sifaka.chain.plugins import PluginRegistry, PluginLoader
from sifaka.chain.interfaces import Plugin

# Create plugin registry
registry = PluginRegistry()

# Register plugin
registry.register_plugin("my_plugin", MyPlugin())

# Get plugin
plugin = registry.get_plugin("my_plugin")

# Create plugin loader
loader = PluginLoader()

# Load plugin from entry point
plugins = loader.load_plugins_from_entry_points("sifaka.chain.plugins")

# Load plugin from module
plugin = loader.load_plugin_from_module("my_plugin_module")
```
"""

from typing import Dict, Optional, Type, cast

from .interfaces import Plugin
from sifaka.core.plugins import PluginRegistry as CorePluginRegistry
from sifaka.core.plugins import PluginLoader as CorePluginLoader
from ..utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class PluginRegistry(CorePluginRegistry):
    """
    Chain-specific plugin registry.

    This class extends the core PluginRegistry with chain-specific functionality.
    It ensures that only chain plugins can be registered.
    """

    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """
        Register a chain plugin.

        Args:
            name: Plugin name
            plugin: Plugin instance

        Raises:
            PluginError: If plugin registration fails
        """
        # Ensure the plugin is a chain plugin
        if not isinstance(plugin, Plugin):
            from ..utils.errors import PluginError

            raise PluginError(f"Plugin '{name}' is not a chain plugin")

        # Register the plugin with the core registry
        super().register_plugin(name, plugin)


class PluginLoader(CorePluginLoader):
    """
    Chain-specific plugin loader.

    This class extends the core PluginLoader with chain-specific functionality.
    It ensures that only chain plugins can be loaded.
    """

    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the chain plugin loader.

        Args:
            registry: Optional chain plugin registry to register plugins with
        """
        # Create a chain plugin registry if none is provided
        chain_registry = registry or PluginRegistry()

        # Initialize the core plugin loader with the chain registry
        super().__init__(chain_registry)

    def get_registry(self) -> PluginRegistry:
        """
        Get the chain plugin registry.

        Returns:
            Chain plugin registry
        """
        # Cast the core registry to a chain registry
        return cast(PluginRegistry, super().get_registry())
