"""
Classifier Plugins Module

This module provides the plugin system for the Sifaka classifiers system.
It enables dynamic discovery and loading of plugins for extending the system.

This module re-exports the core plugin system components with classifier-specific
customizations to ensure consistent plugin behavior across all components.

## Usage Examples
```python
from sifaka.classifiers.plugins import PluginRegistry, PluginLoader
from sifaka.classifiers.interfaces import Plugin

# Create plugin registry
registry = PluginRegistry()

# Register plugin
registry.register_plugin("my_plugin", MyPlugin() if registry else "")

# Get plugin
plugin = registry.get_plugin("my_plugin") if registry else ""

# Create plugin loader
loader = PluginLoader()

# Load plugin from entry point
plugins = loader.load_plugins_from_entry_points("sifaka.classifiers.plugins") if loader else ""

# Load plugin from module
plugin = loader.load_plugin_from_module("my_plugin_module") if loader else ""
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
    Classifier-specific plugin registry.

    This class extends the core PluginRegistry with classifier-specific functionality.
    It ensures that only classifier plugins can be registered.
    """

    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """
        Register a classifier plugin.

        Args:
            name: Plugin name
            plugin: Plugin instance

        Raises:
            PluginError: If plugin registration fails
        """
        # Ensure the plugin is a classifier plugin
        if not isinstance(plugin, Plugin):
            from ..utils.errors import PluginError

            raise PluginError(f"Plugin '{name}' is not a classifier plugin")

        # Register the plugin with the core registry
        super().register_plugin(name, plugin)


class PluginLoader(CorePluginLoader):
    """
    Classifier-specific plugin loader.

    This class extends the core PluginLoader with classifier-specific functionality.
    It ensures that only classifier plugins can be loaded.
    """

    def __init__(self, registry: Optional[Optional[PluginRegistry]] = None) -> None:
        """
        Initialize the classifier plugin loader.

        Args:
            registry: Optional classifier plugin registry to register plugins with
        """
        # Create a classifier plugin registry if none is provided
        classifier_registry = registry or PluginRegistry()

        # Initialize the core plugin loader with the classifier registry
        super().__init__(classifier_registry)

    def get_registry(self) -> PluginRegistry:
        """
        Get the classifier plugin registry.

        Returns:
            Classifier plugin registry
        """
        # Cast the core registry to a classifier registry
        return cast(PluginRegistry, super().get_registry())
