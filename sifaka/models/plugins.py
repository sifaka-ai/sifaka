"""
Model Plugins Module

This module provides the plugin system for the Sifaka models system.
It enables dynamic discovery and loading of plugins for extending the system.

This module re-exports the core plugin system components with model-specific
customizations to ensure consistent plugin behavior across all components.

## Usage Examples
```python
from sifaka.models.plugins import PluginRegistry, PluginLoader
from sifaka.models.interfaces import Plugin

# Create plugin registry
registry = PluginRegistry()

# Register plugin
registry.register_plugin("my_plugin", MyPlugin() if registry else "")

# Get plugin
plugin = registry.get_plugin("my_plugin") if registry else ""

# Create plugin loader
loader = PluginLoader()

# Load plugin from entry point
plugins = loader.load_plugins_from_entry_points("sifaka.models.plugins") if loader else ""

# Load plugin from module
plugin = loader.load_plugin_from_module("my_plugin_module") if loader else ""
```
"""

from typing import Dict, Optional, Type, cast

from sifaka.core.plugins import PluginRegistry as CorePluginRegistry
from sifaka.core.plugins import PluginLoader as CorePluginLoader
from sifaka.core.interfaces import Plugin
from ..utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class ModelPlugin(Plugin):
    """
    Interface for model plugins.

    This interface extends the core Plugin interface with model-specific
    functionality. It ensures that model plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """

    pass


class PluginRegistry(CorePluginRegistry):
    """
    Model-specific plugin registry.

    This class extends the core PluginRegistry with model-specific functionality.
    It ensures that only model plugins can be registered.
    """

    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """
        Register a model plugin.

        Args:
            name: Plugin name
            plugin: Plugin instance

        Raises:
            PluginError: If plugin registration fails
        """
        # Ensure the plugin is a model plugin
        if not isinstance(plugin, ModelPlugin):
            from ..utils.errors import PluginError

            raise PluginError(f"Plugin '{name}' is not a model plugin")

        # Register the plugin with the core registry
        super().register_plugin(name, plugin)


class PluginLoader(CorePluginLoader):
    """
    Model-specific plugin loader.

    This class extends the core PluginLoader with model-specific functionality.
    It ensures that only model plugins can be loaded.
    """

    def __init__(self, registry: Optional[Optional[PluginRegistry]] = None) -> None:
        """
        Initialize the model plugin loader.

        Args:
            registry: Optional model plugin registry to register plugins with
        """
        # Create a model plugin registry if none is provided
        model_registry = registry or PluginRegistry()

        # Initialize the core plugin loader with the model registry
        super().__init__(model_registry)

    def get_registry(self) -> PluginRegistry:
        """
        Get the model plugin registry.

        Returns:
            Model plugin registry
        """
        # Cast the core registry to a model registry
        return cast(PluginRegistry, super().get_registry())  # type: ignore[no-any-return]
