"""
Adapter Plugins Module

This module provides the plugin system for the Sifaka adapters system.
It enables dynamic discovery and loading of plugins for extending the system.

This module re-exports the core plugin system components with adapter-specific
customizations to ensure consistent plugin behavior across all components.

## Usage Examples
```python
from sifaka.adapters.plugins import PluginRegistry, PluginLoader
from sifaka.adapters.interfaces import Plugin

# Create plugin registry
registry = PluginRegistry()

# Register plugin
registry.register_plugin("my_plugin", MyPlugin())

# Get plugin
plugin = registry.get_plugin("my_plugin")

# Create plugin loader
loader = PluginLoader()

# Load plugin from entry point
plugins = loader.load_plugins_from_entry_points("sifaka.adapters.plugins")

# Load plugin from module
plugin = loader.load_plugin_from_module("my_plugin_module")
```
"""

from typing import Dict, Optional, Type, cast

from sifaka.core.plugins import PluginRegistry as CorePluginRegistry
from sifaka.core.plugins import PluginLoader as CorePluginLoader
from sifaka.core.interfaces import Plugin
from ..utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class AdapterPlugin(Plugin):
    """
    Interface for adapter plugins.
    
    This interface extends the core Plugin interface with adapter-specific
    functionality. It ensures that adapter plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """
    pass


class PluginRegistry(CorePluginRegistry):
    """
    Adapter-specific plugin registry.
    
    This class extends the core PluginRegistry with adapter-specific functionality.
    It ensures that only adapter plugins can be registered.
    """
    
    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """
        Register an adapter plugin.
        
        Args:
            name: Plugin name
            plugin: Plugin instance
            
        Raises:
            PluginError: If plugin registration fails
        """
        # Ensure the plugin is an adapter plugin
        if not isinstance(plugin, AdapterPlugin):
            from ..utils.errors import PluginError
            raise PluginError(f"Plugin '{name}' is not an adapter plugin")
            
        # Register the plugin with the core registry
        super().register_plugin(name, plugin)


class PluginLoader(CorePluginLoader):
    """
    Adapter-specific plugin loader.
    
    This class extends the core PluginLoader with adapter-specific functionality.
    It ensures that only adapter plugins can be loaded.
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the adapter plugin loader.
        
        Args:
            registry: Optional adapter plugin registry to register plugins with
        """
        # Create an adapter plugin registry if none is provided
        adapter_registry = registry or PluginRegistry()
        
        # Initialize the core plugin loader with the adapter registry
        super().__init__(adapter_registry)
        
    def get_registry(self) -> PluginRegistry:
        """
        Get the adapter plugin registry.
        
        Returns:
            Adapter plugin registry
        """
        # Cast the core registry to an adapter registry
        return cast(PluginRegistry, super().get_registry())
