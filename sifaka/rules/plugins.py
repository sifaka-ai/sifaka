"""
Rule Plugins Module

This module provides the plugin system for the Sifaka rules system.
It enables dynamic discovery and loading of plugins for extending the system.

This module re-exports the core plugin system components with rule-specific
customizations to ensure consistent plugin behavior across all components.

## Usage Examples
```python
from sifaka.rules.plugins import PluginRegistry, PluginLoader
from sifaka.rules.interfaces import Plugin

# Create plugin registry
registry = PluginRegistry()

# Register plugin
(registry and registry.register_plugin("my_plugin", MyPlugin())

# Get plugin
plugin = (registry and registry.get_plugin("my_plugin")

# Create plugin loader
loader = PluginLoader()

# Load plugin from entry point
plugins = (loader and loader.load_plugins_from_entry_points("sifaka.rules.plugins")

# Load plugin from module
plugin = (loader and loader.load_plugin_from_module("my_plugin_module")
```
"""

from typing import Dict, Optional, Type, cast

from sifaka.core.plugins import PluginRegistry as CorePluginRegistry
from sifaka.core.plugins import PluginLoader as CorePluginLoader
from sifaka.core.interfaces import Plugin
from ..utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class RulePlugin(Plugin):
    """
    Interface for rule plugins.
    
    This interface extends the core Plugin interface with rule-specific
    functionality. It ensures that rule plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """
    pass


class PluginRegistry(CorePluginRegistry):
    """
    Rule-specific plugin registry.
    
    This class extends the core PluginRegistry with rule-specific functionality.
    It ensures that only rule plugins can be registered.
    """
    
    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """
        Register a rule plugin.
        
        Args:
            name: Plugin name
            plugin: Plugin instance
            
        Raises:
            PluginError: If plugin registration fails
        """
        # Ensure the plugin is a rule plugin
        if not isinstance(plugin, RulePlugin):
            from ..utils.errors import PluginError
            raise PluginError(f"Plugin '{name}' is not a rule plugin")
            
        # Register the plugin with the core registry
        super().register_plugin(name, plugin)


class PluginLoader(CorePluginLoader):
    """
    Rule-specific plugin loader.
    
    This class extends the core PluginLoader with rule-specific functionality.
    It ensures that only rule plugins can be loaded.
    """
    
    def def __init__(self, registry: Optional[Optional[PluginRegistry]] = None):
        """
        Initialize the rule plugin loader.
        
        Args:
            registry: Optional rule plugin registry to register plugins with
        """
        # Create a rule plugin registry if none is provided
        rule_registry = registry or PluginRegistry()
        
        # Initialize the core plugin loader with the rule registry
        super().__init__(rule_registry)
        
    def get_registry(self) -> PluginRegistry:
        """
        Get the rule plugin registry.
        
        Returns:
            Rule plugin registry
        """
        # Cast the core registry to a rule registry
        return cast(PluginRegistry, super().get_registry())
