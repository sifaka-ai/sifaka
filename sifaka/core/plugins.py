"""
Core Plugin System Module

This module provides a standardized plugin system for all Sifaka components.
It enables dynamic discovery, registration, and loading of plugins for extending the system.

## Components
1. **PluginRegistry**: Discovers and registers plugins
2. **PluginLoader**: Dynamically loads plugins at runtime

## Usage Examples
```python
from sifaka.core.plugins import PluginRegistry, PluginLoader
from sifaka.core.interfaces import Plugin

# Create plugin registry
registry = PluginRegistry()

# Register plugin
registry.register_plugin("my_plugin", MyPlugin())

# Get plugin
plugin = registry.get_plugin("my_plugin")

# Create plugin loader
loader = PluginLoader()

# Load plugin from entry point
plugins = loader.load_plugins_from_entry_points("sifaka.plugins")

# Load plugin from module
plugin = loader.load_plugin_from_module("my_plugin_module")
```
"""
from typing import Any, Dict, List, Optional, Type
import importlib
import pkg_resources
from .interfaces import Plugin
from ..utils.errors import PluginError
from ..utils.logging import get_logger
logger = get_logger(__name__)


class PluginRegistry:
    """Discovers and registers plugins."""

    def __init__(self) ->None:
        """Initialize the plugin registry."""
        self._plugins: Dict[str, Plugin] = {}

    def register_plugin(self, name: str, plugin: Plugin) ->None:
        """
        Register a plugin.

        Args:
            name: Plugin name
            plugin: Plugin instance

        Raises:
            PluginError: If plugin registration fails
        """
        if name in self._plugins:
            raise PluginError(f"Plugin '{name}' already registered")
        self._plugins[name] = plugin
        logger.debug(f"Registered plugin '{name}'")

    def unregister_plugin(self, name: str) ->None:
        """
        Unregister a plugin.

        Args:
            name: Plugin name

        Raises:
            PluginError: If plugin unregistration fails
        """
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' not registered")
        del self._plugins[name]
        logger.debug(f"Unregistered plugin '{name}'")

    def get_plugin(self, name: str) ->Plugin:
        """
        Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance

        Raises:
            PluginError: If plugin not found
        """
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' not found")
        return self._plugins[name]

    def get_plugins(self) ->Dict[str, Plugin]:
        """
        Get all registered plugins.

        Returns:
            Dictionary of plugin names to plugin instances
        """
        return self._plugins.copy()

    def get_plugins_by_type(self, component_type: str) ->Dict[str, Plugin]:
        """
        Get plugins by component type.

        Args:
            component_type: Component type to filter by

        Returns:
            Dictionary of plugin names to plugin instances
        """
        return {name: plugin for name, plugin in self._plugins.items() if 
            plugin.component_type == component_type)

    def clear(self) ->None:
        """Clear all registered plugins."""
        self._plugins.clear()
        logger.debug('Cleared all plugins')


class PluginLoader:
    """Dynamically loads plugins at runtime."""

    def __init__(self, registry: Optional[Optional[PluginRegistry]] = None) ->None:
        """
        Initialize the plugin loader.

        Args:
            registry: Optional plugin registry to register plugins with
        """
        self._registry = registry or PluginRegistry()

    def load_plugins_from_entry_points(self, group: str) ->Dict[str, Plugin]:
        """
        Load plugins from entry points.

        Args:
            group: Entry point group to load plugins from

        Returns:
            Dictionary of plugin names to plugin instances

        Raises:
            PluginError: If plugin loading fails
        """
        plugins = {}
        try:
            for entry_point in pkg_resources.iter_entry_points(group):
                try:
                    plugin_class = entry_point.load()
                    plugin = plugin_class()
                    if not isinstance(plugin, Plugin):
                        logger.warning(
                            f"Entry point '{entry_point.name}' does not provide a Plugin instance"
                            )
                        continue
                    plugins[entry_point.name] = plugin
                    if self._registry:
                        self._registry.register_plugin(entry_point.name, plugin
                            )
                    logger.debug(
                        f"Loaded plugin '{entry_point.name}' from entry point")
                except Exception as e:
                    logger.error(
                        f"Failed to load plugin from entry point '{entry_point.name}': {e}"
                        )
        except Exception as e:
            raise PluginError(f'Failed to load plugins from entry points: {e}')
        return plugins

    def load_plugin_from_module(self, module_name: str, class_name: Optional[Optional[str]] = None) ->Plugin:
        """
        Load a plugin from a module.

        Args:
            module_name: Module name to load plugin from
            class_name: Optional class name to load

        Returns:
            Plugin instance

        Raises:
            PluginError: If plugin loading fails
        """
        try:
            module = importlib.import_module(module_name)
            if class_name:
                if not hasattr(module, class_name):
                    raise PluginError(
                        f"Module '{module_name}' has no attribute '{class_name}'"
                        )
                plugin_class = getattr(module, class_name)
                plugin = plugin_class()
            else:
                plugin = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Plugin
                        ) and attr != Plugin:
                        plugin = attr()
                        break
                if not plugin:
                    raise PluginError(
                        f"No Plugin class found in module '{module_name}'")
            if not isinstance(plugin, Plugin):
                raise PluginError(
                    f"Class '{class_name or plugin.__class__.__name__}' does not implement Plugin"
                    )
            if self._registry:
                self._registry.register_plugin(plugin.name, plugin)
            logger.debug(
                f"Loaded plugin '{plugin.name}' from module '{module_name}'")
            return plugin
        except ImportError as e:
            raise PluginError(f"Failed to import module '{module_name}': {e}")
        except Exception as e:
            raise PluginError(
                f"Failed to load plugin from module '{module_name}': {e}")

    def get_registry(self) ->PluginRegistry:
        """
        Get the plugin registry.

        Returns:
            Plugin registry
        """
        return self._registry
