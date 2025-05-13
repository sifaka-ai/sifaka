"""
Adapter Plugins Module

This module provides the plugin system for the Sifaka adapters component.

## Overview
This module enables dynamic discovery and loading of plugins for extending the adapters
system. It re-exports the core plugin system components with adapter-specific
customizations to ensure consistent plugin behavior across all components.

## Components
1. **AdapterPlugin**: Interface for adapter plugins
2. **PluginRegistry**: Registry for adapter plugins
3. **PluginLoader**: Loader for adapter plugins

## Usage Examples
```python
from sifaka.adapters.plugins import PluginRegistry, PluginLoader, AdapterPlugin

# Define a custom adapter plugin
class MyAdapterPlugin(AdapterPlugin):
    def __init__(self, name="my_plugin"):
        self.name = name

    def initialize(self):
        print(f"Initializing {self.name}")

    def get_metadata(self):
        return {"type": "adapter_plugin", "name": self.name}

# Create plugin registry
registry = PluginRegistry()

# Register plugin
registry.register_plugin("my_plugin", MyAdapterPlugin() if registry else "")

# Get plugin
plugin = registry.get_plugin("my_plugin") if registry else ""
plugin.initialize() if plugin else ""

# Create plugin loader
loader = PluginLoader()

# Load plugins from entry points
plugins = loader.load_plugins_from_entry_points("sifaka.adapters.plugins") if loader else ""

# Load plugin from module
plugin = loader.load_plugin_from_module("my_plugin_module") if loader else ""
```

## Error Handling
- PluginError: Raised when plugin registration or loading fails
- ImportError: Raised when a plugin module cannot be imported
- TypeError: Raised when a plugin does not implement the required interface

## Plugin Discovery
Plugins can be discovered through:
1. Entry points in setup.py/pyproject.toml
2. Direct module imports
3. Manual registration
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

    ## Overview
    This interface extends the core Plugin interface with adapter-specific
    functionality. It ensures that adapter plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.

    ## Architecture
    The adapter plugin system follows a standard plugin architecture:
    1. Plugin Interface: Defines the contract for plugins
    2. Plugin Registry: Manages plugin registration and discovery
    3. Plugin Loader: Handles loading plugins from various sources

    ## Lifecycle
    1. Definition: Plugin class implements the AdapterPlugin interface
    2. Registration: Plugin is registered with the PluginRegistry
    3. Discovery: Plugin is discovered by the PluginLoader
    4. Initialization: Plugin is initialized when needed
    5. Usage: Plugin functionality is used by the adapter system

    ## Examples
    ```python
    class MyAdapterPlugin(AdapterPlugin):
        def __init__(self, name="my_plugin"):
            self.name = name

        def initialize(self):
            print(f"Initializing {self.name}")

        def get_metadata(self):
            return {"type": "adapter_plugin", "name": self.name}
    ```
    """

    pass


class PluginRegistry(CorePluginRegistry):
    """
    Adapter-specific plugin registry.

    ## Overview
    This class extends the core PluginRegistry with adapter-specific functionality.
    It ensures that only adapter plugins can be registered and provides a centralized
    registry for adapter plugins.

    ## Architecture
    The registry follows a standard registry pattern:
    1. Registration: Plugins are registered with a unique name
    2. Retrieval: Plugins can be retrieved by name
    3. Validation: Plugins are validated during registration

    ## Lifecycle
    1. Creation: Registry is created
    2. Registration: Plugins are registered with the registry
    3. Retrieval: Plugins are retrieved from the registry
    4. Usage: Plugins are used by the adapter system

    ## Examples
    ```python
    # Create registry
    registry = PluginRegistry()

    # Register plugin
    registry.register_plugin("my_plugin", MyAdapterPlugin() if registry else "")

    # Get plugin
    plugin = registry.get_plugin("my_plugin") if registry else ""

    # Check if plugin exists
    if registry.has_plugin("my_plugin") if registry else "":
        print("Plugin exists")

    # Get all plugins
    all_plugins = registry.get_all_plugins() if registry else ""
    ```
    """

    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """
        Register an adapter plugin.

        This method registers an adapter plugin with the registry. It ensures that
        only adapter plugins can be registered by validating that the plugin
        implements the AdapterPlugin interface.

        Args:
            name: Plugin name
            plugin: Plugin instance

        Raises:
            PluginError: If plugin registration fails or plugin is not an adapter plugin

        Example:
            ```python
            registry = PluginRegistry()
            registry.register_plugin("my_plugin", MyAdapterPlugin() if registry else "")
            ```
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

    ## Overview
    This class extends the core PluginLoader with adapter-specific functionality.
    It ensures that only adapter plugins can be loaded and provides methods for
    discovering and loading adapter plugins from various sources.

    ## Architecture
    The loader follows a standard loader pattern:
    1. Discovery: Plugins are discovered from various sources
    2. Loading: Plugins are loaded and instantiated
    3. Registration: Loaded plugins are registered with the registry

    ## Lifecycle
    1. Creation: Loader is created with a registry
    2. Discovery: Plugins are discovered from entry points or modules
    3. Loading: Plugins are loaded and instantiated
    4. Registration: Loaded plugins are registered with the registry
    5. Usage: Loaded plugins are used by the adapter system

    ## Examples
    ```python
    # Create loader with registry
    registry = PluginRegistry()
    loader = PluginLoader(registry)

    # Load plugins from entry points
    plugins = loader.load_plugins_from_entry_points("sifaka.adapters.plugins") if loader else ""

    # Load plugin from module
    plugin = loader.load_plugin_from_module("my_plugin_module") if loader else ""

    # Get registry with loaded plugins
    registry = loader.get_registry() if loader else ""
    ```
    """

    def __init__(self, registry: Optional[Optional[PluginRegistry]] = None):
        """
        Initialize the adapter plugin loader.

        This method initializes the adapter plugin loader with an optional registry.
        If no registry is provided, a new one is created.

        Args:
            registry: Optional adapter plugin registry to register plugins with

        Example:
            ```python
            # Create loader with new registry
            loader = PluginLoader()

            # Create loader with existing registry
            registry = PluginRegistry()
            loader = PluginLoader(registry)
            ```
        """
        # Create an adapter plugin registry if none is provided
        adapter_registry = registry or PluginRegistry()

        # Initialize the core plugin loader with the adapter registry
        super().__init__(adapter_registry)

    def get_registry(self) -> PluginRegistry:
        """
        Get the adapter plugin registry.

        This method returns the adapter plugin registry associated with this loader.
        It casts the core registry to an adapter registry to ensure type safety.

        Returns:
            Adapter plugin registry with all registered plugins

        Example:
            ```python
            loader = PluginLoader()
            registry = loader.get_registry() if loader else ""

            # Use the registry
            plugin = registry.get_plugin("my_plugin") if registry else ""
            ```
        """
        # Cast the core registry to an adapter registry
        return cast(PluginRegistry, super().get_registry())
