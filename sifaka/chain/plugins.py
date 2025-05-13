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
registry.register_plugin("my_plugin", MyPlugin() if registry else "")

# Get plugin
plugin = registry.get_plugin("my_plugin") if registry else ""

# Create plugin loader
loader = PluginLoader()

# Load plugin from entry point
plugins = loader.load_plugins_from_entry_points("sifaka.chain.plugins") if loader else ""

# Load plugin from module
plugin = loader.load_plugin_from_module("my_plugin_module") if loader else ""
```
"""

from typing import Dict, Optional, Type, cast

from sifaka.interfaces.chain.plugin import ChainPlugin as Plugin
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

    ## Architecture
    The PluginRegistry follows the registry pattern to manage plugins:
    - Extends the core PluginRegistry with chain-specific validation
    - Ensures that only chain plugins can be registered
    - Maintains a registry of available plugins
    - Provides methods for plugin registration and retrieval

    ## Lifecycle
    1. **Initialization**: Registry is created
    2. **Registration**: Plugins are registered with the registry
    3. **Retrieval**: Plugins are retrieved from the registry
    4. **Unregistration**: Plugins can be unregistered when no longer needed

    ## Error Handling
    - PluginError: Raised when plugin registration or retrieval fails
    - Validates plugin types before registration

    Attributes:
        _plugins (Dict[str, Plugin]): Dictionary of registered plugins
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

    ## Architecture
    The PluginLoader follows the loader pattern to dynamically load plugins:
    - Extends the core PluginLoader with chain-specific functionality
    - Uses a chain-specific PluginRegistry for plugin registration
    - Loads plugins from entry points and modules
    - Validates that loaded plugins are chain plugins

    ## Lifecycle
    1. **Initialization**: Loader is created with a registry
    2. **Discovery**: Plugins are discovered from entry points or modules
    3. **Loading**: Plugins are loaded and validated
    4. **Registration**: Loaded plugins are registered with the registry

    ## Error Handling
    - PluginError: Raised when plugin loading fails
    - ImportError: Raised when plugin modules cannot be imported
    - Validates plugin types after loading

    Attributes:
        _registry (PluginRegistry): The registry to register plugins with
    """

    def __init__(self, registry: Optional[Optional[PluginRegistry]] = None):
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
