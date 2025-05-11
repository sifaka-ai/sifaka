# Sifaka Plugin System

The Sifaka framework includes a powerful plugin system that allows you to extend its functionality without modifying the core codebase. This document explains how to use the plugin system and create your own plugins.

## Overview

The plugin system provides a standardized way to:

1. **Discover** plugins through entry points or module loading
2. **Register** plugins with a plugin registry
3. **Create components** from plugins when needed

Each component in Sifaka (models, rules, critics, classifiers, adapters, retrieval) has its own plugin system that extends the core plugin system with component-specific functionality.

## Using Plugins

### Loading Plugins

You can load plugins in several ways:

#### 1. From Entry Points

Plugins can be discovered and loaded from entry points:

```python
from sifaka.models.plugins import PluginLoader

# Create a plugin loader
loader = PluginLoader()

# Load plugins from entry points
plugins = loader.load_plugins_from_entry_points("sifaka.models.plugins")
```

#### 2. From Modules

Plugins can be loaded directly from modules:

```python
from sifaka.models.plugins import PluginLoader

# Create a plugin loader
loader = PluginLoader()

# Load a plugin from a module
plugin = loader.load_plugin_from_module("my_plugin_module", "MyPlugin")
```

#### 3. Manual Registration

Plugins can be registered manually:

```python
from sifaka.models.plugins import PluginRegistry
from my_plugin_module import MyPlugin

# Create a plugin registry
registry = PluginRegistry()

# Create and register a plugin
plugin = MyPlugin()
registry.register_plugin(plugin.name, plugin)
```

### Using Registered Plugins

Once plugins are registered, you can use them to create components:

```python
# Get a plugin from the registry
plugin = registry.get_plugin("my_plugin")

# Create a component from the plugin
config = {
    "param1": "value1",
    "param2": "value2"
}
component = plugin.create_component(config)

# Use the component
result = component.process(input_data)
```

## Creating Plugins

To create a plugin, you need to implement the appropriate plugin interface for the component you want to extend.

### Plugin Interface

All plugins must implement the `Plugin` interface from `sifaka.core.interfaces`:

```python
from sifaka.core.interfaces import Plugin

class MyPlugin(Plugin):
    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "my_plugin"
    
    @property
    def version(self) -> str:
        """Get the plugin version."""
        return "1.0.0"
    
    @property
    def component_type(self) -> str:
        """Get the component type this plugin provides."""
        return "my_component_type"
    
    def create_component(self, config: Dict[str, Any]) -> Any:
        """Create a component instance."""
        # Create and return a component instance
        return MyComponent(**config)
```

### Component-Specific Plugins

Each component has its own plugin interface that extends the core `Plugin` interface:

- **Models**: `ModelPlugin` from `sifaka.models.plugins`
- **Rules**: `RulePlugin` from `sifaka.rules.plugins`
- **Critics**: `CriticPlugin` from `sifaka.critics.plugins`
- **Classifiers**: `Plugin` from `sifaka.classifiers.interfaces`
- **Adapters**: `AdapterPlugin` from `sifaka.adapters.plugins`
- **Retrieval**: `RetrievalPlugin` from `sifaka.retrieval.plugins`

### Example: Creating a Model Plugin

Here's an example of creating a model plugin:

```python
from typing import Any, Dict
from sifaka.models.plugins import ModelPlugin
from sifaka.models.base import ModelProvider, ModelConfig

class MyModelPlugin(ModelPlugin):
    @property
    def name(self) -> str:
        return "my_model_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def component_type(self) -> str:
        return "model_provider"
    
    def create_component(self, config: Dict[str, Any]) -> ModelProvider:
        # Extract configuration values
        model_name = config.get("model_name", "default-model")
        temperature = config.get("temperature", 0.7)
        
        # Create model configuration
        model_config = ModelConfig(
            temperature=temperature,
            max_tokens=config.get("max_tokens", 1000),
            params=config.get("params", {})
        )
        
        # Create and return a model provider
        return MyModelProvider(
            model_name=model_name,
            config=model_config
        )
```

## Publishing Plugins

To make your plugin discoverable through entry points, add the following to your `setup.py`:

```python
setup(
    name="my-sifaka-plugin",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "sifaka.models.plugins": [
            "my_model_plugin = my_package.my_module:MyModelPlugin",
        ],
    },
)
```

## Best Practices

1. **Use descriptive names**: Choose clear, descriptive names for your plugins
2. **Version your plugins**: Use semantic versioning for your plugins
3. **Document your plugins**: Include docstrings and examples
4. **Handle errors gracefully**: Catch and handle errors in your plugins
5. **Follow the component's patterns**: Ensure your plugin follows the patterns of the component it extends
