"""Plugin discovery and loading utilities for Sifaka.

This module provides comprehensive plugin loading capabilities, including:
- Directory-based plugin discovery
- Entry point discovery via setuptools
- Plugin dependency resolution
- Plugin validation and initialization
- Error handling and logging

## Usage:

    from sifaka.core.plugin_loader import PluginLoader

    # Create loader
    loader = PluginLoader()

    # Load plugins from directory
    plugins = loader.load_from_directory("./plugins")

    # Load plugins from entry points
    plugins = loader.load_from_entry_points("sifaka.critics")

    # Load all plugins
    plugins = loader.load_all_plugins()

## Plugin Directory Structure:

    plugins/
    ├── my_critic/
    │   ├── __init__.py
    │   ├── plugin.py       # Contains plugin class
    │   └── requirements.txt
    └── my_validator/
        ├── __init__.py
        ├── plugin.py
        └── requirements.txt

## Entry Points:

    # In setup.py or pyproject.toml
    entry_points = {
        'sifaka.critics': [
            'my_critic = my_package.critics:MyCriticPlugin',
        ],
        'sifaka.validators': [
            'my_validator = my_package.validators:MyValidatorPlugin',
        ]
    }
"""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, cast

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore

from .plugin_interfaces import (
    CriticPlugin,
    PluginDependencyError,
    PluginDiscoveryError,
    PluginError,
    PluginInterface,
    PluginType,
    ValidatorPlugin,
    get_plugin_registry,
)

logger = logging.getLogger(__name__)


class PluginLoader:
    """Utility for discovering and loading Sifaka plugins.

    This class provides comprehensive plugin loading capabilities including
    directory scanning, entry point discovery, dependency resolution, and
    validation. It handles various failure modes gracefully and provides
    detailed logging for troubleshooting.
    """

    def __init__(self) -> None:
        """Initialize the plugin loader."""
        self._loaded_plugins: Dict[str, PluginInterface] = {}
        self._failed_plugins: Dict[str, Exception] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self.registry = get_plugin_registry()

    def load_from_directory(self, directory: Union[str, Path]) -> List[PluginInterface]:
        """Load plugins from a directory.

        Scans the directory for Python packages/modules containing plugin
        classes. Each subdirectory is treated as a potential plugin.

        Args:
            directory: Path to directory containing plugin packages

        Returns:
            List of successfully loaded plugin instances

        Raises:
            PluginDiscoveryError: If directory doesn't exist or isn't readable
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise PluginDiscoveryError(f"Plugin directory {directory} does not exist")

        if not dir_path.is_dir():
            raise PluginDiscoveryError(f"Plugin path {directory} is not a directory")

        plugins = []

        # Scan for plugin directories
        for item in dir_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                try:
                    plugin = self._load_plugin_from_directory(item)
                    if plugin:
                        plugins.append(plugin)
                except Exception as e:
                    logger.warning(f"Failed to load plugin from {item}: {e}")
                    self._failed_plugins[item.name] = e

        return plugins

    def load_from_entry_points(self, group: str) -> List[PluginInterface]:
        """Load plugins from setuptools entry points.

        Discovers plugins registered via setuptools entry points in the
        specified group (e.g., 'sifaka.critics', 'sifaka.validators').

        Args:
            group: Entry point group name

        Returns:
            List of successfully loaded plugin instances
        """
        plugins = []

        try:
            # Get entry points for the group
            entry_points = metadata.entry_points()

            # Handle different Python versions
            if hasattr(entry_points, "select"):
                # Python 3.10+
                entries = entry_points.select(group=group)
            else:
                # Python 3.8-3.9
                entries = cast(Any, entry_points).get(group, [])

            for entry_point in entries:
                try:
                    plugin = self._load_plugin_from_entry_point(entry_point)
                    if plugin:
                        plugins.append(plugin)
                except Exception as e:
                    logger.warning(f"Failed to load plugin {entry_point.name}: {e}")
                    self._failed_plugins[entry_point.name] = e

        except Exception as e:
            logger.warning(f"Failed to discover entry points for group {group}: {e}")

        return plugins

    def load_all_plugins(self) -> List[PluginInterface]:
        """Load all plugins from all sources.

        Discovers and loads plugins from:
        1. Entry points in standard groups
        2. Default plugin directories
        3. Environment-specified directories

        Returns:
            List of all successfully loaded plugin instances
        """
        plugins = []

        # Load from entry points
        standard_groups = [
            "sifaka.critics",
            "sifaka.validators",
            "sifaka.storage",
            "sifaka.middleware",
            "sifaka.plugins",
        ]

        for group in standard_groups:
            try:
                group_plugins = self.load_from_entry_points(group)
                plugins.extend(group_plugins)
            except Exception as e:
                logger.warning(
                    f"Failed to load plugins from entry point group {group}: {e}"
                )

        # Load from default directories
        default_dirs = [
            Path.cwd() / "plugins",
            Path.home() / ".sifaka" / "plugins",
            Path(__file__).parent.parent / "plugins",
        ]

        for dir_path in default_dirs:
            if dir_path.exists():
                try:
                    dir_plugins = self.load_from_directory(dir_path)
                    plugins.extend(dir_plugins)
                except Exception as e:
                    logger.warning(
                        f"Failed to load plugins from directory {dir_path}: {e}"
                    )

        # Load from environment-specified directories
        env_dirs = os.environ.get("SIFAKA_PLUGIN_DIRS", "").split(os.pathsep)
        for dir_str in env_dirs:
            if dir_str.strip():
                try:
                    dir_plugins = self.load_from_directory(dir_str.strip())
                    plugins.extend(dir_plugins)
                except Exception as e:
                    logger.warning(
                        f"Failed to load plugins from env directory {dir_str}: {e}"
                    )

        return plugins

    def resolve_dependencies(
        self, plugins: List[PluginInterface]
    ) -> List[PluginInterface]:
        """Resolve plugin dependencies and return plugins in load order.

        Analyzes plugin dependencies and returns them in an order that ensures
        dependencies are loaded before dependents.

        Args:
            plugins: List of plugin instances to resolve

        Returns:
            List of plugins in dependency order

        Raises:
            PluginDependencyError: If circular dependencies or missing dependencies
        """
        # Build dependency graph
        graph = {}
        plugin_map = {plugin.metadata.name: plugin for plugin in plugins}

        for plugin in plugins:
            name = plugin.metadata.name
            deps = set(plugin.metadata.dependencies)
            graph[name] = deps
            self._dependency_graph[name] = deps

        # Topological sort to resolve dependencies
        sorted_names = self._topological_sort(graph)

        # Return plugins in dependency order
        ordered_plugins = []
        for name in sorted_names:
            if name in plugin_map:
                ordered_plugins.append(plugin_map[name])

        return ordered_plugins

    def validate_plugin(self, plugin: PluginInterface) -> bool:
        """Validate a plugin's metadata and interface compliance.

        Args:
            plugin: Plugin instance to validate

        Returns:
            True if plugin is valid

        Raises:
            PluginError: If plugin validation fails
        """
        try:
            # Validate metadata
            metadata = plugin.metadata
            if not metadata.name:
                raise PluginError("Plugin metadata must have a name")

            if not metadata.version:
                raise PluginError("Plugin metadata must have a version")

            # Validate plugin type
            if metadata.plugin_type not in PluginType:
                raise PluginError(f"Invalid plugin type: {metadata.plugin_type}")

            # Validate interface compliance
            if metadata.plugin_type == PluginType.CRITIC:
                if not isinstance(plugin, CriticPlugin):
                    raise PluginError("Critic plugins must inherit from CriticPlugin")
            elif metadata.plugin_type == PluginType.VALIDATOR:
                if not isinstance(plugin, ValidatorPlugin):
                    raise PluginError(
                        "Validator plugins must inherit from ValidatorPlugin"
                    )

            # Validate version compatibility
            # TODO: Implement proper version checking

            return True

        except Exception as e:
            raise PluginError(f"Plugin validation failed: {e}") from e

    def get_loaded_plugins(self) -> Dict[str, PluginInterface]:
        """Get all successfully loaded plugins.

        Returns:
            Dictionary mapping plugin names to instances
        """
        return self._loaded_plugins.copy()

    def get_failed_plugins(self) -> Dict[str, Exception]:
        """Get information about failed plugin loads.

        Returns:
            Dictionary mapping plugin names to their load errors
        """
        return self._failed_plugins.copy()

    def _load_plugin_from_directory(self, directory: Path) -> Optional[PluginInterface]:
        """Load a plugin from a directory.

        Args:
            directory: Path to plugin directory

        Returns:
            Plugin instance or None if not found
        """
        # Look for plugin.py or __init__.py
        plugin_files = [
            directory / "plugin.py",
            directory / "__init__.py",
        ]

        for plugin_file in plugin_files:
            if plugin_file.exists():
                try:
                    return self._load_plugin_from_file(plugin_file, directory.name)
                except Exception as e:
                    logger.warning(f"Failed to load plugin from {plugin_file}: {e}")

        return None

    def _load_plugin_from_file(
        self, file_path: Path, module_name: str
    ) -> Optional[PluginInterface]:
        """Load a plugin from a Python file.

        Args:
            file_path: Path to Python file
            module_name: Module name for importing

        Returns:
            Plugin instance or None if not found
        """
        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find plugin class
        plugin_class = self._find_plugin_class(module)
        if plugin_class is None:
            return None

        # Create plugin instance
        plugin = plugin_class()

        # Validate plugin
        self.validate_plugin(plugin)

        # Store in registry
        self.registry.register(plugin)
        self._loaded_plugins[plugin.metadata.name] = plugin

        return plugin

    def _load_plugin_from_entry_point(
        self, entry_point: Any
    ) -> Optional[PluginInterface]:
        """Load a plugin from an entry point.

        Args:
            entry_point: Setuptools entry point

        Returns:
            Plugin instance or None if not found
        """
        # Load the plugin class
        plugin_class = entry_point.load()

        # Validate it's a plugin class
        if not inspect.isclass(plugin_class):
            raise PluginError(
                f"Entry point {entry_point.name} does not point to a class"
            )

        if not issubclass(plugin_class, PluginInterface):
            raise PluginError(
                f"Entry point {entry_point.name} does not point to a PluginInterface subclass"
            )

        # Create plugin instance
        plugin = plugin_class()

        # Validate plugin
        self.validate_plugin(plugin)

        # Store in registry
        self.registry.register(plugin)
        self._loaded_plugins[plugin.metadata.name] = plugin

        return plugin

    def _find_plugin_class(self, module: Any) -> Optional[Type[PluginInterface]]:
        """Find a plugin class in a module.

        Args:
            module: Python module to search

        Returns:
            Plugin class or None if not found
        """
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, PluginInterface)
                and obj is not PluginInterface
                and obj is not CriticPlugin
                and obj is not ValidatorPlugin
            ):
                return obj

        return None

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Perform topological sort on dependency graph.

        Args:
            graph: Dictionary mapping names to their dependencies

        Returns:
            List of names in dependency order

        Raises:
            PluginDependencyError: If circular dependencies detected
        """
        # Kahn's algorithm for topological sorting
        in_degree = {name: 0 for name in graph}

        # Calculate in-degrees
        for name, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[name] += 1

        # Find nodes with no incoming edges
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Remove edges from current node
            for name, deps in graph.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # Check for circular dependencies
        if len(result) != len(graph):
            remaining = set(graph.keys()) - set(result)
            raise PluginDependencyError(
                f"Circular dependency detected among plugins: {remaining}"
            )

        return result


# Global plugin loader instance
_global_loader = PluginLoader()


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader instance.

    Returns:
        Global plugin loader
    """
    return _global_loader


def load_plugins_from_directory(directory: Union[str, Path]) -> List[PluginInterface]:
    """Load plugins from a directory using the global loader.

    Args:
        directory: Path to directory containing plugin packages

    Returns:
        List of successfully loaded plugin instances
    """
    return _global_loader.load_from_directory(directory)


def load_plugins_from_entry_points(group: str) -> List[PluginInterface]:
    """Load plugins from entry points using the global loader.

    Args:
        group: Entry point group name

    Returns:
        List of successfully loaded plugin instances
    """
    return _global_loader.load_from_entry_points(group)


def load_all_plugins() -> List[PluginInterface]:
    """Load all plugins from all sources using the global loader.

    Returns:
        List of all successfully loaded plugin instances
    """
    return _global_loader.load_all_plugins()


def discover_and_load_plugins() -> Dict[str, List[PluginInterface]]:
    """Discover and load all plugins, organized by type.

    Returns:
        Dictionary mapping plugin types to lists of plugin instances
    """
    loader = get_plugin_loader()
    plugins = loader.load_all_plugins()

    # Organize by type
    by_type: Dict[PluginType, List[PluginInterface]] = {
        PluginType.CRITIC: [],
        PluginType.VALIDATOR: [],
        PluginType.STORAGE: [],
        PluginType.MIDDLEWARE: [],
        PluginType.TOOL: [],
    }

    for plugin in plugins:
        plugin_type = plugin.metadata.plugin_type
        if plugin_type in by_type:
            by_type[plugin_type].append(plugin)

    return {k.value: v for k, v in by_type.items()}
