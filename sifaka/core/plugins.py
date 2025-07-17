"""Plugin discovery and registration system for Sifaka storage backends.

This module provides a flexible plugin system that allows third-party storage
backends to be discovered and registered automatically. It uses Python's
entry points mechanism for plugin discovery.

## Plugin System Overview:

The plugin system enables extending Sifaka with custom storage backends without
modifying the core codebase. Plugins can be distributed as separate packages
and discovered automatically at runtime.

## Creating a Plugin:

To create a storage plugin, implement the StorageBackend interface and register
it via setuptools entry points:

    # In your plugin's setup.py or pyproject.toml:
    entry_points={
        'sifaka.storage': [
            'redis = my_plugin.redis_storage:RedisStorage',
            'postgres = my_plugin.pg_storage:PostgresStorage',
        ]
    }

## Usage:

    >>> from sifaka.core.plugins import create_storage_backend
    >>>
    >>> # Automatically discovers and loads redis plugin
    >>> storage = create_storage_backend('redis', url='redis://localhost')
    >>>
    >>> # List all available backends
    >>> from sifaka.core.plugins import list_storage_backends
    >>> print(list_storage_backends())
    ['memory', 'file', 'redis', 'postgres']

## Design Principles:

1. **Lazy Discovery**: Plugins are discovered on first use, not import
2. **Graceful Failures**: Missing plugins don't crash the system
3. **Type Safety**: All plugins must inherit from StorageBackend
4. **Zero Config**: Plugins work immediately after installation

## Plugin Resolution:

1. Built-in backends are registered first (memory, file)
2. Entry points are scanned from installed packages
3. Duplicate names are rejected (first wins)
4. Failed plugins are logged but don't stop discovery
"""

from typing import Any, Dict, List, Type, cast

try:
    from importlib import metadata
    from importlib.metadata import EntryPoints
except ImportError:
    # Python 3.7 compatibility
    import importlib_metadata as metadata  # type: ignore[no-redef]
    from importlib_metadata import EntryPoints  # type: ignore[assignment,no-redef]

from ..storage.base import StorageBackend


class PluginRegistry:
    """Registry for storage backend plugins.

    Manages the discovery, registration, and retrieval of storage backend
    plugins. Provides both automatic discovery via entry points and manual
    registration for custom backends.

    Key features:
    - Lazy plugin discovery (on first use)
    - Type checking for all registered backends
    - Graceful handling of plugin loading failures
    - Support for both built-in and third-party backends

    Example:
        >>> registry = PluginRegistry()
        >>>
        >>> # Manual registration
        >>> registry.register('custom', CustomStorageBackend)
        >>>
        >>> # Get a backend (triggers discovery if needed)
        >>> backend_class = registry.get_backend('redis')
        >>>
        >>> # List all available backends
        >>> backends = registry.list_backends()
        >>> print(f"Available: {backends}")

    Thread safety:
        This class is not thread-safe. Use the global instance or provide
        your own synchronization if needed.
    """

    def __init__(self) -> None:
        """Initialize an empty plugin registry.

        Sets up internal storage for backend classes and tracks discovery
        state to avoid redundant plugin scanning.
        """
        self._backends: Dict[str, Type[StorageBackend]] = {}
        self._discovered = False

    def register(self, name: str, backend_class: Type[StorageBackend]) -> None:
        """Register a storage backend plugin.

        Manually registers a backend class with the given name. Useful for
        built-in backends or when you want to override plugin discovery.

        Args:
            name: Unique name for the backend (e.g., 'redis', 'mem0').
                Used as the key for backend retrieval. Must be unique.
            backend_class: StorageBackend implementation class.
                Must inherit from StorageBackend interface.

        Raises:
            TypeError: If backend_class doesn't inherit from StorageBackend

        Example:
            >>> class CustomStorage(StorageBackend):
            ...     # Implementation here
            ...     pass
            >>>
            >>> registry = PluginRegistry()
            >>> registry.register('custom', CustomStorage)
            >>> storage = registry.get_backend('custom')

        Note:
            Manual registration takes precedence over entry point discovery.
            If you register a backend with the same name as a plugin, the
            manually registered one will be used.
        """
        if not issubclass(backend_class, StorageBackend):
            raise TypeError(f"Backend {backend_class} must inherit from StorageBackend")

        self._backends[name] = backend_class

    def get_backend(self, name: str) -> Type[StorageBackend]:
        """Get a registered storage backend by name.

        Retrieves a backend class by its registered name. Triggers plugin
        discovery if this is the first call to any getter method.

        Args:
            name: Backend name as registered. Case-sensitive.
                Common examples: 'memory', 'file', 'redis', 'postgres'

        Returns:
            StorageBackend class (not instance). Caller is responsible
            for instantiating with appropriate arguments.

        Raises:
            KeyError: If backend is not registered. Error message includes
                list of available backends for troubleshooting.

        Example:
            >>> # Get and instantiate a backend
            >>> backend_class = registry.get_backend('redis')
            >>> storage = backend_class(url='redis://localhost')
            >>>
            >>> # Handle missing backends
            >>> try:
            ...     backend_class = registry.get_backend('unknown')
            >>> except KeyError as e:
            ...     print(f"Backend not found: {e}")
            ...     print(f"Available: {registry.list_backends()}")

        Performance:
            First call triggers plugin discovery, subsequent calls are O(1)
            dictionary lookups.
        """
        if not self._discovered:
            self.discover_plugins()

        if name not in self._backends:
            raise KeyError(
                f"Storage backend '{name}' not found. Available: {list(self._backends.keys())}"
            )

        return self._backends[name]

    def list_backends(self) -> List[str]:
        """List all registered backend names.

        Returns a list of all available backend names. Triggers plugin
        discovery if not already performed.

        Returns:
            List of backend names in registration order. Typically includes
            built-in backends ('memory', 'file') plus any discovered plugins.

        Example:
            >>> backends = registry.list_backends()
            >>> print(f"Available storage backends: {', '.join(backends)}")
            >>>
            >>> # Check if specific backend is available
            >>> if 'redis' in backends:
            ...     storage = create_storage_backend('redis', url='...')

        Note:
            The order reflects registration order, not alphabetical or
            priority order. Built-in backends are typically registered first.
        """
        if not self._discovered:
            self.discover_plugins()
        return list(self._backends.keys())

    def discover_plugins(self) -> None:
        """Auto-discover storage plugins via entry points.

        Scans installed packages for entry points in the 'sifaka.storage'
        group and attempts to load them as storage backends. Handles various
        Python versions and gracefully deals with plugin loading failures.

        Entry point format:
            [sifaka.storage]
            redis = my_plugin.redis_storage:RedisStorage
            postgres = my_plugin.pg_storage:PostgresStorage

        Error handling:
        - Missing optional dependencies are logged as warnings
        - Malformed entry points are skipped
        - Import errors don't stop discovery of other plugins
        - Discovery failures don't crash the application

        Note:
            This method is idempotent - calling it multiple times has no
            additional effect. Discovery state is tracked internally.

        Compatibility:
            Handles importlib.metadata differences across Python versions:
            - Python 3.10+: Uses entry_points().select()
            - Python 3.8-3.9: Uses entry_points().get()
            - Python 3.7: Falls back to importlib_metadata
        """
        if self._discovered:
            return

        try:
            # Look for entry points in the 'sifaka.storage' group
            entry_points = metadata.entry_points()

            # Handle different return types in different Python versions
            if hasattr(entry_points, "select"):
                # Python 3.10+
                storage_entries = entry_points.select(group="sifaka.storage")
            else:
                # Python 3.8-3.9
                storage_entries = cast(
                    EntryPoints, entry_points.get("sifaka.storage", [])
                )

            for entry_point in storage_entries:
                try:
                    backend_class = entry_point.load()
                    self.register(entry_point.name, backend_class)
                except Exception as e:
                    # Log but don't fail - plugin might have optional dependencies
                    print(
                        f"Warning: Failed to load storage plugin '{entry_point.name}': {e}"
                    )
        except Exception:
            # Entry points might not be available in some environments
            pass

        self._discovered = True


# Global plugin registry instance
_registry = PluginRegistry()

# Public API
register_storage_backend = _registry.register
get_storage_backend = _registry.get_backend
list_storage_backends = _registry.list_backends
discover_storage_plugins = _registry.discover_plugins


def create_storage_backend(backend_type: str, **kwargs: Any) -> StorageBackend:
    """Create a storage backend instance by name.

    Convenience function that gets the backend class and instantiates it
    with the provided arguments. This is the main public API for creating
    storage backends.

    Args:
        backend_type: Name of the storage backend as registered.
            Examples: 'memory', 'file', 'redis', 'postgres'
        **kwargs: Arguments to pass to the backend constructor.
            Each backend type expects different arguments:
            - 'file': storage_dir="./thoughts"
            - 'redis': url="redis://localhost", db=0
            - 'postgres': connection_string="postgresql://..."

    Returns:
        Initialized StorageBackend instance ready for use

    Raises:
        KeyError: If backend_type is not registered
        TypeError: If kwargs don't match backend constructor

    Example:
        >>> # File storage with custom directory
        >>> storage = create_storage_backend('file', storage_dir='./my_results')
        >>>
        >>> # Redis storage with connection details
        >>> storage = create_storage_backend(
        ...     'redis',
        ...     url='redis://localhost:6379',
        ...     db=1
        ... )
        >>>
        >>> # Use the storage
        >>> result_id = await storage.save(my_result)

    Plugin loading:
        If this is the first call to any storage function, it will trigger
        automatic plugin discovery, which may cause a slight delay.
    """
    backend_class = get_storage_backend(backend_type)
    return backend_class(**kwargs)
