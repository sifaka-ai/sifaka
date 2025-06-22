"""Plugin discovery and registration system for Sifaka storage backends."""

from typing import Dict, Type, List, Any
import pkg_resources

from .storage.base import StorageBackend


class PluginRegistry:
    """Registry for storage backend plugins."""

    def __init__(self) -> None:
        self._backends: Dict[str, Type[StorageBackend]] = {}
        self._discovered = False

    def register(self, name: str, backend_class: Type[StorageBackend]) -> None:
        """Register a storage backend plugin.

        Args:
            name: Unique name for the backend (e.g., 'redis', 'mem0')
            backend_class: StorageBackend implementation class
        """
        if not issubclass(backend_class, StorageBackend):
            raise TypeError(f"Backend {backend_class} must inherit from StorageBackend")

        self._backends[name] = backend_class

    def get_backend(self, name: str) -> Type[StorageBackend]:
        """Get a registered storage backend by name.

        Args:
            name: Backend name

        Returns:
            StorageBackend class

        Raises:
            KeyError: If backend is not registered
        """
        if not self._discovered:
            self.discover_plugins()

        if name not in self._backends:
            raise KeyError(
                f"Storage backend '{name}' not found. Available: {list(self._backends.keys())}"
            )

        return self._backends[name]

    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        if not self._discovered:
            self.discover_plugins()
        return list(self._backends.keys())

    def discover_plugins(self) -> None:
        """Auto-discover storage plugins via entry points."""
        if self._discovered:
            return

        try:
            # Look for entry points in the 'sifaka.storage' group
            for entry_point in pkg_resources.iter_entry_points("sifaka.storage"):
                try:
                    backend_class = entry_point.load()
                    self.register(entry_point.name, backend_class)
                except Exception as e:
                    # Log but don't fail - plugin might have optional dependencies
                    print(
                        f"Warning: Failed to load storage plugin '{entry_point.name}': {e}"
                    )
        except Exception:
            # pkg_resources might not be available in some environments
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

    Args:
        backend_type: Name of the storage backend
        **kwargs: Arguments to pass to the backend constructor

    Returns:
        Initialized StorageBackend instance

    Example:
        >>> storage = create_storage_backend('redis', url='redis://localhost')
        >>> storage = create_storage_backend('mem0', user_id='demo')
    """
    backend_class = get_storage_backend(backend_type)
    return backend_class(**kwargs)
