"""Tests for plugin discovery mechanism and entry point registration."""

from typing import List, Optional
from unittest.mock import MagicMock, patch

from sifaka.core.models import SifakaResult
from sifaka.core.plugins import (
    PluginRegistry,
    create_storage_backend,
    discover_storage_plugins,
    list_storage_backends,
)
from sifaka.storage.base import StorageBackend


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""

    def __init__(self, config: str = "default"):
        self.config = config

    async def save(self, result: SifakaResult) -> str:
        return result.id

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        return None

    async def delete(self, result_id: str) -> bool:
        return True

    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        return []

    async def search(self, query: str, limit: int = 10) -> List[str]:
        return []


class TestPluginDiscovery:
    """Test plugin discovery via entry points."""

    def test_discover_plugins_success(self):
        """Test successful plugin discovery."""
        # Create mock entry points
        mock_entry_point1 = MagicMock()
        mock_entry_point1.name = "test_backend_1"
        mock_entry_point1.load.return_value = MockStorageBackend

        mock_entry_point2 = MagicMock()
        mock_entry_point2.name = "test_backend_2"
        mock_entry_point2.load.return_value = MockStorageBackend

        # Patch iter_entry_points to return our mocks
        with patch("pkg_resources.iter_entry_points") as mock_iter:
            mock_iter.return_value = [mock_entry_point1, mock_entry_point2]

            # Create a new registry to test discovery
            registry = PluginRegistry()
            registry.discover_plugins()

            # Verify plugins were discovered
            backends = registry.list_backends()
            assert "test_backend_1" in backends
            assert "test_backend_2" in backends

            # Verify entry points were loaded
            mock_entry_point1.load.assert_called_once()
            mock_entry_point2.load.assert_called_once()

    def test_discover_plugins_with_load_error(self):
        """Test plugin discovery handles load errors gracefully."""
        # Create mock entry points
        mock_entry_point1 = MagicMock()
        mock_entry_point1.name = "working_backend"
        mock_entry_point1.load.return_value = MockStorageBackend

        mock_entry_point2 = MagicMock()
        mock_entry_point2.name = "broken_backend"
        mock_entry_point2.load.side_effect = ImportError("Missing dependency")

        mock_entry_point3 = MagicMock()
        mock_entry_point3.name = "another_working"
        mock_entry_point3.load.return_value = MockStorageBackend

        with patch("pkg_resources.iter_entry_points") as mock_iter:
            mock_iter.return_value = [
                mock_entry_point1,
                mock_entry_point2,
                mock_entry_point3,
            ]

            # Capture print output
            with patch("builtins.print") as mock_print:
                registry = PluginRegistry()
                registry.discover_plugins()

                # Verify working plugins were loaded
                backends = registry.list_backends()
                assert "working_backend" in backends
                assert "broken_backend" not in backends
                assert "another_working" in backends

                # Verify warning was printed
                mock_print.assert_called()
                warning_call = mock_print.call_args_list[0]
                assert "Failed to load storage plugin 'broken_backend'" in str(
                    warning_call
                )

    def test_discover_plugins_called_once(self):
        """Test that discover_plugins is only called once."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "test_backend"
        mock_entry_point.load.return_value = MockStorageBackend

        with patch("sifaka.core.plugins.metadata.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps

            registry = PluginRegistry()

            # Call discover_plugins multiple times
            registry.discover_plugins()
            registry.discover_plugins()
            registry.discover_plugins()

            # Verify entry_points was only called once
            mock_entry_points.assert_called_once()

    def test_discover_plugins_no_pkg_resources(self):
        """Test plugin discovery when pkg_resources is not available."""
        with patch(
            "pkg_resources.iter_entry_points",
            side_effect=ImportError("No pkg_resources"),
        ):
            registry = PluginRegistry()

            # Should not raise an error
            registry.discover_plugins()

            # Registry should still work with manual registration
            registry.register("manual", MockStorageBackend)
            assert "manual" in registry.list_backends()

    def test_lazy_discovery_on_get_backend(self):
        """Test that discovery happens lazily when getting a backend."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "lazy_backend"
        mock_entry_point.load.return_value = MockStorageBackend

        with patch("sifaka.core.plugins.metadata.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps

            registry = PluginRegistry()

            # Discovery should not have happened yet
            assert registry._discovered is False

            # Getting a backend should trigger discovery
            try:
                registry.get_backend("lazy_backend")
            except KeyError:
                pass  # Expected if backend wasn't manually registered

            # Verify discovery was triggered
            assert registry._discovered is True
            mock_entry_points.assert_called()

    def test_lazy_discovery_on_list_backends(self):
        """Test that discovery happens lazily when listing backends."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "lazy_backend"
        mock_entry_point.load.return_value = MockStorageBackend

        with patch("sifaka.core.plugins.metadata.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps

            registry = PluginRegistry()

            # Discovery should not have happened yet
            assert registry._discovered is False

            # Listing backends should trigger discovery
            backends = registry.list_backends()

            # Verify discovery was triggered
            assert registry._discovered is True
            assert "lazy_backend" in backends
            mock_entry_points.assert_called()

    def test_entry_point_with_invalid_backend(self):
        """Test handling of entry points that don't return StorageBackend."""

        class NotABackend:
            """Not a storage backend."""

        mock_entry_point = MagicMock()
        mock_entry_point.name = "invalid_backend"
        mock_entry_point.load.return_value = NotABackend

        with patch("sifaka.core.plugins.metadata.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps

            with patch("builtins.print") as mock_print:
                registry = PluginRegistry()
                registry.discover_plugins()

                # Backend should not be registered
                assert "invalid_backend" not in registry.list_backends()

                # Warning should be printed
                mock_print.assert_called()
                warning_call = mock_print.call_args_list[0]
                assert "Failed to load storage plugin 'invalid_backend'" in str(
                    warning_call
                )

    def test_global_plugin_discovery(self):
        """Test the global plugin discovery function."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "global_test"
        mock_entry_point.load.return_value = MockStorageBackend

        with patch("sifaka.core.plugins.metadata.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps

            # Reset the global registry
            from sifaka.core.plugins import _registry

            _registry._discovered = False
            _registry._backends.clear()

            # Use the global function
            discover_storage_plugins()

            # Verify plugin was discovered
            backends = list_storage_backends()
            assert "global_test" in backends

    def test_create_backend_triggers_discovery(self):
        """Test that creating a backend triggers discovery."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "auto_discovered"
        mock_entry_point.load.return_value = MockStorageBackend

        with patch("sifaka.core.plugins.metadata.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps

            # Reset the global registry
            from sifaka.core.plugins import _registry

            _registry._discovered = False
            _registry._backends.clear()

            # Try to create a backend
            backend = create_storage_backend("auto_discovered", config="test")

            # Verify backend was created
            assert isinstance(backend, MockStorageBackend)
            assert backend.config == "test"

            # Verify discovery happened
            mock_entry_points.assert_called()


class TestPluginRegistrationEdgeCases:
    """Test edge cases in plugin registration."""

    def test_register_with_empty_name(self):
        """Test registering a plugin with empty name."""
        from sifaka.core.plugins import _registry

        # Should allow empty string (though not recommended)
        _registry.register("", MockStorageBackend)
        assert "" in _registry.list_backends()

        # Clean up
        _registry._backends.pop("", None)

    def test_register_overwrites_existing(self):
        """Test that registering overwrites existing plugins."""
        from sifaka.core.plugins import _registry

        class Backend1(MockStorageBackend):
            version = 1

        class Backend2(MockStorageBackend):
            version = 2

        _registry.register("overwrite_test", Backend1)
        backend_class = _registry.get_backend("overwrite_test")
        assert backend_class.version == 1

        _registry.register("overwrite_test", Backend2)
        backend_class = _registry.get_backend("overwrite_test")
        assert backend_class.version == 2

        # Clean up
        _registry._backends.pop("overwrite_test", None)

    def test_entry_point_group_name(self):
        """Test that the correct entry point group name is used."""
        with patch("pkg_resources.iter_entry_points") as mock_iter:
            mock_iter.return_value = []

            registry = PluginRegistry()
            registry.discover_plugins()

            # Verify the correct group name was used
            mock_iter.assert_called_once_with("sifaka.storage")


class TestPluginIsolation:
    """Test plugin isolation and thread safety."""

    def test_registry_isolation(self):
        """Test that plugin registries are isolated."""
        registry1 = PluginRegistry()
        registry2 = PluginRegistry()

        registry1.register("isolated1", MockStorageBackend)
        registry2.register("isolated2", MockStorageBackend)

        assert "isolated1" in registry1.list_backends()
        assert "isolated1" not in registry2.list_backends()
        assert "isolated2" in registry2.list_backends()
        assert "isolated2" not in registry1.list_backends()

    def test_manual_vs_discovered_plugins(self):
        """Test interaction between manual and discovered plugins."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "discovered"
        mock_entry_point.load.return_value = MockStorageBackend

        with patch("sifaka.core.plugins.metadata.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps

            registry = PluginRegistry()

            # Manually register a plugin
            registry.register("manual", MockStorageBackend)

            # Trigger discovery
            registry.discover_plugins()

            # Both should be available
            backends = registry.list_backends()
            assert "manual" in backends
            assert "discovered" in backends
