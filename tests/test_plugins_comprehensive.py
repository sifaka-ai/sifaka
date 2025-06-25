"""Comprehensive tests for the plugin system."""

import pytest
from unittest.mock import patch
from typing import Any, Dict, List, Optional

from sifaka.core.plugins import (
    register_storage_backend,
    create_storage_backend,
    list_storage_backends,
    _registry,
)
from sifaka.storage.base import StorageBackend
from sifaka.core.models import (
    SifakaResult,
)
from sifaka.core.exceptions import ConfigurationError


class TestStorageBackend:
    """Test custom storage backend implementation."""

    def test_storage_backend_interface(self):
        """Test that storage backend interface is properly defined."""
        # Should not be able to instantiate abstract base class
        with pytest.raises(TypeError):
            StorageBackend()

    def test_custom_storage_backend_implementation(self):
        """Test implementing a custom storage backend."""

        class CustomStorageBackend(StorageBackend):
            def __init__(self):
                self.data = {}

            async def save(self, result: SifakaResult) -> str:
                self.data[result.id] = result
                return result.id

            async def load(self, result_id: str) -> Optional[SifakaResult]:
                return self.data.get(result_id)

            async def delete(self, result_id: str) -> bool:
                if result_id in self.data:
                    del self.data[result_id]
                    return True
                return False

            async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
                return list(self.data.keys())[offset : offset + limit]

            async def search(self, query: str, limit: int = 10) -> List[str]:
                # Simple search implementation
                # Could implement actual search logic here
                return []

        # Should be able to instantiate concrete implementation
        backend = CustomStorageBackend()
        assert hasattr(backend, "save")
        assert hasattr(backend, "load")
        assert hasattr(backend, "delete")
        assert hasattr(backend, "list")
        assert hasattr(backend, "search")

    @pytest.mark.asyncio
    async def test_custom_storage_backend_functionality(self):
        """Test custom storage backend functionality."""

        class TestStorageBackend(StorageBackend):
            def __init__(self):
                self.storage = {}

            async def save(self, result: SifakaResult) -> str:
                self.storage[result.id] = result
                return result.id

            async def load(self, result_id: str) -> Optional[SifakaResult]:
                return self.storage.get(result_id)

            async def delete(self, result_id: str) -> bool:
                return self.storage.pop(result_id, None) is not None

            async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
                return list(self.storage.keys())[offset : offset + limit]

            async def search(self, query: str, limit: int = 10) -> List[str]:
                # Simple search implementation
                # Could implement actual search logic here
                return []

        backend = TestStorageBackend()

        # Create a test result
        result = SifakaResult(
            original_text="test",
            final_text="improved test",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            total_cost=0.01,
            processing_time=1.0,
        )

        # Test save
        saved_id = await backend.save(result)
        assert saved_id == result.id

        # Test load
        loaded = await backend.load(result.id)
        assert loaded is not None
        assert loaded.id == result.id
        assert loaded.original_text == result.original_text

        # Test list
        result_ids = await backend.list()
        assert result.id in result_ids

        # Test delete
        deleted = await backend.delete(result.id)
        assert deleted is True

        # Test load after delete
        loaded = await backend.load(result.id)
        assert loaded is None


class TestPluginRegistration:
    """Test plugin registration system."""

    def setup_method(self):
        """Setup for each test method."""
        # Clear any existing registrations
        _registry._backends.clear()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Clear registrations to avoid affecting other tests
        _registry._backends.clear()

    def test_register_storage_backend(self):
        """Test registering a storage backend."""

        class TestBackend(StorageBackend):
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

        register_storage_backend("test", TestBackend)

        backends = list_storage_backends()
        assert "test" in backends

    def test_register_storage_backend_duplicate(self):
        """Test registering duplicate storage backend name."""

        class TestBackend1(StorageBackend):
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

        class TestBackend2(StorageBackend):
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

        # First registration should work
        register_storage_backend("test", TestBackend1)

        # Second registration should overwrite (or warn, depending on implementation)
        register_storage_backend("test", TestBackend2)

        backends = list_storage_backends()
        assert "test" in backends

    def test_register_invalid_backend(self):
        """Test registering invalid backend."""

        class NotABackend:
            pass

        with pytest.raises((TypeError, ValueError)):
            register_storage_backend("invalid", NotABackend)

    def test_get_registered_backends_empty(self):
        """Test getting registered backends when none are registered."""
        backends = list_storage_backends()
        assert isinstance(backends, list)
        assert len(backends) == 0

    def test_get_registered_backends_multiple(self):
        """Test getting multiple registered backends."""

        class Backend1(StorageBackend):
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

        class Backend2(StorageBackend):
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

        register_storage_backend("backend1", Backend1)
        register_storage_backend("backend2", Backend2)

        backends = list_storage_backends()
        assert len(backends) == 2
        assert "backend1" in backends
        assert "backend2" in backends


class TestPluginFactory:
    """Test plugin factory functionality."""

    def setup_method(self):
        """Setup for each test method."""
        _registry._backends.clear()

    def teardown_method(self):
        """Cleanup after each test method."""
        _registry._backends.clear()

    def test_create_storage_backend_registered(self):
        """Test creating a registered storage backend."""

        class TestBackend(StorageBackend):
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

        register_storage_backend("test", TestBackend)

        # Create without args
        backend = create_storage_backend("test")
        assert isinstance(backend, TestBackend)
        assert backend.config == "default"

        # Create with args
        backend = create_storage_backend("test", config="custom")
        assert isinstance(backend, TestBackend)
        assert backend.config == "custom"

    def test_create_storage_backend_unregistered(self):
        """Test creating an unregistered storage backend."""
        with pytest.raises(KeyError, match="not found"):
            create_storage_backend("nonexistent")

    def test_create_storage_backend_with_kwargs(self):
        """Test creating storage backend with keyword arguments."""

        class ConfigurableBackend(StorageBackend):
            def __init__(self, max_size: int = 100, prefix: str = "test"):
                self.max_size = max_size
                self.prefix = prefix

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

        register_storage_backend("configurable", ConfigurableBackend)

        backend = create_storage_backend("configurable", max_size=500, prefix="prod")
        assert backend.max_size == 500
        assert backend.prefix == "prod"

    def test_create_storage_backend_invalid_args(self):
        """Test creating storage backend with invalid arguments."""

        class SimpleBackend(StorageBackend):
            def __init__(self):
                pass

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

        register_storage_backend("simple", SimpleBackend)

        # Should work without args
        backend = create_storage_backend("simple")
        assert isinstance(backend, SimpleBackend)

        # Should fail with unexpected args
        with pytest.raises(TypeError):
            create_storage_backend("simple", unexpected_arg="value")


class TestPluginIntegration:
    """Test plugin integration with the main system."""

    def setup_method(self):
        """Setup for each test method."""
        _registry._backends.clear()

    def teardown_method(self):
        """Cleanup after each test method."""
        _registry._backends.clear()

    @pytest.mark.asyncio
    async def test_custom_storage_integration(self):
        """Test integration of custom storage with improve function."""

        class MemoryStorageBackend(StorageBackend):
            def __init__(self):
                self.storage = {}

            async def save(self, result: SifakaResult) -> str:
                self.storage[result.id] = result
                return result.id

            async def load(self, result_id: str) -> Optional[SifakaResult]:
                return self.storage.get(result_id)

            async def delete(self, result_id: str) -> bool:
                return self.storage.pop(result_id, None) is not None

            async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
                return list(self.storage.keys())[offset : offset + limit]

            async def search(self, query: str, limit: int = 10) -> List[str]:
                # Simple search implementation
                # Could implement actual search logic here
                return []

        register_storage_backend("memory", MemoryStorageBackend)
        storage = create_storage_backend("memory")

        # Mock the improve function to use our custom storage
        with patch("sifaka.improve") as mock_improve:
            # Create a mock result
            mock_result = SifakaResult(
                original_text="test input",
                final_text="improved output",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                total_cost=0.01,
                processing_time=1.0,
            )

            # Configure mock to call our storage
            async def mock_improve_func(*args, **kwargs):
                if "storage" in kwargs:
                    await kwargs["storage"].save(mock_result)
                return mock_result

            mock_improve.side_effect = mock_improve_func

            # Test the integration
            from sifaka import improve

            result = await improve("test input", storage=storage)

            # Verify result was saved to our custom storage
            loaded = await storage.load(result.id)
            assert loaded is not None
            assert loaded.id == result.id

    def test_plugin_configuration_validation(self):
        """Test validation of plugin configuration."""

        class StrictBackend(StorageBackend):
            def __init__(self, required_param: str):
                if not required_param:
                    raise ValueError("required_param cannot be empty")
                self.required_param = required_param

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

        register_storage_backend("strict", StrictBackend)

        # Should work with valid parameter
        backend = create_storage_backend("strict", required_param="valid")
        assert backend.required_param == "valid"

        # Should fail with invalid parameter
        with pytest.raises(ValueError):
            create_storage_backend("strict", required_param="")


class TestAdvancedPluginFeatures:
    """Test advanced plugin features and edge cases."""

    def setup_method(self):
        """Setup for each test method."""
        _registry._backends.clear()

    def teardown_method(self):
        """Cleanup after each test method."""
        _registry._backends.clear()

    def test_plugin_inheritance(self):
        """Test plugin inheritance and method overriding."""

        class BaseBackend(StorageBackend):
            def __init__(self, name: str = "base"):
                self.name = name

            async def save(self, result: SifakaResult) -> str:
                return f"{self.name}_{result.id}"

            async def load(self, result_id: str) -> Optional[SifakaResult]:
                return None

            async def delete(self, result_id: str) -> bool:
                return True

            async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
                return []

            async def search(self, query: str, limit: int = 10) -> List[str]:
                return []

        class ExtendedBackend(BaseBackend):
            def __init__(self, name: str = "extended", prefix: str = "ext"):
                super().__init__(name)
                self.prefix = prefix

            async def save(self, result: SifakaResult) -> str:
                base_result = await super().save(result)
                return f"{self.prefix}_{base_result}"

        register_storage_backend("extended", ExtendedBackend)

        backend = create_storage_backend("extended", name="test", prefix="custom")
        assert backend.name == "test"
        assert backend.prefix == "custom"

    @pytest.mark.asyncio
    async def test_plugin_async_initialization(self):
        """Test plugins with async initialization."""

        class AsyncInitBackend(StorageBackend):
            def __init__(self):
                self.initialized = False
                self.connection = None

            async def initialize(self):
                """Custom async initialization method."""
                self.connection = "mock_connection"
                self.initialized = True

            async def save(self, result: SifakaResult) -> str:
                if not self.initialized:
                    await self.initialize()
                return result.id

            async def load(self, result_id: str) -> Optional[SifakaResult]:
                return None

            async def delete(self, result_id: str) -> bool:
                return True

            async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
                return []

            async def search(self, query: str, limit: int = 10) -> List[str]:
                return []

        register_storage_backend("async_init", AsyncInitBackend)
        backend = create_storage_backend("async_init")

        # Create a test result and save it (should trigger initialization)
        result = SifakaResult(
            original_text="test",
            final_text="test",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            total_cost=0.01,
            processing_time=1.0,
        )

        await backend.save(result)
        assert backend.initialized is True
        assert backend.connection == "mock_connection"

    def test_plugin_with_complex_configuration(self):
        """Test plugin with complex configuration."""

        class ComplexBackend(StorageBackend):
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.validate_config()

            def validate_config(self):
                required_keys = ["host", "port", "database"]
                for key in required_keys:
                    if key not in self.config:
                        raise ConfigurationError(f"Missing required config key: {key}")

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

        register_storage_backend("complex", ComplexBackend)

        # Valid configuration
        valid_config = {
            "host": "localhost",
            "port": 5432,
            "database": "sifaka",
            "username": "user",
            "password": "pass",
        }

        backend = create_storage_backend("complex", config=valid_config)
        assert backend.config == valid_config

        # Invalid configuration
        invalid_config = {"host": "localhost"}

        with pytest.raises(ConfigurationError):
            create_storage_backend("complex", config=invalid_config)

    @pytest.mark.asyncio
    async def test_plugin_error_handling(self):
        """Test plugin error handling."""

        class ErrorProneBackend(StorageBackend):
            def __init__(self, fail_on_save: bool = False):
                self.fail_on_save = fail_on_save

            async def save(self, result: SifakaResult) -> str:
                if self.fail_on_save:
                    raise RuntimeError("Simulated save failure")
                return result.id

            async def load(self, result_id: str) -> Optional[SifakaResult]:
                if result_id == "error":
                    raise RuntimeError("Simulated load failure")
                return None

            async def delete(self, result_id: str) -> bool:
                return True

            async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
                return []

            async def search(self, query: str, limit: int = 10) -> List[str]:
                return []

        register_storage_backend("error_prone", ErrorProneBackend)

        # Test save error
        backend = create_storage_backend("error_prone", fail_on_save=True)
        result = SifakaResult(
            original_text="test",
            final_text="test",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            total_cost=0.01,
            processing_time=1.0,
        )

        with pytest.raises(RuntimeError, match="Simulated save failure"):
            await backend.save(result)

        # Test load error
        backend = create_storage_backend("error_prone", fail_on_save=False)
        with pytest.raises(RuntimeError, match="Simulated load failure"):
            await backend.load("error")

    def test_plugin_registration_edge_cases(self):
        """Test edge cases in plugin registration."""

        # Test registering with None name
        class TestBackend(StorageBackend):
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

        # The current implementation allows None as a name
        # This might not be ideal, but we test the actual behavior
        register_storage_backend(None, TestBackend)
        assert None in list_storage_backends()
        _registry._backends.clear()

        # Test registering with empty string name - this is allowed
        register_storage_backend("", TestBackend)
        assert "" in list_storage_backends()
        _registry._backends.clear()

        # Test registering with None backend
        with pytest.raises((ValueError, TypeError)):
            register_storage_backend("test", None)

    def test_plugin_registry_isolation(self):
        """Test that plugin registry is properly isolated."""

        # Register a backend
        class TestBackend(StorageBackend):
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

        register_storage_backend("isolation_test", TestBackend)

        # Verify it's registered
        assert "isolation_test" in list_storage_backends()

        # Clear the registry
        _registry._backends.clear()

        # Verify it's no longer registered
        assert "isolation_test" not in list_storage_backends()

    def test_multiple_plugin_instances(self):
        """Test creating multiple instances of the same plugin."""

        class StatefulBackend(StorageBackend):
            def __init__(self, instance_id: str):
                self.instance_id = instance_id
                self.call_count = 0

            async def save(self, result: SifakaResult) -> str:
                self.call_count += 1
                return f"{self.instance_id}_{result.id}_{self.call_count}"

            async def load(self, result_id: str) -> Optional[SifakaResult]:
                return None

            async def delete(self, result_id: str) -> bool:
                return True

            async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
                return []

            async def search(self, query: str, limit: int = 10) -> List[str]:
                return []

        register_storage_backend("stateful", StatefulBackend)

        # Create multiple instances
        backend1 = create_storage_backend("stateful", instance_id="instance1")
        backend2 = create_storage_backend("stateful", instance_id="instance2")

        assert backend1.instance_id == "instance1"
        assert backend2.instance_id == "instance2"
        assert backend1.call_count == 0
        assert backend2.call_count == 0

        # Verify they maintain separate state
        backend1.call_count = 5
        assert backend2.call_count == 0
