"""Comprehensive unit tests for Flexible Hybrid storage backend.

This module tests the FlexibleHybridPersistence implementation:
- Multi-backend storage with configurable routing
- Priority-based read/write operations
- Failover and read repair functionality
- Backend role management and configuration

Tests cover:
- Multi-backend coordination
- Priority-based routing logic
- Failover scenarios and recovery
- Backend configuration and roles
- Mock-based testing without actual backend connections
"""

from unittest.mock import AsyncMock, Mock

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.storage.flexible_hybrid import BackendConfig, BackendRole, FlexibleHybridPersistence


class TestFlexibleHybridPersistence:
    """Test suite for FlexibleHybridPersistence class."""

    @pytest.fixture
    def mock_backends(self):
        """Create mock storage backends."""
        backends = {}

        # Cache backend (highest priority)
        cache_backend = Mock()
        cache_backend.store_thought = AsyncMock()
        cache_backend.retrieve_thought = AsyncMock()
        cache_backend.delete_thought = AsyncMock()
        cache_backend._store_raw = AsyncMock()
        cache_backend._retrieve_raw = AsyncMock()
        cache_backend._delete_raw = AsyncMock()
        cache_backend._list_keys = AsyncMock()
        backends["cache"] = cache_backend

        # Primary backend
        primary_backend = Mock()
        primary_backend.store_thought = AsyncMock()
        primary_backend.retrieve_thought = AsyncMock()
        primary_backend.delete_thought = AsyncMock()
        primary_backend._store_raw = AsyncMock()
        primary_backend._retrieve_raw = AsyncMock()
        primary_backend._delete_raw = AsyncMock()
        primary_backend._list_keys = AsyncMock()
        backends["primary"] = primary_backend

        # Backup backend
        backup_backend = Mock()
        backup_backend.store_thought = AsyncMock()
        backup_backend.retrieve_thought = AsyncMock()
        backup_backend.delete_thought = AsyncMock()
        backup_backend._store_raw = AsyncMock()
        backup_backend._retrieve_raw = AsyncMock()
        backup_backend._delete_raw = AsyncMock()
        backup_backend._list_keys = AsyncMock()
        backends["backup"] = backup_backend

        return backends

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Test flexible hybrid storage",
            final_text="This is a test thought for flexible hybrid storage.",
            iteration=1,
            max_iterations=3,
        )
        thought.add_generation("Generated text", "gpt-4", {"temperature": 0.7})
        thought.add_validation("length_validator", True, {"word_count": 12})
        return thought

    def test_backend_config_creation(self):
        """Test BackendConfig creation and validation."""
        config = BackendConfig(
            backend=Mock(),
            role=BackendRole.CACHE,
            priority=1,
            read_enabled=True,
            write_enabled=True,
        )

        assert config.role == BackendRole.CACHE
        assert config.priority == 1
        assert config.read_enabled is True
        assert config.write_enabled is True

    def test_flexible_hybrid_creation_minimal(self, mock_backends):
        """Test creating FlexibleHybridPersistence with minimal configuration."""
        backend_configs = [BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 1)]

        persistence = FlexibleHybridPersistence(backend_configs)

        assert len(persistence.backend_configs) == 1
        assert persistence.backend_configs[0].role == BackendRole.PRIMARY

    def test_flexible_hybrid_creation_multi_backend(self, mock_backends):
        """Test creating FlexibleHybridPersistence with multiple backends."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
            BackendConfig(mock_backends["backup"], BackendRole.BACKUP, 3),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        assert len(persistence.backend_configs) == 3

        # Verify priority ordering
        priorities = [config.priority for config in persistence.backend_configs]
        assert priorities == sorted(priorities)

    @pytest.mark.asyncio
    async def test_store_thought_single_backend(self, mock_backends, sample_thought):
        """Test storing thought with single backend."""
        backend_configs = [BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 1)]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock successful storage
        mock_backends["primary"].store_thought.return_value = None

        await persistence.store_thought(sample_thought)

        # Verify storage was called
        mock_backends["primary"].store_thought.assert_called_once_with(sample_thought)

    @pytest.mark.asyncio
    async def test_store_thought_multi_backend(self, mock_backends, sample_thought):
        """Test storing thought with multiple backends."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock successful storage
        mock_backends["cache"].store_thought.return_value = None
        mock_backends["primary"].store_thought.return_value = None

        await persistence.store_thought(sample_thought)

        # Verify storage was called on both backends
        mock_backends["cache"].store_thought.assert_called_once_with(sample_thought)
        mock_backends["primary"].store_thought.assert_called_once_with(sample_thought)

    @pytest.mark.asyncio
    async def test_retrieve_thought_cache_hit(self, mock_backends, sample_thought):
        """Test retrieving thought with cache hit."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock cache hit
        mock_backends["cache"].retrieve_thought.return_value = sample_thought

        result = await persistence.retrieve_thought(sample_thought.id)

        # Verify cache was checked first
        mock_backends["cache"].retrieve_thought.assert_called_once_with(sample_thought.id)

        # Verify primary was not called (cache hit)
        mock_backends["primary"].retrieve_thought.assert_not_called()

        assert result == sample_thought

    @pytest.mark.asyncio
    async def test_retrieve_thought_cache_miss_primary_hit(self, mock_backends, sample_thought):
        """Test retrieving thought with cache miss but primary hit."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock cache miss, primary hit
        mock_backends["cache"].retrieve_thought.return_value = None
        mock_backends["primary"].retrieve_thought.return_value = sample_thought

        # Mock cache write for read repair
        mock_backends["cache"].store_thought.return_value = None

        result = await persistence.retrieve_thought(sample_thought.id)

        # Verify both backends were checked
        mock_backends["cache"].retrieve_thought.assert_called_once_with(sample_thought.id)
        mock_backends["primary"].retrieve_thought.assert_called_once_with(sample_thought.id)

        # Verify read repair (cache write)
        mock_backends["cache"].store_thought.assert_called_once_with(sample_thought)

        assert result == sample_thought

    @pytest.mark.asyncio
    async def test_retrieve_thought_all_miss(self, mock_backends):
        """Test retrieving thought when all backends miss."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
            BackendConfig(mock_backends["backup"], BackendRole.BACKUP, 3),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock all misses
        mock_backends["cache"].retrieve_thought.return_value = None
        mock_backends["primary"].retrieve_thought.return_value = None
        mock_backends["backup"].retrieve_thought.return_value = None

        result = await persistence.retrieve_thought("nonexistent-id")

        # Verify all backends were checked
        mock_backends["cache"].retrieve_thought.assert_called_once()
        mock_backends["primary"].retrieve_thought.assert_called_once()
        mock_backends["backup"].retrieve_thought.assert_called_once()

        assert result is None

    @pytest.mark.asyncio
    async def test_store_thought_with_failover(self, mock_backends, sample_thought):
        """Test storing thought with backend failover."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock cache failure, primary success
        mock_backends["cache"].store_thought.side_effect = Exception("Cache error")
        mock_backends["primary"].store_thought.return_value = None

        await persistence.store_thought(sample_thought)

        # Verify both backends were attempted
        mock_backends["cache"].store_thought.assert_called_once_with(sample_thought)
        mock_backends["primary"].store_thought.assert_called_once_with(sample_thought)

    @pytest.mark.asyncio
    async def test_delete_thought_multi_backend(self, mock_backends, sample_thought):
        """Test deleting thought from multiple backends."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock successful deletion
        mock_backends["cache"].delete_thought.return_value = True
        mock_backends["primary"].delete_thought.return_value = True

        result = await persistence.delete_thought(sample_thought.id)

        # Verify deletion was called on both backends
        mock_backends["cache"].delete_thought.assert_called_once_with(sample_thought.id)
        mock_backends["primary"].delete_thought.assert_called_once_with(sample_thought.id)

        assert result is True

    @pytest.mark.asyncio
    async def test_store_raw_priority_routing(self, mock_backends):
        """Test raw storage with priority-based routing."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        test_key = "test:key"
        test_data = "test data"

        # Mock successful storage
        mock_backends["cache"]._store_raw.return_value = None
        mock_backends["primary"]._store_raw.return_value = None

        await persistence._store_raw(test_key, test_data)

        # Verify storage was called on both backends
        mock_backends["cache"]._store_raw.assert_called_once_with(test_key, test_data)
        mock_backends["primary"]._store_raw.assert_called_once_with(test_key, test_data)

    @pytest.mark.asyncio
    async def test_retrieve_raw_priority_routing(self, mock_backends):
        """Test raw retrieval with priority-based routing."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        test_key = "test:key"
        expected_data = "test data"

        # Mock cache hit
        mock_backends["cache"]._retrieve_raw.return_value = expected_data

        result = await persistence._retrieve_raw(test_key)

        # Verify cache was checked first
        mock_backends["cache"]._retrieve_raw.assert_called_once_with(test_key)

        # Verify primary was not called (cache hit)
        mock_backends["primary"]._retrieve_raw.assert_not_called()

        assert result == expected_data

    @pytest.mark.asyncio
    async def test_backend_role_filtering(self, mock_backends):
        """Test backend filtering by role."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
            BackendConfig(mock_backends["backup"], BackendRole.BACKUP, 3),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Test getting backends by role
        cache_backends = [
            config for config in persistence.backend_configs if config.role == BackendRole.CACHE
        ]
        primary_backends = [
            config for config in persistence.backend_configs if config.role == BackendRole.PRIMARY
        ]
        backup_backends = [
            config for config in persistence.backend_configs if config.role == BackendRole.BACKUP
        ]

        assert len(cache_backends) == 1
        assert len(primary_backends) == 1
        assert len(backup_backends) == 1

    @pytest.mark.asyncio
    async def test_read_write_enabled_flags(self, mock_backends, sample_thought):
        """Test read/write enabled flags."""
        backend_configs = [
            BackendConfig(
                mock_backends["cache"], BackendRole.CACHE, 1, read_enabled=True, write_enabled=False
            ),
            BackendConfig(
                mock_backends["primary"],
                BackendRole.PRIMARY,
                2,
                read_enabled=True,
                write_enabled=True,
            ),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock responses
        mock_backends["cache"].retrieve_thought.return_value = sample_thought
        mock_backends["primary"].store_thought.return_value = None

        # Test read operation (should use cache)
        result = await persistence.retrieve_thought(sample_thought.id)
        mock_backends["cache"].retrieve_thought.assert_called_once()
        assert result == sample_thought

        # Test write operation (should only use primary, not cache)
        await persistence.store_thought(sample_thought)
        mock_backends["primary"].store_thought.assert_called_once()
        # Cache should not be called for write since write_enabled=False
        mock_backends["cache"].store_thought.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_keys_aggregation(self, mock_backends):
        """Test key listing aggregation across backends."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        pattern = "test:*"

        # Mock key listings
        mock_backends["cache"]._list_keys.return_value = ["test:key1", "test:key2"]
        mock_backends["primary"]._list_keys.return_value = ["test:key2", "test:key3"]

        result = await persistence._list_keys(pattern)

        # Verify both backends were queried
        mock_backends["cache"]._list_keys.assert_called_once_with(pattern)
        mock_backends["primary"]._list_keys.assert_called_once_with(pattern)

        # Verify results are deduplicated and sorted
        expected_keys = ["test:key1", "test:key2", "test:key3"]
        assert sorted(result) == expected_keys

    @pytest.mark.asyncio
    async def test_error_handling_all_backends_fail(self, mock_backends, sample_thought):
        """Test error handling when all backends fail."""
        backend_configs = [
            BackendConfig(mock_backends["cache"], BackendRole.CACHE, 1),
            BackendConfig(mock_backends["primary"], BackendRole.PRIMARY, 2),
        ]

        persistence = FlexibleHybridPersistence(backend_configs)

        # Mock all backends failing
        mock_backends["cache"].store_thought.side_effect = Exception("Cache error")
        mock_backends["primary"].store_thought.side_effect = Exception("Primary error")

        with pytest.raises(Exception):
            await persistence.store_thought(sample_thought)

    def test_backend_role_enum_values(self):
        """Test BackendRole enum values."""
        assert BackendRole.CACHE.value == "cache"
        assert BackendRole.PRIMARY.value == "primary"
        assert BackendRole.BACKUP.value == "backup"
        assert BackendRole.SEARCH.value == "search"
        assert BackendRole.ARCHIVE.value == "archive"
