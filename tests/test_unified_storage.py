"""Tests for unified storage architecture.

This module tests the core functionality of the unified 3-tier storage
system without requiring external dependencies like Redis or Milvus.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from sifaka.core.thought import Thought
from sifaka.retrievers import InMemoryRetriever
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage.base import InMemoryStorage, StorageError
from sifaka.storage.thoughts import CachedThoughtStorage
from sifaka.storage.checkpoints import CachedCheckpointStorage, ChainCheckpoint
from sifaka.storage.retrievers import CachedRetriever
from sifaka.storage.metrics import CachedMetricsStorage, PerformanceMetric
from sifaka.storage.manager import SifakaStorage


class TestInMemoryStorage:
    """Test the in-memory storage component."""

    def test_basic_operations(self):
        """Test basic get/set operations."""
        storage = InMemoryStorage(max_size=3)

        # Test set and get
        storage.set("key1", "value1")
        assert storage.get("key1") == "value1"

        # Test miss
        assert storage.get("nonexistent") is None

        # Test stats
        stats = storage.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_lru_eviction(self):
        """Test LRU eviction behavior."""
        storage = InMemoryStorage(max_size=2)

        # Fill to capacity
        storage.set("key1", "value1")
        storage.set("key2", "value2")

        # Access key1 to make it recently used
        storage.get("key1")

        # Add key3, should evict key2 (least recently used)
        storage.set("key3", "value3")

        assert storage.get("key1") == "value1"  # Still there
        assert storage.get("key2") is None  # Evicted
        assert storage.get("key3") == "value3"  # New item

    def test_clear(self):
        """Test clearing storage."""
        storage = InMemoryStorage()
        storage.set("key1", "value1")
        storage.set("key2", "value2")

        storage.clear()

        assert storage.get("key1") is None
        assert storage.get("key2") is None
        assert storage.get_stats()["size"] == 0


class TestCachedThoughtStorage:
    """Test the cached thought storage."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock CachedStorage for testing."""
        mock = Mock()
        mock.get.return_value = None
        mock.set.return_value = None
        mock.search_similar.return_value = []
        return mock

    @pytest.fixture
    def thought_storage(self, mock_storage):
        """Create a CachedThoughtStorage with mock backend."""
        return CachedThoughtStorage(mock_storage)

    def test_save_thought(self, thought_storage, mock_storage):
        """Test saving a thought."""
        thought = Thought(prompt="Test prompt", text="Test text")

        thought_storage.save_thought(thought)

        # Verify storage.set was called with correct key
        mock_storage.set.assert_called_once()
        args = mock_storage.set.call_args
        assert args[0][0] == f"thought:{thought.id}"
        assert args[0][1] == thought

    def test_get_thought(self, thought_storage, mock_storage):
        """Test retrieving a thought."""
        thought = Thought(prompt="Test prompt")
        mock_storage.get.return_value = thought

        result = thought_storage.get_thought(thought.id)

        assert result == thought
        mock_storage.get.assert_called_once_with(f"thought:{thought.id}")

    def test_find_similar_thoughts(self, thought_storage, mock_storage):
        """Test finding similar thoughts."""
        thought1 = Thought(prompt="AI research")
        thought2 = Thought(prompt="Machine learning")
        mock_storage.search_similar.return_value = [thought1, thought2]

        results = thought_storage.find_similar_thoughts_by_text("artificial intelligence")

        assert len(results) == 2
        assert thought1 in results
        assert thought2 in results


class TestCachedCheckpointStorage:
    """Test the cached checkpoint storage."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock CachedStorage for testing."""
        mock = Mock()
        mock.get.return_value = None
        mock.set.return_value = None
        mock.search_similar.return_value = []
        mock.memory = Mock()
        mock.memory.data = {}
        return mock

    @pytest.fixture
    def checkpoint_storage(self, mock_storage):
        """Create a CachedCheckpointStorage with mock backend."""
        return CachedCheckpointStorage(mock_storage)

    def test_save_checkpoint(self, checkpoint_storage, mock_storage):
        """Test saving a checkpoint."""
        thought = Thought(prompt="Test prompt")
        checkpoint = ChainCheckpoint(
            chain_id="test_chain",
            current_step="validation",
            iteration=1,
            thought=thought,
            recovery_point="start_validation",
        )

        checkpoint_storage.save_checkpoint(checkpoint)

        # Verify storage.set was called
        mock_storage.set.assert_called_once()
        args = mock_storage.set.call_args
        assert args[0][0] == f"checkpoint:{checkpoint.checkpoint_id}"
        assert args[0][1] == checkpoint

    def test_get_checkpoint(self, checkpoint_storage, mock_storage):
        """Test retrieving a checkpoint."""
        thought = Thought(prompt="Test prompt")
        checkpoint = ChainCheckpoint(
            chain_id="test_chain",
            current_step="validation",
            iteration=1,
            thought=thought,
            recovery_point="start_validation",
        )
        mock_storage.get.return_value = checkpoint

        result = checkpoint_storage.get_checkpoint(checkpoint.checkpoint_id)

        assert result == checkpoint
        mock_storage.get.assert_called_once_with(f"checkpoint:{checkpoint.checkpoint_id}")


class TestCachedRetriever:
    """Test the cached retriever wrapper."""

    @pytest.fixture
    def base_retriever(self):
        """Create a base retriever for testing."""
        retriever = InMemoryRetriever()
        retriever.add_document("doc1", "Artificial intelligence is fascinating")
        retriever.add_document("doc2", "Machine learning enables automation")
        return retriever

    @pytest.fixture
    def mock_storage(self):
        """Create a mock CachedStorage for testing."""
        mock = Mock()
        mock.get.return_value = None  # Cache miss by default
        mock.set.return_value = None
        return mock

    @pytest.fixture
    def cached_retriever(self, mock_storage, base_retriever):
        """Create a CachedRetriever with mock storage."""
        return CachedRetriever(mock_storage, base_retriever)

    def test_cache_miss_and_set(self, cached_retriever, mock_storage, base_retriever):
        """Test cache miss behavior and result caching."""
        query = "artificial intelligence"

        # First call should be cache miss
        results = cached_retriever.retrieve(query)

        # Should get results from base retriever
        assert len(results) > 0
        assert any("intelligence" in result.lower() for result in results)

        # Should have tried to get from cache
        mock_storage.get.assert_called_once()

        # Should have cached the results
        mock_storage.set.assert_called_once()

    def test_cache_hit(self, cached_retriever, mock_storage):
        """Test cache hit behavior."""
        query = "test query"
        cached_results = ["cached result 1", "cached result 2"]
        mock_storage.get.return_value = cached_results

        results = cached_retriever.retrieve(query)

        assert results == cached_results
        # Should not have called set since it was a cache hit
        mock_storage.set.assert_not_called()

    def test_cache_stats(self, cached_retriever, mock_storage):
        """Test cache statistics tracking."""
        # Mock the get_stats method to return a proper dict
        mock_storage.get_stats.return_value = {
            "memory": {"size": 2, "hits": 1, "misses": 1},
            "cache": {"hits": 1, "misses": 1},
            "persistence": {"connected": True},
        }

        # Simulate cache miss
        mock_storage.get.return_value = None
        cached_retriever.retrieve("query1")

        # Simulate cache hit
        mock_storage.get.return_value = ["result"]
        cached_retriever.retrieve("query2")

        stats = cached_retriever.get_cache_stats()
        assert stats["cache_performance"]["hits"] == 1
        assert stats["cache_performance"]["misses"] == 1
        assert stats["cache_performance"]["hit_rate"] == 0.5


class TestCachedMetricsStorage:
    """Test the cached metrics storage."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock CachedStorage for testing."""
        mock = Mock()
        mock.get.return_value = None
        mock.set.return_value = None
        mock.search_similar.return_value = []
        mock.memory = Mock()
        mock.memory.data = {}
        return mock

    @pytest.fixture
    def metrics_storage(self, mock_storage):
        """Create a CachedMetricsStorage with mock backend."""
        return CachedMetricsStorage(mock_storage)

    def test_record_metric(self, metrics_storage, mock_storage):
        """Test recording a performance metric."""
        metric_id = metrics_storage.record_metric(
            operation="text_generation",
            component="TestModel",
            duration_ms=123.45,
            metadata={"test": "data"},
            tags=["test", "demo"],
        )

        assert isinstance(metric_id, str)

        # Verify storage.set was called
        mock_storage.set.assert_called_once()
        args = mock_storage.set.call_args
        assert args[0][0] == f"metric:{metric_id}"

        # Verify the metric object
        metric = args[0][1]
        assert isinstance(metric, PerformanceMetric)
        assert metric.operation == "text_generation"
        assert metric.component == "TestModel"
        assert metric.duration_ms == 123.45


class TestSifakaStorage:
    """Test the unified storage manager."""

    @pytest.fixture
    def storage_configs(self):
        """Create mock MCP configurations."""
        redis_config = MCPServerConfig(
            name="test-redis", transport_type=MCPTransportType.STDIO, url="mock://redis"
        )
        milvus_config = MCPServerConfig(
            name="test-milvus", transport_type=MCPTransportType.STDIO, url="mock://milvus"
        )
        return redis_config, milvus_config

    def test_storage_manager_creation(self, storage_configs):
        """Test creating the storage manager."""
        redis_config, milvus_config = storage_configs

        storage = SifakaStorage(
            redis_config=redis_config, milvus_config=milvus_config, memory_size=100, cache_ttl=300
        )

        assert storage.memory_size == 100
        assert storage.cache_ttl == 300
        assert storage.redis_config == redis_config
        assert storage.milvus_config == milvus_config

    def test_get_component_storage(self, storage_configs):
        """Test getting component-specific storage instances."""
        redis_config, milvus_config = storage_configs
        storage = SifakaStorage(redis_config, milvus_config)

        # Test lazy initialization
        thought_storage = storage.get_thought_storage()
        assert isinstance(thought_storage, CachedThoughtStorage)

        checkpoint_storage = storage.get_checkpoint_storage()
        assert isinstance(checkpoint_storage, CachedCheckpointStorage)

        metrics_storage = storage.get_metrics_storage()
        assert isinstance(metrics_storage, CachedMetricsStorage)

        # Test retriever cache
        base_retriever = InMemoryRetriever()
        cached_retriever = storage.get_retriever_cache(base_retriever)
        assert isinstance(cached_retriever, CachedRetriever)
        assert cached_retriever.base == base_retriever


if __name__ == "__main__":
    pytest.main([__file__])
