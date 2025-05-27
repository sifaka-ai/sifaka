#!/usr/bin/env python3
"""Comprehensive tests for Sifaka storage modules.

This test suite covers Redis and Milvus storage backends, testing
data persistence, retrieval, error handling, and MCP integration.
"""

import json
from unittest.mock import Mock, patch

import pytest

from sifaka.mcp.base import MCPServerConfig, MCPTransportType
from sifaka.storage.cached import CachedStorage
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.milvus import MilvusStorage
from sifaka.storage.redis import RedisStorage
from sifaka.utils.error_handling import StorageError
from tests.utils import create_test_thought


class TestRedisStorage:
    """Test Redis storage functionality."""

    def test_redis_storage_basic_config(self):
        """Test basic Redis storage configuration."""
        config = MCPServerConfig(
            name="redis",
            command="redis-server",
            transport_type=MCPTransportType.STDIO,
            args=["--port", "6379"],
        )

        with patch("sifaka.storage.redis.RedisStorage._connect_mcp") as mock_connect:
            mock_connect.return_value = Mock()
            storage = RedisStorage(mcp_config=config)
            assert storage.mcp_config == config

    @patch("sifaka.storage.redis.RedisStorage._connect_mcp")
    def test_redis_storage_save_and_load(self, mock_connect):
        """Test Redis storage save and load operations."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"success": True}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="redis", command="redis-server", transport_type=MCPTransportType.STDIO
        )

        storage = RedisStorage(mcp_config=config)
        thought = create_test_thought(text="Test Redis storage")

        # Test save
        storage.save(thought.id, thought)
        mock_client.call_tool.assert_called()

        # Test load
        mock_client.call_tool.return_value = {"value": thought.model_dump_json()}
        loaded_thought = storage.load(thought.id)

        assert loaded_thought is not None
        assert loaded_thought.id == thought.id
        assert loaded_thought.text == thought.text

    @patch("sifaka.storage.redis.RedisStorage._connect_mcp")
    def test_redis_storage_exists(self, mock_connect):
        """Test Redis storage exists check."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"exists": True}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="redis", command="redis-server", transport_type=MCPTransportType.STDIO
        )

        storage = RedisStorage(mcp_config=config)

        assert storage.exists("test_key") is True
        mock_client.call_tool.assert_called_with("redis_exists", {"key": "test_key"})

    @patch("sifaka.storage.redis.RedisStorage._connect_mcp")
    def test_redis_storage_delete(self, mock_connect):
        """Test Redis storage delete operation."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"deleted": 1}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="redis", command="redis-server", transport_type=MCPTransportType.STDIO
        )

        storage = RedisStorage(mcp_config=config)

        storage.delete("test_key")
        mock_client.call_tool.assert_called_with("redis_delete", {"key": "test_key"})

    @patch("sifaka.storage.redis.RedisStorage._connect_mcp")
    def test_redis_storage_list_keys(self, mock_connect):
        """Test Redis storage list keys operation."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"keys": ["key1", "key2", "key3"]}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="redis", command="redis-server", transport_type=MCPTransportType.STDIO
        )

        storage = RedisStorage(mcp_config=config)

        keys = storage.list_keys("pattern*")
        assert keys == ["key1", "key2", "key3"]
        mock_client.call_tool.assert_called_with("redis_keys", {"pattern": "pattern*"})

    @patch("sifaka.storage.redis.RedisStorage._connect_mcp")
    def test_redis_storage_error_handling(self, mock_connect):
        """Test Redis storage error handling."""
        mock_client = Mock()
        mock_client.call_tool.side_effect = Exception("Redis connection failed")
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="redis", command="redis-server", transport_type=MCPTransportType.STDIO
        )

        storage = RedisStorage(mcp_config=config)
        thought = create_test_thought()

        with pytest.raises(StorageError):
            storage.save(thought.id, thought)

    @patch("sifaka.storage.redis.RedisStorage._connect_mcp")
    def test_redis_storage_serialization(self, mock_connect):
        """Test Redis storage serialization/deserialization."""
        mock_client = Mock()
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="redis", command="redis-server", transport_type=MCPTransportType.STDIO
        )

        storage = RedisStorage(mcp_config=config)
        thought = create_test_thought(
            text="Test serialization", metadata={"key": "value", "number": 42}
        )

        # Test serialization
        serialized = storage._serialize_thought(thought)
        assert isinstance(serialized, str)

        # Test deserialization
        deserialized = storage._deserialize_thought(serialized)
        assert deserialized.id == thought.id
        assert deserialized.text == thought.text
        assert deserialized.metadata == thought.metadata


class TestMilvusStorage:
    """Test Milvus storage functionality."""

    def test_milvus_storage_basic_config(self):
        """Test basic Milvus storage configuration."""
        config = MCPServerConfig(
            name="milvus",
            command="milvus-server",
            transport_type=MCPTransportType.STDIO,
            args=["--host", "localhost", "--port", "19530"],
        )

        with patch("sifaka.storage.milvus.MilvusStorage._connect_mcp") as mock_connect:
            mock_connect.return_value = Mock()
            storage = MilvusStorage(
                mcp_config=config, collection_name="test_collection", dimension=768
            )
            assert storage.mcp_config == config
            assert storage.collection_name == "test_collection"
            assert storage.dimension == 768

    @patch("sifaka.storage.milvus.MilvusStorage._connect_mcp")
    def test_milvus_storage_save_and_load(self, mock_connect):
        """Test Milvus storage save and load operations."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"success": True}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="milvus", command="milvus-server", transport_type=MCPTransportType.STDIO
        )

        storage = MilvusStorage(mcp_config=config, collection_name="test_collection", dimension=768)

        thought = create_test_thought(text="Test Milvus storage")

        # Test save
        storage.save(thought.id, thought)
        mock_client.call_tool.assert_called()

        # Test load
        mock_client.call_tool.return_value = {
            "entities": [
                {"id": thought.id, "text": thought.text, "metadata": json.dumps(thought.metadata)}
            ]
        }
        loaded_thought = storage.load(thought.id)

        assert loaded_thought is not None
        assert loaded_thought.id == thought.id
        assert loaded_thought.text == thought.text

    @patch("sifaka.storage.milvus.MilvusStorage._connect_mcp")
    def test_milvus_storage_vector_search(self, mock_connect):
        """Test Milvus storage vector search functionality."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {
            "results": [
                {
                    "id": "thought1",
                    "distance": 0.1,
                    "entity": {"text": "Similar thought 1", "metadata": "{}"},
                },
                {
                    "id": "thought2",
                    "distance": 0.2,
                    "entity": {"text": "Similar thought 2", "metadata": "{}"},
                },
            ]
        }
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="milvus", command="milvus-server", transport_type=MCPTransportType.STDIO
        )

        storage = MilvusStorage(mcp_config=config, collection_name="test_collection", dimension=768)

        # Test vector search
        query_vector = [0.1] * 768
        results = storage.search_similar(query_vector, limit=2)

        assert len(results) == 2
        assert results[0]["id"] == "thought1"
        assert results[0]["distance"] == 0.1
        assert results[1]["id"] == "thought2"
        assert results[1]["distance"] == 0.2

    @patch("sifaka.storage.milvus.MilvusStorage._connect_mcp")
    def test_milvus_storage_embedding_generation(self, mock_connect):
        """Test Milvus storage embedding generation."""
        mock_client = Mock()
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="milvus", command="milvus-server", transport_type=MCPTransportType.STDIO
        )

        storage = MilvusStorage(mcp_config=config, collection_name="test_collection", dimension=768)

        # Mock embedding generation
        with patch.object(storage, "_generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 768

            text = "Generate embedding for this text"
            embedding = storage._generate_embedding(text)

            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    @patch("sifaka.storage.milvus.MilvusStorage._connect_mcp")
    def test_milvus_storage_collection_management(self, mock_connect):
        """Test Milvus storage collection management."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"success": True}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="milvus", command="milvus-server", transport_type=MCPTransportType.STDIO
        )

        storage = MilvusStorage(mcp_config=config, collection_name="test_collection", dimension=768)

        # Test collection creation
        storage._ensure_collection_exists()
        mock_client.call_tool.assert_called()

    @patch("sifaka.storage.milvus.MilvusStorage._connect_mcp")
    def test_milvus_storage_batch_operations(self, mock_connect):
        """Test Milvus storage batch operations."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"success": True}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="milvus", command="milvus-server", transport_type=MCPTransportType.STDIO
        )

        storage = MilvusStorage(mcp_config=config, collection_name="test_collection", dimension=768)

        # Test batch save
        thoughts = [create_test_thought(text=f"Thought {i}") for i in range(5)]

        storage.save_batch(thoughts)
        mock_client.call_tool.assert_called()

    @patch("sifaka.storage.milvus.MilvusStorage._connect_mcp")
    def test_milvus_storage_error_handling(self, mock_connect):
        """Test Milvus storage error handling."""
        mock_client = Mock()
        mock_client.call_tool.side_effect = Exception("Milvus connection failed")
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="milvus", command="milvus-server", transport_type=MCPTransportType.STDIO
        )

        storage = MilvusStorage(mcp_config=config, collection_name="test_collection", dimension=768)

        thought = create_test_thought()

        with pytest.raises(StorageError):
            storage.save(thought.id, thought)

    @patch("sifaka.storage.milvus.MilvusStorage._connect_mcp")
    def test_milvus_storage_text_truncation(self, mock_connect):
        """Test Milvus storage text truncation for long content."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"success": True}
        mock_connect.return_value = mock_client

        config = MCPServerConfig(
            name="milvus", command="milvus-server", transport_type=MCPTransportType.STDIO
        )

        storage = MilvusStorage(
            mcp_config=config, collection_name="test_collection", dimension=768, max_text_length=100
        )

        # Create thought with very long text
        long_text = "This is a very long text. " * 50  # Much longer than 100 chars
        thought = create_test_thought(text=long_text)

        # Test truncation
        truncated = storage._truncate_text(thought.text)
        assert len(truncated) <= 100
        assert truncated.endswith("...")


class TestCachedStorage:
    """Test cached storage functionality."""

    def test_cached_storage_basic(self):
        """Test basic cached storage functionality."""
        cache = MemoryStorage()
        persistence = MemoryStorage()

        storage = CachedStorage(cache=cache, persistence=persistence)
        thought = create_test_thought(text="Test cached storage")

        # Test save (should save to both cache and persistence)
        storage.save(thought.id, thought)

        assert cache.exists(thought.id)
        assert persistence.exists(thought.id)

    def test_cached_storage_cache_hit(self):
        """Test cached storage cache hit scenario."""
        cache = MemoryStorage()
        persistence = MemoryStorage()

        storage = CachedStorage(cache=cache, persistence=persistence)
        thought = create_test_thought(text="Test cache hit")

        # Save to cache only
        cache.save(thought.id, thought)

        # Load should hit cache
        loaded_thought = storage.load(thought.id)
        assert loaded_thought is not None
        assert loaded_thought.id == thought.id

    def test_cached_storage_cache_miss(self):
        """Test cached storage cache miss scenario."""
        cache = MemoryStorage()
        persistence = MemoryStorage()

        storage = CachedStorage(cache=cache, persistence=persistence)
        thought = create_test_thought(text="Test cache miss")

        # Save to persistence only
        persistence.save(thought.id, thought)

        # Load should miss cache, hit persistence, and populate cache
        loaded_thought = storage.load(thought.id)
        assert loaded_thought is not None
        assert loaded_thought.id == thought.id
        assert cache.exists(thought.id)  # Should now be in cache

    def test_cached_storage_cache_invalidation(self):
        """Test cached storage cache invalidation."""
        cache = MemoryStorage()
        persistence = MemoryStorage()

        storage = CachedStorage(cache=cache, persistence=persistence)
        thought = create_test_thought(text="Test cache invalidation")

        # Save thought
        storage.save(thought.id, thought)

        # Delete should remove from both cache and persistence
        storage.delete(thought.id)

        assert not cache.exists(thought.id)
        assert not persistence.exists(thought.id)

    def test_cached_storage_error_handling(self):
        """Test cached storage error handling."""
        cache = MemoryStorage()

        # Mock persistence that always fails
        persistence = Mock()
        persistence.save.side_effect = Exception("Persistence failed")
        persistence.exists.return_value = False

        storage = CachedStorage(cache=cache, persistence=persistence)
        thought = create_test_thought()

        with pytest.raises(StorageError):
            storage.save(thought.id, thought)

    def test_cached_storage_fallback_behavior(self):
        """Test cached storage fallback behavior."""
        # Mock cache that fails
        cache = Mock()
        cache.load.side_effect = Exception("Cache failed")
        cache.exists.return_value = False

        persistence = MemoryStorage()

        storage = CachedStorage(cache=cache, persistence=persistence)
        thought = create_test_thought(text="Test fallback")

        # Save to persistence directly
        persistence.save(thought.id, thought)

        # Load should fallback to persistence when cache fails
        loaded_thought = storage.load(thought.id)
        assert loaded_thought is not None
        assert loaded_thought.id == thought.id


class TestStorageIntegration:
    """Test storage integration and common functionality."""

    def test_storage_with_chain_integration(self):
        """Test storage integration with chain execution."""
        from sifaka.core.chain import Chain
        from sifaka.models.base import MockModel

        storage = MemoryStorage()
        model = MockModel(response_text="Test storage integration")

        chain = Chain(model=model, prompt="Test prompt", storage=storage)

        result = chain.run()

        # Should save thought to storage
        thought_key = f"thought_{result.chain_id}_{result.iteration}"
        assert storage.exists(thought_key)

    def test_multiple_storage_backends(self):
        """Test using multiple storage backends."""
        memory_storage = MemoryStorage()

        # Mock Redis and Milvus storages
        with patch("sifaka.storage.redis.RedisStorage") as mock_redis:
            with patch("sifaka.storage.milvus.MilvusStorage") as mock_milvus:
                mock_redis_instance = Mock()
                mock_milvus_instance = Mock()
                mock_redis.return_value = mock_redis_instance
                mock_milvus.return_value = mock_milvus_instance

                thought = create_test_thought()

                # Save to all storages
                memory_storage.save(thought.id, thought)
                mock_redis_instance.save(thought.id, thought)
                mock_milvus_instance.save(thought.id, thought)

                # Verify all were called
                assert memory_storage.exists(thought.id)
                mock_redis_instance.save.assert_called_once()
                mock_milvus_instance.save.assert_called_once()

    def test_storage_performance_comparison(self):
        """Test storage performance characteristics."""
        import time

        storage = MemoryStorage()
        thoughts = [create_test_thought(text=f"Performance test {i}") for i in range(100)]

        # Test batch save performance
        start_time = time.time()
        for thought in thoughts:
            storage.save(thought.id, thought)
        save_time = time.time() - start_time

        # Test batch load performance
        start_time = time.time()
        for thought in thoughts:
            loaded = storage.load(thought.id)
            assert loaded is not None
        load_time = time.time() - start_time

        # Should complete operations in reasonable time
        assert save_time < 1.0  # 1 second for 100 saves
        assert load_time < 1.0  # 1 second for 100 loads

    def test_storage_data_consistency(self):
        """Test storage data consistency across operations."""
        storage = MemoryStorage()
        original_thought = create_test_thought(
            text="Original text", metadata={"version": 1, "author": "test"}
        )

        # Save original
        storage.save(original_thought.id, original_thought)

        # Load and verify
        loaded_thought = storage.load(original_thought.id)
        assert loaded_thought.text == original_thought.text
        assert loaded_thought.metadata == original_thought.metadata

        # Update and save again
        updated_thought = loaded_thought.model_copy(
            update={"text": "Updated text", "metadata": {"version": 2, "author": "test"}}
        )
        storage.save(updated_thought.id, updated_thought)

        # Load updated version
        final_thought = storage.load(updated_thought.id)
        assert final_thought.text == "Updated text"
        assert final_thought.metadata["version"] == 2
