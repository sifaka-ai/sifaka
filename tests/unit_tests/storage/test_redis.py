"""Comprehensive unit tests for Redis storage backend.

This module tests the RedisPersistence implementation:
- Redis-based persistence via MCP server
- JSON storage and retrieval operations
- TTL (Time-To-Live) functionality
- Error handling and edge cases

Tests cover:
- Storage and retrieval operations
- Redis-specific features (TTL, JSON operations)
- MCP server integration
- Error scenarios and recovery
- Mock-based testing without actual Redis connection
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.storage.redis import RedisPersistence


class _TestableRedisPersistence(RedisPersistence):
    """Testable version of RedisPersistence with mocked abstract methods."""

    async def load_all(self, *args, **kwargs):
        """Mock implementation."""
        return []

    async def load_next(self, *args, **kwargs):
        """Mock implementation."""
        return None

    async def record_run(self, *args, **kwargs):
        """Mock implementation."""

    async def snapshot_end(self, *args, **kwargs):
        """Mock implementation."""

    async def snapshot_node_if_new(self, *args, **kwargs):
        """Mock implementation."""


class TestRedisPersistence:
    """Test suite for RedisPersistence class."""

    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCP server."""
        server = Mock()
        server.__aenter__ = AsyncMock(return_value=server)
        server.__aexit__ = AsyncMock(return_value=None)
        server.call_tool = AsyncMock()
        return server

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Test Redis storage",
            final_text="This is a test thought for Redis storage.",
            iteration=1,
            max_iterations=3,
        )
        thought.add_generation("Generated text", "gpt-4", {"temperature": 0.7})
        thought.add_validation("length_validator", True, {"word_count": 10})
        return thought

    def test_redis_persistence_creation_minimal(self, mock_mcp_server):
        """Test creating RedisPersistence with minimal parameters."""
        persistence = _TestableRedisPersistence(mcp_server=mock_mcp_server)

        assert persistence.mcp_server == mock_mcp_server
        assert persistence.ttl_seconds is None  # Default no TTL
        assert persistence.key_prefix == "sifaka"  # Default prefix

    def test_redis_persistence_creation_with_ttl(self, mock_mcp_server):
        """Test creating RedisPersistence with TTL."""
        persistence = _TestableRedisPersistence(
            mcp_server=mock_mcp_server, ttl_seconds=3600, key_prefix="test"
        )

        assert persistence.ttl_seconds == 3600
        assert persistence.key_prefix == "test"

    @pytest.mark.asyncio
    async def test_store_raw_basic(self, mock_mcp_server):
        """Test basic raw data storage."""
        persistence = _TestableRedisPersistence(mcp_server=mock_mcp_server)

        test_data = '{"test": "data", "number": 42}'
        test_key = "test:key"

        # Mock successful storage
        mock_mcp_server.call_tool.return_value = {"success": True}

        await persistence._store_raw(test_key, test_data)

        # Verify json_set was called
        mock_mcp_server.call_tool.assert_called_with(
            "json_set",
            arguments={"name": test_key, "path": "$", "value": {"test": "data", "number": 42}},
        )

    @pytest.mark.asyncio
    async def test_store_raw_with_ttl(self, mock_mcp_server):
        """Test raw data storage with TTL."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server, ttl_seconds=1800)

        test_data = '{"test": "data"}'
        test_key = "test:key"

        # Mock successful storage and TTL setting
        mock_mcp_server.call_tool.return_value = {"success": True}

        await persistence._store_raw(test_key, test_data)

        # Verify both json_set and expire were called
        calls = mock_mcp_server.call_tool.call_args_list
        assert len(calls) == 2

        # First call should be json_set
        assert calls[0][0][0] == "json_set"

        # Second call should be expire
        assert calls[1][0][0] == "expire"
        assert calls[1][1]["arguments"]["name"] == test_key
        assert calls[1][1]["arguments"]["seconds"] == 1800

    @pytest.mark.asyncio
    async def test_retrieve_raw_basic(self, mock_mcp_server):
        """Test basic raw data retrieval."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        test_key = "test:key"
        expected_data = {"test": "data", "number": 42}

        # Mock successful retrieval
        mock_mcp_server.call_tool.return_value = {"success": True, "result": expected_data}

        result = await persistence._retrieve_raw(test_key)

        # Verify json_get was called
        mock_mcp_server.call_tool.assert_called_with(
            "json_get", arguments={"name": test_key, "path": "$"}
        )

        # Verify result is JSON string
        assert result == json.dumps(expected_data)

    @pytest.mark.asyncio
    async def test_retrieve_raw_not_found(self, mock_mcp_server):
        """Test raw data retrieval when key doesn't exist."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        test_key = "nonexistent:key"

        # Mock key not found
        mock_mcp_server.call_tool.return_value = {"success": False, "error": "Key not found"}

        result = await persistence._retrieve_raw(test_key)

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_raw_basic(self, mock_mcp_server):
        """Test basic raw data deletion."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        test_key = "test:key"

        # Mock successful deletion
        mock_mcp_server.call_tool.return_value = {"success": True, "deleted": 1}

        result = await persistence._delete_raw(test_key)

        # Verify del was called
        mock_mcp_server.call_tool.assert_called_with("del", arguments={"name": test_key})

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_raw_not_found(self, mock_mcp_server):
        """Test raw data deletion when key doesn't exist."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        test_key = "nonexistent:key"

        # Mock key not found
        mock_mcp_server.call_tool.return_value = {"success": True, "deleted": 0}

        result = await persistence._delete_raw(test_key)

        assert result is False

    @pytest.mark.asyncio
    async def test_list_keys_basic(self, mock_mcp_server):
        """Test basic key listing."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        pattern = "test:*"
        expected_keys = ["test:key1", "test:key2", "test:key3"]

        # Mock successful key listing
        mock_mcp_server.call_tool.return_value = {"success": True, "keys": expected_keys}

        result = await persistence._list_keys(pattern)

        # Verify keys was called
        mock_mcp_server.call_tool.assert_called_with("keys", arguments={"pattern": pattern})

        assert result == expected_keys

    @pytest.mark.asyncio
    async def test_store_thought_complete(self, mock_mcp_server, sample_thought):
        """Test complete thought storage."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        # Mock successful storage
        mock_mcp_server.call_tool.return_value = {"success": True}

        await persistence.store_thought(sample_thought)

        # Verify storage was called
        mock_mcp_server.call_tool.assert_called()

        # Check that the key follows the expected pattern
        call_args = mock_mcp_server.call_tool.call_args
        assert "json_set" in call_args[0]
        assert sample_thought.id in call_args[1]["arguments"]["name"]

    @pytest.mark.asyncio
    async def test_retrieve_thought_complete(self, mock_mcp_server, sample_thought):
        """Test complete thought retrieval."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        # Mock successful retrieval with thought data
        thought_data = sample_thought.model_dump()
        mock_mcp_server.call_tool.return_value = {"success": True, "result": thought_data}

        result = await persistence.retrieve_thought(sample_thought.id)

        # Verify retrieval was called
        mock_mcp_server.call_tool.assert_called()

        # Verify result is a SifakaThought
        assert isinstance(result, SifakaThought)
        assert result.id == sample_thought.id
        assert result.prompt == sample_thought.prompt

    @pytest.mark.asyncio
    async def test_store_raw_error_handling(self, mock_mcp_server):
        """Test error handling in raw storage."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        test_data = '{"test": "data"}'
        test_key = "test:key"

        # Mock storage error
        mock_mcp_server.call_tool.side_effect = Exception("Redis connection error")

        with pytest.raises(Exception, match="Redis connection error"):
            await persistence._store_raw(test_key, test_data)

    @pytest.mark.asyncio
    async def test_retrieve_raw_error_handling(self, mock_mcp_server):
        """Test error handling in raw retrieval."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        test_key = "test:key"

        # Mock retrieval error
        mock_mcp_server.call_tool.side_effect = Exception("Redis connection error")

        result = await persistence._retrieve_raw(test_key)

        # Should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, mock_mcp_server):
        """Test handling of invalid JSON data."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        invalid_json = "not valid json"
        test_key = "test:key"

        with pytest.raises(json.JSONDecodeError):
            await persistence._store_raw(test_key, invalid_json)

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_mcp_server):
        """Test that MCP server is used as context manager."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        test_data = '{"test": "data"}'
        test_key = "test:key"

        mock_mcp_server.call_tool.return_value = {"success": True}

        await persistence._store_raw(test_key, test_data)

        # Verify context manager methods were called
        mock_mcp_server.__aenter__.assert_called_once()
        mock_mcp_server.__aexit__.assert_called_once()

    def test_key_generation(self, mock_mcp_server):
        """Test Redis key generation."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server, key_prefix="redis_test")

        # Test thought key generation
        thought_key = persistence._make_key("thought:123")
        assert thought_key == "redis_test:thought:123"

        # Test custom key generation
        custom_key = persistence._make_key("custom:data")
        assert custom_key == "redis_test:custom:data"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_mcp_server):
        """Test concurrent Redis operations."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        # Mock successful operations
        mock_mcp_server.call_tool.return_value = {"success": True}

        # Create multiple concurrent operations
        tasks = []
        for i in range(5):
            task = persistence._store_raw(f"test:key{i}", f'{{"data": {i}}}')
            tasks.append(task)

        # Execute concurrently
        await asyncio.gather(*tasks)

        # Verify all operations were called
        assert mock_mcp_server.call_tool.call_count == 5

    @pytest.mark.asyncio
    async def test_large_data_storage(self, mock_mcp_server):
        """Test storage of large data objects."""
        persistence = RedisPersistence(mcp_server=mock_mcp_server)

        # Create large data object
        large_data = {"large_list": list(range(1000)), "text": "x" * 10000}
        test_key = "test:large"

        mock_mcp_server.call_tool.return_value = {"success": True}

        await persistence._store_raw(test_key, json.dumps(large_data))

        # Verify storage was attempted
        mock_mcp_server.call_tool.assert_called_once()

        # Verify large data was passed correctly
        call_args = mock_mcp_server.call_tool.call_args
        assert call_args[1]["arguments"]["value"] == large_data
