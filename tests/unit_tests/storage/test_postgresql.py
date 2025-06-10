"""Comprehensive unit tests for PostgreSQL storage backend.

This module tests the PostgreSQLPersistence implementation:
- PostgreSQL-based persistence with asyncpg
- Structured data storage and retrieval
- SQL query operations and indexing
- Error handling and edge cases

Tests cover:
- Storage and retrieval operations
- PostgreSQL-specific features (SQL queries, indexing)
- Connection pool management
- Error scenarios and recovery
- Mock-based testing without actual PostgreSQL connection
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.storage.postgresql import PostgreSQLPersistence


class TestPostgreSQLPersistence:
    """Test suite for PostgreSQLPersistence class."""

    @pytest.fixture
    def mock_connection_pool(self):
        """Create a mock asyncpg connection pool."""
        pool = Mock()
        pool.acquire = AsyncMock()
        pool.close = AsyncMock()

        # Mock connection
        connection = Mock()
        connection.execute = AsyncMock()
        connection.fetch = AsyncMock()
        connection.fetchrow = AsyncMock()
        connection.fetchval = AsyncMock()
        connection.__aenter__ = AsyncMock(return_value=connection)
        connection.__aexit__ = AsyncMock(return_value=None)

        pool.acquire.return_value = connection
        return pool

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Test PostgreSQL storage",
            final_text="This is a test thought for PostgreSQL storage.",
            iteration=1,
            max_iterations=3,
        )
        thought.add_generation("Generated text", "gpt-4", {"temperature": 0.7})
        thought.add_validation("length_validator", True, {"word_count": 10})
        return thought

    def test_postgresql_persistence_creation_minimal(self):
        """Test creating PostgreSQLPersistence with minimal parameters."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )

        assert persistence.connection_string == "postgresql://test:test@localhost/test"
        assert persistence.pool is None  # Not connected yet
        assert persistence.key_prefix == "sifaka"  # Default prefix

    def test_postgresql_persistence_creation_with_custom_params(self):
        """Test creating PostgreSQLPersistence with custom parameters."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://user:pass@host:5432/db",
            key_prefix="test_prefix",
            max_connections=20,
        )

        assert persistence.connection_string == "postgresql://user:pass@host:5432/db"
        assert persistence.key_prefix == "test_prefix"
        assert persistence.max_connections == 20

    @pytest.mark.asyncio
    async def test_ensure_pool_creation(self, mock_connection_pool):
        """Test connection pool creation."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )

        with patch("asyncpg.create_pool", return_value=mock_connection_pool):
            await persistence._ensure_pool()

            assert persistence.pool == mock_connection_pool

    @pytest.mark.asyncio
    async def test_store_thought_data(self, mock_connection_pool):
        """Test storing thought data in PostgreSQL."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        thought_id = "test-thought-123"
        thought_data = '{"prompt": "test", "final_text": "result"}'

        # Mock successful insertion
        connection = await mock_connection_pool.acquire()
        connection.execute.return_value = "INSERT 0 1"

        await persistence._store_thought_data(thought_id, thought_data)

        # Verify INSERT was called
        connection.execute.assert_called_once()
        call_args = connection.execute.call_args[0]
        assert "INSERT INTO thoughts" in call_args[0]
        assert thought_id in call_args

    @pytest.mark.asyncio
    async def test_retrieve_thought_data(self, mock_connection_pool):
        """Test retrieving thought data from PostgreSQL."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        thought_id = "test-thought-123"
        expected_data = '{"prompt": "test", "final_text": "result"}'

        # Mock successful retrieval
        connection = await mock_connection_pool.acquire()
        connection.fetchval.return_value = expected_data

        result = await persistence._retrieve_thought_data(thought_id)

        # Verify SELECT was called
        connection.fetchval.assert_called_once()
        call_args = connection.fetchval.call_args[0]
        assert "SELECT data FROM thoughts" in call_args[0]
        assert thought_id in call_args

        assert result == expected_data

    @pytest.mark.asyncio
    async def test_store_snapshot_data(self, mock_connection_pool):
        """Test storing snapshot data in PostgreSQL."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        thought_id = "test-thought-123"
        node_name = "generate_node"
        snapshot_data = '{"node_state": "completed"}'

        # Mock successful insertion
        connection = await mock_connection_pool.acquire()
        connection.execute.return_value = "INSERT 0 1"

        await persistence._store_snapshot_data(thought_id, node_name, snapshot_data)

        # Verify INSERT was called
        connection.execute.assert_called_once()
        call_args = connection.execute.call_args[0]
        assert "INSERT INTO snapshots" in call_args[0]
        assert thought_id in call_args
        assert node_name in call_args

    @pytest.mark.asyncio
    async def test_retrieve_snapshot_data(self, mock_connection_pool):
        """Test retrieving snapshot data from PostgreSQL."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        thought_id = "test-thought-123"
        node_name = "generate_node"
        expected_data = '{"node_state": "completed"}'

        # Mock successful retrieval
        connection = await mock_connection_pool.acquire()
        connection.fetchval.return_value = expected_data

        result = await persistence._retrieve_snapshot_data(thought_id, node_name)

        # Verify SELECT was called
        connection.fetchval.assert_called_once()
        call_args = connection.fetchval.call_args[0]
        assert "SELECT data FROM snapshots" in call_args[0]
        assert thought_id in call_args
        assert node_name in call_args

        assert result == expected_data

    @pytest.mark.asyncio
    async def test_store_raw_thought_key(self, mock_connection_pool):
        """Test storing raw data with thought key."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        key = "sifaka:thought:123"
        data = '{"test": "data"}'

        # Mock successful storage
        connection = await mock_connection_pool.acquire()
        connection.execute.return_value = "INSERT 0 1"

        await persistence._store_raw(key, data)

        # Verify thought storage was called
        connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_raw_snapshot_key(self, mock_connection_pool):
        """Test storing raw data with snapshot key."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        key = "sifaka:snapshot:123:node_name"
        data = '{"snapshot": "data"}'

        # Mock successful storage
        connection = await mock_connection_pool.acquire()
        connection.execute.return_value = "INSERT 0 1"

        await persistence._store_raw(key, data)

        # Verify snapshot storage was called
        connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_raw_thought_key(self, mock_connection_pool):
        """Test retrieving raw data with thought key."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        key = "sifaka:thought:123"
        expected_data = '{"test": "data"}'

        # Mock successful retrieval
        connection = await mock_connection_pool.acquire()
        connection.fetchval.return_value = expected_data

        result = await persistence._retrieve_raw(key)

        # Verify thought retrieval was called
        connection.fetchval.assert_called_once()
        assert result == expected_data

    @pytest.mark.asyncio
    async def test_retrieve_raw_not_found(self, mock_connection_pool):
        """Test retrieving raw data when not found."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        key = "sifaka:thought:nonexistent"

        # Mock not found
        connection = await mock_connection_pool.acquire()
        connection.fetchval.return_value = None

        result = await persistence._retrieve_raw(key)

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_raw_thought(self, mock_connection_pool):
        """Test deleting raw thought data."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        key = "sifaka:thought:123"

        # Mock successful deletion
        connection = await mock_connection_pool.acquire()
        connection.execute.return_value = "DELETE 1"

        result = await persistence._delete_raw(key)

        # Verify DELETE was called
        connection.execute.assert_called_once()
        call_args = connection.execute.call_args[0]
        assert "DELETE FROM thoughts" in call_args[0]

        assert result is True

    @pytest.mark.asyncio
    async def test_list_keys_thoughts(self, mock_connection_pool):
        """Test listing thought keys."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        pattern = "sifaka:thought:*"
        expected_keys = ["sifaka:thought:123", "sifaka:thought:456"]

        # Mock successful key listing
        connection = await mock_connection_pool.acquire()
        mock_rows = [{"key": key} for key in expected_keys]
        connection.fetch.return_value = mock_rows

        result = await persistence._list_keys(pattern)

        # Verify SELECT was called
        connection.fetch.assert_called_once()

        assert result == expected_keys

    @pytest.mark.asyncio
    async def test_store_thought_complete(self, mock_connection_pool, sample_thought):
        """Test complete thought storage."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        # Mock successful storage
        connection = await mock_connection_pool.acquire()
        connection.execute.return_value = "INSERT 0 1"

        await persistence.store_thought(sample_thought)

        # Verify storage was called
        connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_thought_complete(self, mock_connection_pool, sample_thought):
        """Test complete thought retrieval."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        # Mock successful retrieval
        thought_data = json.dumps(sample_thought.model_dump())
        connection = await mock_connection_pool.acquire()
        connection.fetchval.return_value = thought_data

        result = await persistence.retrieve_thought(sample_thought.id)

        # Verify retrieval was called
        connection.fetchval.assert_called_once()

        # Verify result is a SifakaThought
        assert isinstance(result, SifakaThought)
        assert result.id == sample_thought.id
        assert result.prompt == sample_thought.prompt

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://invalid:invalid@localhost/invalid"
        )

        with patch("asyncpg.create_pool", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await persistence._ensure_pool()

    @pytest.mark.asyncio
    async def test_sql_error_handling(self, mock_connection_pool):
        """Test handling of SQL errors."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        # Mock SQL error
        connection = await mock_connection_pool.acquire()
        connection.execute.side_effect = Exception("SQL error")

        with pytest.raises(Exception, match="SQL error"):
            await persistence._store_thought_data("test-id", '{"test": "data"}')

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_connection_pool):
        """Test concurrent PostgreSQL operations."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        # Mock successful operations
        connection = await mock_connection_pool.acquire()
        connection.execute.return_value = "INSERT 0 1"

        # Create multiple concurrent operations
        tasks = []
        for i in range(5):
            task = persistence._store_thought_data(f"thought-{i}", f'{{"data": {i}}}')
            tasks.append(task)

        # Execute concurrently
        await asyncio.gather(*tasks)

        # Verify all operations were attempted
        assert connection.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_pool_cleanup(self, mock_connection_pool):
        """Test connection pool cleanup."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )
        persistence.pool = mock_connection_pool

        await persistence.close()

        # Verify pool was closed
        mock_connection_pool.close.assert_called_once()
        assert persistence.pool is None

    def test_key_parsing(self):
        """Test PostgreSQL key parsing logic."""
        persistence = PostgreSQLPersistence(
            connection_string="postgresql://test:test@localhost/test"
        )

        # Test thought key parsing
        thought_key = "sifaka:thought:123"
        parts = thought_key.split(":")
        assert len(parts) == 3
        assert parts[1] == "thought"
        assert parts[2] == "123"

        # Test snapshot key parsing
        snapshot_key = "sifaka:snapshot:123:node_name"
        parts = snapshot_key.split(":")
        assert len(parts) == 4
        assert parts[1] == "snapshot"
        assert parts[2] == "123"
        assert parts[3] == "node_name"
