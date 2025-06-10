"""Comprehensive unit tests for storage retrieval tools.

This module tests the storage retrieval tools implementation:
- RedisRetrievalTool for Redis-specific operations
- PostgreSQLRetrievalTool for SQL-based queries
- GenericStorageRetrievalTool for any backend
- Tool factory functions and registration

Tests cover:
- Tool creation and configuration
- Backend-specific retrieval operations
- PydanticAI tool integration
- Error handling and validation
- Mock-based testing without actual storage connections
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.tools.base import ToolConfigurationError
from sifaka.tools.retrieval.storage_retrieval import (
    GenericStorageRetrievalTool,
    PostgreSQLRetrievalTool,
    RedisRetrievalTool,
    create_storage_retrieval_tools,
)


class TestRedisRetrievalTool:
    """Test suite for RedisRetrievalTool class."""

    @pytest.fixture
    def mock_redis_persistence(self):
        """Create a mock Redis persistence backend."""
        persistence = Mock()
        persistence.__class__.__name__ = "RedisPersistence"
        persistence._list_keys = AsyncMock()
        persistence._get_raw = AsyncMock()
        return persistence

    def test_redis_retrieval_tool_creation(self, mock_redis_persistence):
        """Test creating RedisRetrievalTool."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        assert tool.name == "redis_retrieval"
        assert tool.description == "Retrieve data from Redis storage backend"
        assert tool.category == "retrieval"
        assert tool.provider == "redis"
        assert tool.requires_auth is True
        assert tool.persistence == mock_redis_persistence

    def test_redis_retrieval_tool_validation_success(self, mock_redis_persistence):
        """Test RedisRetrievalTool validation with correct backend."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        # Should not raise exception
        tool.validate_configuration()

    def test_redis_retrieval_tool_validation_failure(self):
        """Test RedisRetrievalTool validation with incorrect backend."""
        wrong_persistence = Mock()
        wrong_persistence.__class__.__name__ = "MemoryPersistence"

        tool = RedisRetrievalTool(wrong_persistence)

        with pytest.raises(ToolConfigurationError):
            tool.validate_configuration()

    @pytest.mark.asyncio
    async def test_search_redis_keys(self, mock_redis_persistence):
        """Test Redis key search functionality."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        pattern = "sifaka:thought:*"
        expected_keys = ["sifaka:thought:123", "sifaka:thought:456"]

        # Mock key search
        mock_redis_persistence._list_keys.return_value = expected_keys

        result = await tool.search_redis_keys(pattern)

        # Verify search was called
        mock_redis_persistence._list_keys.assert_called_once_with(pattern)

        # Verify result
        assert result == {"pattern": pattern, "keys": expected_keys, "count": 2}

    @pytest.mark.asyncio
    async def test_search_redis_keys_no_results(self, mock_redis_persistence):
        """Test Redis key search with no results."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        pattern = "nonexistent:*"

        # Mock empty search
        mock_redis_persistence._list_keys.return_value = []

        result = await tool.search_redis_keys(pattern)

        assert result == {"pattern": pattern, "keys": [], "count": 0}

    @pytest.mark.asyncio
    async def test_get_redis_value(self, mock_redis_persistence):
        """Test Redis value retrieval."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        key = "sifaka:thought:123"
        expected_value = '{"test": "data"}'

        # Mock value retrieval
        mock_redis_persistence._get_raw.return_value = expected_value

        result = await tool.get_redis_value(key)

        # Verify retrieval was called
        mock_redis_persistence._get_raw.assert_called_once_with(key)

        # Verify result
        assert result == {"key": key, "value": expected_value}

    @pytest.mark.asyncio
    async def test_get_redis_value_not_found(self, mock_redis_persistence):
        """Test Redis value retrieval when key not found."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        key = "nonexistent:key"

        # Mock not found
        mock_redis_persistence._get_raw.return_value = None

        result = await tool.get_redis_value(key)

        assert result is None

    @pytest.mark.asyncio
    async def test_redis_error_handling(self, mock_redis_persistence):
        """Test Redis tool error handling."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        # Mock error
        mock_redis_persistence._list_keys.side_effect = Exception("Redis error")

        result = await tool.search_redis_keys("test:*")

        # Should return empty result on error
        assert result == {"pattern": "test:*", "keys": [], "count": 0}

    def test_create_pydantic_tools(self, mock_redis_persistence):
        """Test creating PydanticAI tools."""
        tool = RedisRetrievalTool(mock_redis_persistence)

        pydantic_tools = tool.create_pydantic_tools()

        assert len(pydantic_tools) == 2
        # Verify tool names/functions
        tool_names = [t.function.__name__ for t in pydantic_tools]
        assert "search_redis_keys" in tool_names
        assert "get_redis_value" in tool_names


class TestPostgreSQLRetrievalTool:
    """Test suite for PostgreSQLRetrievalTool class."""

    @pytest.fixture
    def mock_postgresql_persistence(self):
        """Create a mock PostgreSQL persistence backend."""
        persistence = Mock()
        persistence.__class__.__name__ = "PostgreSQLPersistence"
        persistence.pool = Mock()
        return persistence

    def test_postgresql_retrieval_tool_creation(self, mock_postgresql_persistence):
        """Test creating PostgreSQLRetrievalTool."""
        tool = PostgreSQLRetrievalTool(mock_postgresql_persistence)

        assert tool.name == "postgres_retrieval"
        assert tool.description == "Retrieve data from PostgreSQL storage backend"
        assert tool.category == "retrieval"
        assert tool.provider == "postgresql"
        assert tool.requires_auth is True
        assert tool.persistence == mock_postgresql_persistence

    def test_postgresql_retrieval_tool_validation_success(self, mock_postgresql_persistence):
        """Test PostgreSQLRetrievalTool validation with correct backend."""
        tool = PostgreSQLRetrievalTool(mock_postgresql_persistence)

        # Should not raise exception
        tool.validate_configuration()

    def test_postgresql_retrieval_tool_validation_failure(self):
        """Test PostgreSQLRetrievalTool validation with incorrect backend."""
        wrong_persistence = Mock()
        wrong_persistence.__class__.__name__ = "RedisPersistence"

        tool = PostgreSQLRetrievalTool(wrong_persistence)

        with pytest.raises(ToolConfigurationError):
            tool.validate_configuration()

    @pytest.mark.asyncio
    async def test_search_thoughts_sql(self, mock_postgresql_persistence):
        """Test SQL-based thought search."""
        tool = PostgreSQLRetrievalTool(mock_postgresql_persistence)

        # Mock connection and query
        mock_connection = Mock()
        mock_connection.fetch = AsyncMock()
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock()

        mock_postgresql_persistence.pool.acquire.return_value = mock_connection

        query = "SELECT * FROM thoughts WHERE prompt LIKE '%AI%'"
        expected_results = [{"id": "123", "prompt": "AI prompt", "final_text": "AI response"}]
        mock_connection.fetch.return_value = expected_results

        result = await tool.search_thoughts_sql(query)

        # Verify query was executed
        mock_connection.fetch.assert_called_once_with(query)

        # Verify result
        assert result == {"query": query, "results": expected_results, "count": 1}

    @pytest.mark.asyncio
    async def test_get_thought_analytics(self, mock_postgresql_persistence):
        """Test thought analytics retrieval."""
        tool = PostgreSQLRetrievalTool(mock_postgresql_persistence)

        # Mock connection and query
        mock_connection = Mock()
        mock_connection.fetchrow = AsyncMock()
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock()

        mock_postgresql_persistence.pool.acquire.return_value = mock_connection

        expected_analytics = {"total_thoughts": 100, "avg_iterations": 2.5, "success_rate": 0.85}
        mock_connection.fetchrow.return_value = expected_analytics

        result = await tool.get_thought_analytics()

        # Verify analytics query was executed
        mock_connection.fetchrow.assert_called_once()

        # Verify result
        assert result == expected_analytics

    def test_create_pydantic_tools(self, mock_postgresql_persistence):
        """Test creating PydanticAI tools."""
        tool = PostgreSQLRetrievalTool(mock_postgresql_persistence)

        pydantic_tools = tool.create_pydantic_tools()

        assert len(pydantic_tools) == 2
        # Verify tool names/functions
        tool_names = [t.function.__name__ for t in pydantic_tools]
        assert "search_thoughts_sql" in tool_names
        assert "get_thought_analytics" in tool_names


class TestGenericStorageRetrievalTool:
    """Test suite for GenericStorageRetrievalTool class."""

    @pytest.fixture
    def mock_generic_persistence(self):
        """Create a mock generic persistence backend."""
        persistence = Mock()
        persistence.retrieve_thought = AsyncMock()
        persistence._list_keys = AsyncMock()
        return persistence

    def test_generic_storage_retrieval_tool_creation(self, mock_generic_persistence):
        """Test creating GenericStorageRetrievalTool."""
        tool = GenericStorageRetrievalTool(mock_generic_persistence)

        assert tool.name == "storage_retrieval"
        assert tool.description == "Retrieve data from any Sifaka storage backend"
        assert tool.category == "retrieval"
        assert tool.provider == "generic"
        assert tool.persistence == mock_generic_persistence

    @pytest.mark.asyncio
    async def test_search_thoughts_generic(self, mock_generic_persistence):
        """Test generic thought search."""
        tool = GenericStorageRetrievalTool(mock_generic_persistence)

        query = "AI technology"
        expected_keys = ["sifaka:thought:123", "sifaka:thought:456"]

        # Mock key search
        mock_generic_persistence._list_keys.return_value = expected_keys

        result = await tool.search_thoughts_generic(query)

        # Verify search was called
        mock_generic_persistence._list_keys.assert_called_once()

        # Verify result structure
        assert "query" in result
        assert "matching_keys" in result
        assert "count" in result
        assert result["query"] == query

    @pytest.mark.asyncio
    async def test_get_thought_by_id(self, mock_generic_persistence):
        """Test thought retrieval by ID."""
        tool = GenericStorageRetrievalTool(mock_generic_persistence)

        thought_id = "test-thought-123"
        sample_thought = SifakaThought(prompt="Test prompt", final_text="Test response")
        sample_thought.id = thought_id

        # Mock thought retrieval
        mock_generic_persistence.retrieve_thought.return_value = sample_thought

        result = await tool.get_thought_by_id(thought_id)

        # Verify retrieval was called
        mock_generic_persistence.retrieve_thought.assert_called_once_with(thought_id)

        # Verify result
        assert result is not None
        assert result["id"] == thought_id
        assert result["prompt"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_get_thought_by_id_not_found(self, mock_generic_persistence):
        """Test thought retrieval when not found."""
        tool = GenericStorageRetrievalTool(mock_generic_persistence)

        thought_id = "nonexistent-thought"

        # Mock not found
        mock_generic_persistence.retrieve_thought.return_value = None

        result = await tool.get_thought_by_id(thought_id)

        assert result is None

    def test_create_pydantic_tools(self, mock_generic_persistence):
        """Test creating PydanticAI tools."""
        tool = GenericStorageRetrievalTool(mock_generic_persistence)

        pydantic_tools = tool.create_pydantic_tools()

        assert len(pydantic_tools) == 2
        # Verify tool names/functions
        tool_names = [t.function.__name__ for t in pydantic_tools]
        assert "search_thoughts_generic" in tool_names
        assert "get_thought_by_id" in tool_names


class TestStorageRetrievalFactory:
    """Test suite for storage retrieval tool factory functions."""

    def test_create_storage_retrieval_tools_redis(self):
        """Test creating Redis-specific retrieval tools."""
        mock_redis_persistence = Mock()
        mock_redis_persistence.__class__.__name__ = "RedisPersistence"

        with patch(
            "sifaka.tools.retrieval.storage_retrieval.RedisPersistence",
            mock_redis_persistence.__class__,
        ):
            tools = create_storage_retrieval_tools(mock_redis_persistence, backend_specific=True)

        assert len(tools) >= 2  # Should have Redis-specific tools

    def test_create_storage_retrieval_tools_postgresql(self):
        """Test creating PostgreSQL-specific retrieval tools."""
        mock_postgresql_persistence = Mock()
        mock_postgresql_persistence.__class__.__name__ = "PostgreSQLPersistence"

        with patch(
            "sifaka.tools.retrieval.storage_retrieval.PostgreSQLPersistence",
            mock_postgresql_persistence.__class__,
        ):
            tools = create_storage_retrieval_tools(
                mock_postgresql_persistence, backend_specific=True
            )

        assert len(tools) >= 2  # Should have PostgreSQL-specific tools

    def test_create_storage_retrieval_tools_generic_fallback(self):
        """Test creating generic retrieval tools as fallback."""
        mock_unknown_persistence = Mock()
        mock_unknown_persistence.__class__.__name__ = "UnknownPersistence"

        tools = create_storage_retrieval_tools(mock_unknown_persistence, backend_specific=True)

        assert len(tools) >= 2  # Should have generic tools

    def test_create_storage_retrieval_tools_force_generic(self):
        """Test creating generic retrieval tools when forced."""
        mock_redis_persistence = Mock()
        mock_redis_persistence.__class__.__name__ = "RedisPersistence"

        tools = create_storage_retrieval_tools(mock_redis_persistence, backend_specific=False)

        assert len(tools) >= 2  # Should have generic tools even for Redis backend
