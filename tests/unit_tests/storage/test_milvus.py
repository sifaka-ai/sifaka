"""Comprehensive unit tests for Milvus storage backend.

This module tests the MilvusPersistence implementation:
- Milvus vector database integration
- Embedding-based storage and retrieval
- Vector similarity search operations
- Error handling and edge cases

Tests cover:
- Vector storage and retrieval operations
- Embedding generation and management
- Milvus-specific features (collections, indexing)
- Error scenarios and recovery
- Mock-based testing without actual Milvus connection
"""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sifaka.core.thought import SifakaThought
from sifaka.storage.milvus import MilvusPersistence


class TestMilvusPersistence:
    """Test suite for MilvusPersistence class."""

    @pytest.fixture
    def mock_milvus_client(self):
        """Create a mock Milvus client."""
        client = Mock()
        client.create_collection = Mock()
        client.insert = Mock()
        client.search = Mock()
        client.query = Mock()
        client.delete = Mock()
        client.has_collection = Mock(return_value=True)
        client.load_collection = Mock()
        client.create_index = Mock()
        return client

    @pytest.fixture
    def mock_embedding_function(self):
        """Create a mock embedding function."""

        def embedding_func(text):
            # Return a mock 768-dimensional embedding
            return np.random.rand(768).tolist()

        return embedding_func

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(
            prompt="Test Milvus storage with vector embeddings",
            final_text="This is a test thought for Milvus vector storage and retrieval.",
            iteration=1,
            max_iterations=3,
        )
        thought.add_generation("Generated text", "gpt-4", {"temperature": 0.7})
        thought.add_validation("length_validator", True, {"word_count": 15})
        return thought

    def test_milvus_persistence_creation_minimal(self, mock_embedding_function):
        """Test creating MilvusPersistence with minimal parameters."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )

        assert persistence.connection_params == {"host": "localhost", "port": 19530}
        assert persistence.embedding_function == mock_embedding_function
        assert persistence.collection_name == "sifaka_thoughts"  # Default
        assert persistence.embedding_dim == 768  # Default

    def test_milvus_persistence_creation_with_custom_params(self, mock_embedding_function):
        """Test creating MilvusPersistence with custom parameters."""
        persistence = MilvusPersistence(
            connection_params={"uri": "https://cloud.milvus.io", "token": "test"},
            embedding_function=mock_embedding_function,
            collection_name="custom_collection",
            embedding_dim=1536,
            key_prefix="test_prefix",
        )

        assert persistence.connection_params == {"uri": "https://cloud.milvus.io", "token": "test"}
        assert persistence.collection_name == "custom_collection"
        assert persistence.embedding_dim == 1536
        assert persistence.key_prefix == "test_prefix"

    @pytest.mark.asyncio
    async def test_ensure_connection(self, mock_milvus_client, mock_embedding_function):
        """Test Milvus connection establishment."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )

        with patch("pymilvus.MilvusClient", return_value=mock_milvus_client):
            await persistence._ensure_connection()

            assert persistence.client == mock_milvus_client
            # Verify collection setup was called
            mock_milvus_client.has_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_thought_vector(
        self, mock_milvus_client, mock_embedding_function, sample_thought
    ):
        """Test storing thought as vector in Milvus."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        thought_data = json.dumps(sample_thought.model_dump())

        # Mock successful insertion
        mock_milvus_client.insert.return_value = {"insert_count": 1}

        await persistence._store_thought_vector(sample_thought.id, thought_data)

        # Verify insert was called
        mock_milvus_client.insert.assert_called_once()
        call_args = mock_milvus_client.insert.call_args

        # Verify collection name
        assert call_args[1]["collection_name"] == "sifaka_thoughts"

        # Verify data structure
        data = call_args[1]["data"][0]
        assert data["id"] == sample_thought.id
        assert "embedding" in data
        assert len(data["embedding"]) == 768  # Default embedding dimension

    @pytest.mark.asyncio
    async def test_search_similar_thoughts(self, mock_milvus_client, mock_embedding_function):
        """Test vector similarity search."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        query_text = "test query"

        # Mock search results
        mock_results = [
            [
                {"id": "thought-1", "distance": 0.1, "entity": {"prompt": "similar prompt"}},
                {"id": "thought-2", "distance": 0.2, "entity": {"prompt": "another prompt"}},
            ]
        ]
        mock_milvus_client.search.return_value = mock_results

        results = await persistence._search_similar_thoughts(query_text, limit=5)

        # Verify search was called
        mock_milvus_client.search.assert_called_once()
        call_args = mock_milvus_client.search.call_args

        # Verify search parameters
        assert call_args[1]["collection_name"] == "sifaka_thoughts"
        assert call_args[1]["limit"] == 5
        assert "anns_field" in call_args[1]
        assert "data" in call_args[1]

        # Verify results
        assert len(results) == 2
        assert results[0]["id"] == "thought-1"
        assert results[1]["id"] == "thought-2"

    @pytest.mark.asyncio
    async def test_retrieve_thought_by_id(
        self, mock_milvus_client, mock_embedding_function, sample_thought
    ):
        """Test retrieving thought by ID from Milvus."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        # Mock query result
        thought_data = sample_thought.model_dump()
        mock_results = [
            {
                "id": sample_thought.id,
                "prompt": thought_data["prompt"],
                "final_text": thought_data["final_text"],
                "metadata": json.dumps(thought_data),
            }
        ]
        mock_milvus_client.query.return_value = mock_results

        result = await persistence._retrieve_thought_by_id(sample_thought.id)

        # Verify query was called
        mock_milvus_client.query.assert_called_once()
        call_args = mock_milvus_client.query.call_args

        # Verify query parameters
        assert call_args[1]["collection_name"] == "sifaka_thoughts"
        assert sample_thought.id in call_args[1]["filter"]

        # Verify result
        assert result is not None
        assert json.loads(result)["id"] == sample_thought.id

    @pytest.mark.asyncio
    async def test_store_raw_thought_key(self, mock_milvus_client, mock_embedding_function):
        """Test storing raw data with thought key."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        key = "sifaka:thought:123"
        data = '{"test": "data"}'

        # Mock successful insertion
        mock_milvus_client.insert.return_value = {"insert_count": 1}

        await persistence._store_raw(key, data)

        # Verify thought vector storage was called
        mock_milvus_client.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_raw_non_thought_key(self, mock_milvus_client, mock_embedding_function):
        """Test storing raw data with non-thought key (should be skipped)."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        key = "sifaka:config:settings"
        data = '{"setting": "value"}'

        await persistence._store_raw(key, data)

        # Verify insert was NOT called for non-thought data
        mock_milvus_client.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_raw_thought_key(self, mock_milvus_client, mock_embedding_function):
        """Test retrieving raw data with thought key."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        key = "sifaka:thought:123"
        expected_data = '{"test": "data"}'

        # Mock query result
        mock_results = [{"metadata": expected_data}]
        mock_milvus_client.query.return_value = mock_results

        result = await persistence._retrieve_raw(key)

        # Verify query was called
        mock_milvus_client.query.assert_called_once()
        assert result == expected_data

    @pytest.mark.asyncio
    async def test_retrieve_raw_not_found(self, mock_milvus_client, mock_embedding_function):
        """Test retrieving raw data when not found."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        key = "sifaka:thought:nonexistent"

        # Mock empty query result
        mock_milvus_client.query.return_value = []

        result = await persistence._retrieve_raw(key)

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_raw_thought(self, mock_milvus_client, mock_embedding_function):
        """Test deleting thought from Milvus."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        key = "sifaka:thought:123"

        # Mock successful deletion
        mock_milvus_client.delete.return_value = {"delete_count": 1}

        result = await persistence._delete_raw(key)

        # Verify delete was called
        mock_milvus_client.delete.assert_called_once()
        call_args = mock_milvus_client.delete.call_args

        # Verify delete parameters
        assert call_args[1]["collection_name"] == "sifaka_thoughts"
        assert "123" in call_args[1]["filter"]

        assert result is True

    @pytest.mark.asyncio
    async def test_list_keys_thoughts(self, mock_milvus_client, mock_embedding_function):
        """Test listing thought keys from Milvus."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        pattern = "sifaka:thought:*"

        # Mock query results
        mock_results = [
            {"id": "thought-1", "key_prefix": "sifaka"},
            {"id": "thought-2", "key_prefix": "sifaka"},
        ]
        mock_milvus_client.query.return_value = mock_results

        result = await persistence._list_keys(pattern)

        # Verify query was called
        mock_milvus_client.query.assert_called_once()

        # Verify results
        expected_keys = ["sifaka:thought:thought-1", "sifaka:thought:thought-2"]
        assert result == expected_keys

    @pytest.mark.asyncio
    async def test_embedding_generation(self, mock_embedding_function):
        """Test embedding generation for text."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )

        test_text = "This is a test text for embedding generation."

        embedding = persistence._generate_embedding(test_text)

        # Verify embedding properties
        assert isinstance(embedding, list)
        assert len(embedding) == 768  # Default dimension
        assert all(isinstance(x, (int, float)) for x in embedding)

    @pytest.mark.asyncio
    async def test_collection_creation(self, mock_milvus_client, mock_embedding_function):
        """Test Milvus collection creation."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )

        # Mock collection doesn't exist
        mock_milvus_client.has_collection.return_value = False

        with patch("pymilvus.MilvusClient", return_value=mock_milvus_client):
            await persistence._ensure_connection()

            # Verify collection creation was called
            mock_milvus_client.create_collection.assert_called_once()
            call_args = mock_milvus_client.create_collection.call_args
            assert call_args[1]["collection_name"] == "sifaka_thoughts"

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, mock_embedding_function):
        """Test handling of connection errors."""
        persistence = MilvusPersistence(
            connection_params={"host": "invalid", "port": 19530},
            embedding_function=mock_embedding_function,
        )

        with patch("pymilvus.MilvusClient", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await persistence._ensure_connection()

    @pytest.mark.asyncio
    async def test_embedding_error_handling(self, mock_milvus_client):
        """Test handling of embedding generation errors."""

        def failing_embedding_func(text):
            raise Exception("Embedding generation failed")

        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=failing_embedding_func,
        )
        persistence.client = mock_milvus_client

        with pytest.raises(Exception, match="Embedding generation failed"):
            await persistence._store_thought_vector("test-id", '{"test": "data"}')

    @pytest.mark.asyncio
    async def test_store_thought_complete(
        self, mock_milvus_client, mock_embedding_function, sample_thought
    ):
        """Test complete thought storage in Milvus."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        # Mock successful insertion
        mock_milvus_client.insert.return_value = {"insert_count": 1}

        await persistence.store_thought(sample_thought)

        # Verify storage was called
        mock_milvus_client.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_thought_complete(
        self, mock_milvus_client, mock_embedding_function, sample_thought
    ):
        """Test complete thought retrieval from Milvus."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=mock_embedding_function,
        )
        persistence.client = mock_milvus_client

        # Mock successful retrieval
        thought_data = json.dumps(sample_thought.model_dump())
        mock_results = [{"metadata": thought_data}]
        mock_milvus_client.query.return_value = mock_results

        result = await persistence.retrieve_thought(sample_thought.id)

        # Verify retrieval was called
        mock_milvus_client.query.assert_called_once()

        # Verify result is a SifakaThought
        assert isinstance(result, SifakaThought)
        assert result.id == sample_thought.id
        assert result.prompt == sample_thought.prompt

    def test_key_parsing(self):
        """Test Milvus key parsing logic."""
        persistence = MilvusPersistence(
            connection_params={"host": "localhost", "port": 19530},
            embedding_function=lambda x: [0.1] * 768,
        )

        # Test thought key parsing
        thought_key = "sifaka:thought:123"
        parts = thought_key.split(":")
        assert len(parts) == 3
        assert parts[1] == "thought"
        assert parts[2] == "123"

        # Test non-thought key
        config_key = "sifaka:config:settings"
        parts = config_key.split(":")
        assert len(parts) == 3
        assert parts[1] == "config"  # Should be skipped by Milvus storage
