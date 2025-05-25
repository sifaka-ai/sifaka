#!/usr/bin/env python3
"""
Integration tests for MCP-based retrievers.

Tests MilvusRetriever and RedisRetriever with MCP backend functionality.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from sifaka.core.thought import Thought, Document
from sifaka.mcp import MCPServerConfig, MCPTransportType, MCPResponse
from sifaka.utils.error_handling import RetrieverError

# Import retrievers with error handling for optional dependencies
try:
    from sifaka.retrievers.milvus import MilvusRetriever

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from sifaka.retrievers.redis import RedisRetriever

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@pytest.mark.skipif(not MILVUS_AVAILABLE, reason="Milvus dependencies not available")
class TestMilvusRetriever:
    """Test MilvusRetriever with MCP backend."""

    @pytest.fixture
    def milvus_config(self):
        """Milvus MCP configuration fixture."""
        return MCPServerConfig(
            name="milvus-server",
            transport_type=MCPTransportType.STDIO,
            url="npx -y @milvus-io/mcp-server-milvus",
            timeout=30.0,
        )

    @pytest.fixture
    def milvus_retriever(self, milvus_config):
        """MilvusRetriever fixture."""
        return MilvusRetriever(
            mcp_config=milvus_config,
            collection_name="test_collection",
            embedding_model="BAAI/bge-m3",
            dimension=1024,
            max_results=5,
        )

    def test_milvus_initialization(self, milvus_retriever, milvus_config):
        """Test MilvusRetriever initialization."""
        assert milvus_retriever.collection_name == "test_collection"
        assert milvus_retriever.embedding_model == "BAAI/bge-m3"
        assert milvus_retriever.dimension == 1024
        assert milvus_retriever.max_results == 5
        assert milvus_retriever.mcp_client.config == milvus_config

    @pytest.mark.asyncio
    async def test_milvus_retrieve_async(self, milvus_retriever):
        """Test async retrieval from Milvus."""
        # Mock the MCP request
        mock_response = MCPResponse(
            result={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            [
                                {"text": "Document about AI", "score": 0.95},
                                {"text": "Machine learning basics", "score": 0.87},
                            ]
                        ),
                    }
                ]
            },
            error=None,
            id="search-1",
        )

        with patch.object(milvus_retriever, "_send_mcp_request", return_value=mock_response):
            with patch.object(milvus_retriever, "embedding_function", return_value=[0.1] * 1024):
                results = await milvus_retriever._retrieve_async("artificial intelligence")

                assert len(results) == 2
                assert "AI" in results[0]
                assert "Machine learning" in results[1]

    def test_milvus_retrieve_sync(self, milvus_retriever):
        """Test synchronous retrieval from Milvus."""

        # Mock the async method
        async def mock_retrieve_async(query):
            return ["Document about AI", "Machine learning basics"]

        with patch.object(milvus_retriever, "_retrieve_async", side_effect=mock_retrieve_async):
            results = milvus_retriever.retrieve("artificial intelligence")

            assert len(results) == 2
            assert "AI" in results[0]

    def test_milvus_retrieve_for_thought(self, milvus_retriever):
        """Test retrieval for thought enhancement."""
        thought = Thought(prompt="What is artificial intelligence?")

        # Mock the retrieve method
        with patch.object(
            milvus_retriever, "retrieve", return_value=["AI is a field of computer science"]
        ):
            enhanced_thought = milvus_retriever.retrieve_for_thought(thought)

            assert len(enhanced_thought.pre_generation_context) == 1
            assert (
                enhanced_thought.pre_generation_context[0].text
                == "AI is a field of computer science"
            )

    @pytest.mark.asyncio
    async def test_milvus_add_document_async(self, milvus_retriever):
        """Test adding document to Milvus."""
        mock_response = MCPResponse(result={"success": True}, error=None, id="add-1")

        with patch.object(milvus_retriever, "_send_mcp_request", return_value=mock_response):
            with patch.object(milvus_retriever, "embedding_function", return_value=[0.1] * 1024):
                await milvus_retriever.add_document_async(
                    "doc1", "This is a test document", {"category": "test"}
                )

                # If no exception is raised, the test passes

    def test_milvus_add_document_sync(self, milvus_retriever):
        """Test synchronous document addition."""

        async def mock_add_async(doc_id, text, metadata):  # noqa: ARG001
            pass  # Just complete without error

        with patch.object(milvus_retriever, "add_document_async", side_effect=mock_add_async):
            # Should not raise an exception
            milvus_retriever.add_document("doc1", "Test document")

    def test_milvus_error_handling(self, milvus_retriever):
        """Test error handling in Milvus retriever."""
        # Mock MCP error response
        mock_response = MCPResponse(
            result=None, error={"code": -1, "message": "Collection not found"}, id="error-1"
        )

        with patch.object(milvus_retriever, "_send_mcp_request", return_value=mock_response):
            # Error handling should return empty list, not raise
            results = asyncio.run(milvus_retriever._retrieve_async("test query"))
            assert results == []


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis dependencies not available")
class TestRedisRetriever:
    """Test RedisRetriever with MCP backend."""

    @pytest.fixture
    def redis_config(self):
        """Redis MCP configuration fixture."""
        return MCPServerConfig(
            name="redis-server",
            transport_type=MCPTransportType.STDIO,
            url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
            timeout=30.0,
        )

    @pytest.fixture
    def redis_retriever(self, redis_config):
        """RedisRetriever fixture."""
        return RedisRetriever(
            mcp_config=redis_config, cache_ttl=3600, key_prefix="sifaka:test", max_results=5
        )

    @pytest.fixture
    def mock_base_retriever(self):
        """Mock base retriever for caching tests."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = ["cached doc 1", "cached doc 2"]
        return mock_retriever

    def test_redis_initialization(self, redis_retriever, redis_config):
        """Test RedisRetriever initialization."""
        assert redis_retriever.cache_ttl == 3600
        assert redis_retriever.key_prefix == "sifaka:test"
        assert redis_retriever.max_results == 5
        assert redis_retriever.mcp_client.config == redis_config

    @pytest.mark.asyncio
    async def test_redis_cache_miss_with_base_retriever(self, redis_config, mock_base_retriever):
        """Test cache miss with base retriever."""
        redis_retriever = RedisRetriever(
            mcp_config=redis_config, base_retriever=mock_base_retriever, cache_ttl=3600
        )

        # Mock cache miss (no cached data)
        mock_get_response = MCPResponse(
            result={"content": [{"type": "text", "text": "null"}]}, error=None, id="get-1"
        )

        # Mock successful cache set
        mock_set_response = MCPResponse(result={"success": True}, error=None, id="set-1")

        with patch.object(
            redis_retriever, "_send_mcp_request", side_effect=[mock_get_response, mock_set_response]
        ):
            results = await redis_retriever._retrieve_async("test query")

            assert results == ["cached doc 1", "cached doc 2"]
            mock_base_retriever.retrieve.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_redis_cache_hit(self, redis_retriever):
        """Test cache hit scenario."""
        cached_data = ["cached doc 1", "cached doc 2"]

        # Mock cache hit
        mock_response = MCPResponse(
            result={"content": [{"type": "text", "text": json.dumps(cached_data)}]},
            error=None,
            id="get-1",
        )

        with patch.object(redis_retriever, "_send_mcp_request", return_value=mock_response):
            results = await redis_retriever._retrieve_async("test query")

            assert results == cached_data

    def test_redis_retrieve_sync(self, redis_retriever):
        """Test synchronous retrieval from Redis."""

        async def mock_retrieve_async(query):  # noqa: ARG001
            return ["cached doc 1", "cached doc 2"]

        with patch.object(redis_retriever, "_retrieve_async", side_effect=mock_retrieve_async):
            results = redis_retriever.retrieve("test query")

            assert len(results) == 2
            assert "cached doc 1" in results

    @pytest.mark.asyncio
    async def test_redis_add_document_async(self, redis_retriever):
        """Test adding document to Redis."""
        mock_response = MCPResponse(result={"success": True}, error=None, id="set-1")

        with patch.object(redis_retriever, "_send_mcp_request", return_value=mock_response):
            await redis_retriever.add_document_async(
                "doc1", "This is a test document", {"category": "test"}
            )

            # If no exception is raised, the test passes

    def test_redis_get_document(self, redis_retriever):
        """Test getting document from Redis."""
        doc_data = {"text": "Test document", "metadata": {"category": "test"}}

        mock_response = MCPResponse(
            result={"content": [{"type": "text", "text": json.dumps(doc_data)}]},
            error=None,
            id="get-1",
        )

        with patch.object(redis_retriever, "_send_mcp_request", return_value=mock_response):
            result = redis_retriever.get_document("doc1")

            assert result == doc_data

    def test_redis_clear_cache(self, redis_retriever):
        """Test clearing Redis cache."""
        # Mock list response
        mock_list_response = MCPResponse(
            result={
                "content": [
                    {"type": "text", "text": "Found keys:\nsifaka:test:doc:1\nsifaka:test:doc:2"}
                ]
            },
            error=None,
            id="list-1",
        )

        # Mock delete responses
        mock_delete_response = MCPResponse(result={"deleted": 1}, error=None, id="del-1")

        with patch.object(
            redis_retriever,
            "_send_mcp_request",
            side_effect=[mock_list_response, mock_delete_response, mock_delete_response],
        ):
            deleted_count = redis_retriever.clear_cache()

            assert deleted_count == 2

    def test_redis_error_handling(self, redis_retriever):
        """Test error handling in Redis retriever."""
        # Mock MCP error response
        mock_response = MCPResponse(
            result=None, error={"code": -1, "message": "Redis connection failed"}, id="error-1"
        )

        with patch.object(redis_retriever, "_send_mcp_request", return_value=mock_response):
            # Error handling should return empty list, not raise
            results = asyncio.run(redis_retriever._retrieve_async("test query"))
            assert results == []


class TestMCPRetrieverIntegration:
    """Test integration between different MCP retrievers."""

    @pytest.mark.skipif(
        not (MILVUS_AVAILABLE and REDIS_AVAILABLE),
        reason="Both Milvus and Redis dependencies required",
    )
    def test_multi_retriever_composition(self):
        """Test using multiple MCP retrievers together."""
        milvus_config = MCPServerConfig(
            name="milvus-server",
            transport_type=MCPTransportType.STDIO,
            url="npx -y @milvus-io/mcp-server-milvus",
        )

        redis_config = MCPServerConfig(
            name="redis-server",
            transport_type=MCPTransportType.STDIO,
            url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
        )

        milvus_retriever = MilvusRetriever(
            mcp_config=milvus_config, collection_name="knowledge_base", max_results=3
        )

        redis_retriever = RedisRetriever(
            mcp_config=redis_config, base_retriever=milvus_retriever, cache_ttl=1800
        )

        # Verify configuration
        assert milvus_retriever.mcp_client.config.name == "milvus-server"
        assert redis_retriever.mcp_client.config.name == "redis-server"
        assert redis_retriever.base_retriever == milvus_retriever

    def test_thought_enhancement_workflow(self):
        """Test complete thought enhancement workflow with MCP retrievers."""
        if not MILVUS_AVAILABLE:
            pytest.skip("Milvus dependencies not available")

        config = MCPServerConfig(
            name="test-server", transport_type=MCPTransportType.STDIO, url="test-command"
        )

        retriever = MilvusRetriever(
            mcp_config=config, collection_name="test_collection", max_results=3
        )

        thought = Thought(prompt="What is machine learning?")

        # Mock retrieval
        with patch.object(
            retriever,
            "retrieve",
            return_value=[
                "Machine learning is a subset of AI",
                "It involves training algorithms on data",
                "Common applications include image recognition",
            ],
        ):
            enhanced_thought = retriever.retrieve_for_thought(thought)

            assert len(enhanced_thought.pre_generation_context) == 3
            assert enhanced_thought.prompt == "What is machine learning?"

            # Check that documents were properly created
            for doc in enhanced_thought.pre_generation_context:
                assert isinstance(doc, Document)
                assert doc.text is not None
                assert doc.score is None or doc.score >= 0.0  # Score can be None


if __name__ == "__main__":
    pytest.main([__file__])
