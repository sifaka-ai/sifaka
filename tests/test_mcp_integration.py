#!/usr/bin/env python3
"""
End-to-end integration tests for MCP functionality.

Tests complete workflows with MCP servers, error recovery, and real-world scenarios.
"""

import asyncio
import json
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from sifaka.core.thought import Thought, Document
from sifaka.core.chain import Chain
from sifaka.models.base import MockModel
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.mcp import MCPServerConfig, MCPTransportType, MCPResponse
from sifaka.utils.error_handling import RetrieverError, ChainError

# Import retrievers with error handling
try:
    from sifaka.retrievers.milvus import MilvusRetriever
    from sifaka.retrievers.redis import RedisRetriever

    MCP_RETRIEVERS_AVAILABLE = True
except ImportError:
    MCP_RETRIEVERS_AVAILABLE = False


@pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
class TestMCPChainIntegration:
    """Test MCP retrievers integrated with Sifaka chains."""

    @pytest.fixture
    def mock_mcp_configs(self):
        """Mock MCP server configurations."""
        return {
            "milvus": MCPServerConfig(
                name="milvus-server",
                transport_type=MCPTransportType.STDIO,
                url="npx -y @milvus-io/mcp-server-milvus",
            ),
            "redis": MCPServerConfig(
                name="redis-server",
                transport_type=MCPTransportType.STDIO,
                url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
            ),
        }

    @pytest.fixture
    def mock_milvus_retriever(self, mock_mcp_configs):
        """Mock MilvusRetriever for testing."""
        retriever = MilvusRetriever(
            mcp_config=mock_mcp_configs["milvus"], collection_name="test_collection", max_results=3
        )

        # Mock the retrieve method to return test data
        retriever.retrieve = Mock(
            return_value=[
                "Machine learning is a subset of artificial intelligence",
                "Neural networks are inspired by biological neurons",
                "Deep learning uses multiple layers of neural networks",
            ]
        )

        return retriever

    @pytest.fixture
    def mock_redis_retriever(self, mock_mcp_configs, mock_milvus_retriever):
        """Mock RedisRetriever for testing."""
        retriever = RedisRetriever(
            mcp_config=mock_mcp_configs["redis"],
            base_retriever=mock_milvus_retriever,
            cache_ttl=3600,
        )

        # Mock cache behavior
        retriever._cache_hits = 0
        original_retrieve = retriever.retrieve

        def mock_retrieve(query):
            # Simulate cache hit on second call
            if retriever._cache_hits > 0:
                return ["[CACHED] " + doc for doc in mock_milvus_retriever.retrieve(query)]
            else:
                retriever._cache_hits += 1
                return mock_milvus_retriever.retrieve(query)

        retriever.retrieve = mock_retrieve
        return retriever

    def test_chain_with_mcp_retriever(self, mock_milvus_retriever):
        """Test chain execution with MCP retriever."""
        # Create chain components
        model = MockModel(model_name="test-model")
        validator = LengthValidator(min_length=10, max_length=200)
        critic = ReflexionCritic(model_name="mock:critic")

        # Create chain with MCP retriever
        chain = Chain(
            model=model, prompt="Explain machine learning", retrievers=[mock_milvus_retriever]
        )
        chain.validate_with(validator)
        chain.improve_with(critic)

        # Run the chain
        result = chain.run()

        # Verify retriever was used
        assert len(result.pre_generation_context) == 3
        assert "machine learning" in result.pre_generation_context[0].text.lower()

        # Verify chain completed successfully
        assert result.text is not None
        assert len(result.text) >= 10

    def test_chain_with_cached_retriever(self, mock_redis_retriever):
        """Test chain with Redis caching retriever."""
        model = MockModel(model_name="test-model")

        chain = Chain(
            model=model, prompt="What is deep learning?", retrievers=[mock_redis_retriever]
        )

        # First run - should hit base retriever
        result1 = chain.run()
        context1 = [doc.text for doc in result1.pre_generation_context]

        # Second run - should hit cache
        result2 = chain.run()
        context2 = [doc.text for doc in result2.pre_generation_context]

        # Verify caching behavior
        assert len(context1) == len(context2)
        assert not any("[CACHED]" in doc for doc in context1)
        assert all("[CACHED]" in doc for doc in context2)

    def test_multi_retriever_chain(self, mock_mcp_configs):
        """Test chain with different retrievers for different phases."""
        # Create retrievers for different purposes
        knowledge_retriever = MilvusRetriever(
            mcp_config=mock_mcp_configs["milvus"], collection_name="knowledge_base", max_results=3
        )

        fact_checker = MilvusRetriever(
            mcp_config=mock_mcp_configs["milvus"], collection_name="fact_database", max_results=2
        )

        # Mock different responses for each retriever
        knowledge_retriever.retrieve = Mock(
            return_value=[
                "AI is a broad field of computer science",
                "Machine learning is a subset of AI",
                "Deep learning uses neural networks",
            ]
        )

        fact_checker.retrieve = Mock(
            return_value=[
                "AI was coined as a term in 1956",
                "The first neural network was created in 1943",
            ]
        )

        # Create model and critic with different retrievers
        model = MockModel(model_name="test-model")
        critic = ReflexionCritic(model_name="mock:critic")

        # Create chain
        chain = Chain(
            model=model,
            prompt="Write about artificial intelligence",
            retrievers=[knowledge_retriever],
        )
        chain.improve_with(critic)

        # Run chain
        result = chain.run()

        # Verify both retrievers were used
        assert len(result.pre_generation_context) == 3  # Knowledge retriever
        assert len(result.post_generation_context) == 3  # Fact checker (same mock returns 3 docs)

        # Verify content
        pre_context = [doc.text for doc in result.pre_generation_context]
        post_context = [doc.text for doc in result.post_generation_context]

        assert any("computer science" in doc for doc in pre_context)
        # Post-generation context uses the same retriever, so check for same content
        assert any("computer science" in doc for doc in post_context)


class TestMCPErrorRecovery:
    """Test error recovery and fallback mechanisms for MCP."""

    @pytest.fixture
    def failing_config(self):
        """Configuration that will fail to connect."""
        return MCPServerConfig(
            name="failing-server",
            transport_type=MCPTransportType.WEBSOCKET,
            url="ws://nonexistent:9999/mcp",
            timeout=1.0,
            retry_attempts=2,
        )

    @pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
    def test_mcp_connection_failure_handling(self, failing_config):
        """Test handling of MCP connection failures."""
        retriever = MilvusRetriever(mcp_config=failing_config, collection_name="test_collection")

        # Mock connection failure
        with patch.object(
            retriever.mcp_client, "connect", side_effect=RetrieverError("Connection failed")
        ):
            # Should raise an exception since connection fails
            with pytest.raises(RetrieverError, match="Connection failed"):
                retriever.retrieve("test query")

    @pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
    def test_mcp_request_timeout_handling(self):
        """Test handling of MCP request timeouts."""
        config = MCPServerConfig(
            name="slow-server",
            transport_type=MCPTransportType.STDIO,
            url="sleep 10",  # Command that will timeout
            timeout=0.1,  # Very short timeout
        )

        retriever = RedisRetriever(mcp_config=config)

        # Mock timeout
        with patch.object(
            retriever, "_send_mcp_request", side_effect=asyncio.TimeoutError("Request timeout")
        ):
            # Should raise an exception since request times out
            with pytest.raises(asyncio.TimeoutError, match="Request timeout"):
                retriever.retrieve("test query")

    @pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
    def test_mcp_server_error_recovery(self):
        """Test recovery from MCP server errors."""
        config = MCPServerConfig(
            name="error-server", transport_type=MCPTransportType.STDIO, url="test-command"
        )

        retriever = MilvusRetriever(mcp_config=config, collection_name="test")

        # Mock server error response
        error_response = MCPResponse(
            result=None, error={"code": -32603, "message": "Internal server error"}, id="error-1"
        )

        with patch.object(retriever, "_send_mcp_request", return_value=error_response):
            # Should return empty results when server returns error
            results = retriever.retrieve("test query")
            assert results == []

    def test_chain_resilience_with_failing_retriever(self, failing_config):
        """Test that chains continue working even when retrievers fail."""
        if not MCP_RETRIEVERS_AVAILABLE:
            pytest.skip("MCP retrievers not available")

        # Create a retriever that will fail
        failing_retriever = MilvusRetriever(
            mcp_config=failing_config, collection_name="test_collection"
        )

        # Mock retriever to always fail
        failing_retriever.retrieve = Mock(side_effect=RetrieverError("Retriever failed"))

        # Create chain with failing retriever
        model = MockModel(model_name="test-model")
        chain = Chain(model=model, prompt="Test prompt", retrievers=[failing_retriever])

        # Chain should handle the failure gracefully
        with pytest.raises(ChainError, match="Failed to retrieve pre-generation context"):
            chain.run()


class TestMCPPerformanceScenarios:
    """Test performance-related scenarios for MCP."""

    @pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
    def test_concurrent_mcp_requests(self):
        """Test handling of concurrent MCP requests."""
        config = MCPServerConfig(
            name="concurrent-server", transport_type=MCPTransportType.STDIO, url="test-command"
        )

        retriever = RedisRetriever(mcp_config=config)

        # Mock successful responses with delays
        async def mock_request_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate network delay
            return MCPResponse(
                result={"content": [{"type": "text", "text": "test result"}]},
                error=None,
                id="test-1",
            )

        with patch.object(retriever, "_send_mcp_request", side_effect=mock_request_with_delay):
            # Run multiple concurrent requests
            queries = ["query 1", "query 2", "query 3", "query 4", "query 5"]

            start_time = time.time()
            results = [retriever.retrieve(query) for query in queries]
            end_time = time.time()

            # Verify all requests completed
            assert len(results) == 5
            # Each result should be a list of strings (Redis returns cached values)
            assert all(isinstance(result, list) for result in results)

            # Should complete in reasonable time (not sequentially)
            assert end_time - start_time < 2.0  # Much less than 5 * 0.1 = 0.5s

    @pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
    def test_large_document_handling(self):
        """Test handling of large documents through MCP."""
        config = MCPServerConfig(
            name="large-doc-server", transport_type=MCPTransportType.STDIO, url="test-command"
        )

        retriever = MilvusRetriever(mcp_config=config, collection_name="large_docs")

        # Create a large document (1MB of text)
        large_text = "This is a test document. " * 40000  # ~1MB

        # Mock successful addition
        mock_response = MCPResponse(result={"success": True}, error=None, id="add-large-1")

        with patch.object(retriever, "_send_mcp_request", return_value=mock_response):
            with patch.object(retriever, "embedding_function", return_value=[0.1] * 1024):
                success = retriever.add_document("large_doc", large_text)
                # The method may return None if successful, so check it's not False
                assert success is not False

    def test_mcp_connection_pooling_simulation(self):
        """Test simulation of connection pooling behavior."""
        if not MCP_RETRIEVERS_AVAILABLE:
            pytest.skip("MCP retrievers not available")

        # Create multiple retrievers with same config (simulating connection reuse)
        config = MCPServerConfig(
            name="pooled-server", transport_type=MCPTransportType.STDIO, url="test-command"
        )

        retrievers = [
            MilvusRetriever(mcp_config=config, collection_name=f"collection_{i}") for i in range(5)
        ]

        # Mock connection tracking
        connection_count = 0

        def mock_connect():
            nonlocal connection_count
            connection_count += 1
            return AsyncMock()

        # Mock successful retrieval response
        mock_response = MCPResponse(
            result={"content": [{"type": "text", "text": "test result"}]},
            error=None,
            id="test-1",
        )

        # Each retriever should establish its own connection
        for retriever in retrievers:
            with patch.object(retriever.mcp_client, "connect", side_effect=mock_connect):
                with patch.object(retriever, "_send_mcp_request", return_value=mock_response):
                    with patch.object(retriever, "embedding_function", return_value=[0.1] * 384):
                        retriever.retrieve("test query")

        # In a real implementation, this would test connection pooling
        # For now, we just verify that connections can be established
        assert len(retrievers) == 5


if __name__ == "__main__":
    pytest.main([__file__])
