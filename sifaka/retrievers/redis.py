"""Redis-based retriever using MCP.

This module provides a Redis-based retriever that uses MCP (Model Context Protocol)
for all Redis operations, providing standardized communication and better error handling.

The RedisRetriever can operate in two modes:
1. Standalone: Store and retrieve documents directly from Redis via MCP
2. Caching wrapper: Cache results from another retriever for performance

Example:
    ```python
    from sifaka.retrievers.redis import RedisRetriever
    from sifaka.retrievers.mcp_base import MCPServerConfig, MCPTransportType
    from sifaka.retrievers.base import InMemoryRetriever

    # Create MCP server configuration
    config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/redis"
    )

    # Standalone Redis retriever
    redis_retriever = RedisRetriever(mcp_config=config)
    redis_retriever.add_document("doc1", "This is a document about AI.")

    # As a caching wrapper
    base_retriever = InMemoryRetriever()
    cached_retriever = RedisRetriever(
        mcp_config=config,
        base_retriever=base_retriever
    )
    ```
"""

import asyncio
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Retriever
from sifaka.core.thought import Document, Thought
from sifaka.retrievers.base import BaseMCPRetriever, MCPServerConfig, MCPRequest
from sifaka.utils.error_handling import RetrieverError, error_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class RedisRetriever:
    """Redis-based retriever with caching capabilities using MCP.

    This retriever uses MCP (Model Context Protocol) for all Redis operations,
    providing standardized communication, better error handling, and improved
    scalability.

    This retriever can operate in two modes:
    1. Standalone: Store and retrieve documents directly from Redis
    2. Caching wrapper: Cache results from another retriever

    Attributes:
        mcp_retriever: Internal MCP-based retriever.
        base_retriever: Optional base retriever to cache results from.
        cache_ttl: Time-to-live for cached results in seconds.
        key_prefix: Prefix for Redis keys.
        max_results: Maximum number of documents to return.
    """

    def __init__(
        self,
        mcp_config: MCPServerConfig,
        base_retriever: Optional[Retriever] = None,
        cache_ttl: int = 3600,  # 1 hour default
        key_prefix: str = "sifaka:retriever",
        max_results: int = 10,
    ):
        """Initialize the Redis retriever with MCP backend.

        Args:
            mcp_config: MCP server configuration for Redis.
            base_retriever: Optional base retriever to cache results from.
            cache_ttl: Cache time-to-live in seconds (default: 3600).
            key_prefix: Prefix for Redis keys (default: "sifaka:retriever").
            max_results: Maximum number of documents to return (default: 10).
            **kwargs: Additional configuration parameters.
        """
        self.base_retriever = base_retriever
        self.cache_ttl = cache_ttl
        self.key_prefix = key_prefix
        self.max_results = max_results

        # Create internal MCP-based retriever
        self.mcp_retriever = BaseMCPRetriever(mcp_config, max_results)
        self._loop = None

        logger.info(f"Initialized RedisRetriever with MCP backend: {mcp_config.name}")

    def _get_loop(self):
        """Get or create event loop for async operations."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Add a document to Redis storage.

        Args:
            doc_id: The document ID.
            text: The document text.
            metadata: Optional metadata for the document.
            ttl: Optional time-to-live for the document (uses cache_ttl if None).
        """
        with error_context(
            component="RedisRetriever",
            operation="add document",
            error_class=RetrieverError,
            message_prefix="Failed to add document to Redis via MCP",
        ):
            # Add Redis-specific context to metadata
            enhanced_metadata = {
                "key_prefix": self.key_prefix,
                "ttl": ttl or self.cache_ttl,
                **(metadata or {}),
            }

            # Use async operation via event loop
            loop = self._get_loop()
            loop.run_until_complete(
                self.mcp_retriever.add_document(doc_id, text, enhanced_metadata)
            )

            logger.debug(f"Added document {doc_id} to Redis with TTL: {ttl or self.cache_ttl}")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document from Redis storage.

        Args:
            doc_id: The document ID.

        Returns:
            The document data or None if not found.
        """
        with error_context(
            component="RedisRetriever",
            operation="get document",
            error_class=RetrieverError,
            message_prefix="Failed to get document from Redis via MCP",
        ):

            async def _get_document():
                if not self.mcp_retriever.mcp_client._connected:
                    await self.mcp_retriever.mcp_client.connect()

                request = MCPRequest(
                    method="get_document", params={"doc_id": doc_id, "key_prefix": self.key_prefix}
                )

                response = await self.mcp_retriever.mcp_client.transport.send_request(request)

                if response.result and response.result.get("found"):
                    return response.result["document"]

                return None

            loop = self._get_loop()
            return loop.run_until_complete(_get_document())

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        with error_context(
            component="RedisRetriever",
            operation="retrieval",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents from Redis via MCP",
        ):
            logger.debug(f"Retrieving documents for query: {query[:50]}...")

            async def _retrieve():
                # Add Redis-specific context to the query
                context = {
                    "key_prefix": self.key_prefix,
                    "cache_ttl": self.cache_ttl,
                    "search_type": "keyword_matching",
                    "has_base_retriever": self.base_retriever is not None,
                }

                # Try to get from cache first via MCP
                documents = await self.mcp_retriever.mcp_client.query(query, context)

                if documents:
                    # Cache hit
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return [doc.text for doc in documents]

                logger.debug(f"Cache miss for query: {query[:50]}...")

                # If we have a base retriever, use it and cache the results
                if self.base_retriever:
                    results = self.base_retriever.retrieve(query)

                    # Cache the results via MCP
                    for i, text in enumerate(results):
                        await self.mcp_retriever.add_document(
                            f"cache_{hash(query)}_{i}", text, {"query": query, "cached": True}
                        )

                    logger.debug(f"Cached {len(results)} results for query")
                    return results[: self.max_results]

                # Otherwise, search our own documents via MCP
                return [doc.text for doc in documents]

            loop = self._get_loop()
            return loop.run_until_complete(_retrieve())

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought.

        Args:
            thought: The thought to retrieve documents for.
            is_pre_generation: Whether this is pre-generation or post-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        with error_context(
            component="RedisRetriever",
            operation="retrieval for thought",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents for thought from Redis via MCP",
        ):
            # Use async operation via event loop
            loop = self._get_loop()
            enhanced_thought = loop.run_until_complete(
                self.mcp_retriever.retrieve_for_thought(thought, is_pre_generation)
            )

            # Enhance document metadata with Redis-specific info
            context_docs = (
                enhanced_thought.pre_generation_context
                if is_pre_generation
                else enhanced_thought.post_generation_context
            )

            for doc in context_docs:
                doc.metadata.update(
                    {
                        "source": "redis_mcp",
                        "key_prefix": self.key_prefix,
                        "cached": True,
                        "cache_ttl": self.cache_ttl,
                    }
                )

            return enhanced_thought

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cached results.

        Args:
            pattern: Optional pattern to match keys (default: all retriever keys).

        Returns:
            Number of keys deleted.
        """
        with error_context(
            component="RedisRetriever",
            operation="clear cache",
            error_class=RetrieverError,
            message_prefix="Failed to clear Redis cache via MCP",
        ):

            async def _clear_cache():
                if not self.mcp_retriever.mcp_client._connected:
                    await self.mcp_retriever.mcp_client.connect()

                request = MCPRequest(
                    method="clear_cache",
                    params={
                        "pattern": pattern or f"{self.key_prefix}:*",
                        "key_prefix": self.key_prefix,
                    },
                )

                response = await self.mcp_retriever.mcp_client.transport.send_request(request)
                deleted_count = response.result.get("deleted", 0) if response.result else 0

                logger.info(f"Cleared {deleted_count} keys from Redis cache")
                return deleted_count

            loop = self._get_loop()
            return loop.run_until_complete(_clear_cache())

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        with error_context(
            component="RedisRetriever",
            operation="get cache stats",
            error_class=RetrieverError,
            message_prefix="Failed to get Redis cache stats via MCP",
        ):

            async def _get_stats():
                if not self.mcp_retriever.mcp_client._connected:
                    await self.mcp_retriever.mcp_client.connect()

                request = MCPRequest(
                    method="get_cache_stats", params={"key_prefix": self.key_prefix}
                )

                response = await self.mcp_retriever.mcp_client.transport.send_request(request)

                if response.result:
                    return response.result
                else:
                    return {
                        "key_prefix": self.key_prefix,
                        "error": response.error or "Unknown error",
                        "cache_ttl": self.cache_ttl,
                    }

            loop = self._get_loop()
            return loop.run_until_complete(_get_stats())


def create_redis_retriever(
    mcp_config: MCPServerConfig,
    base_retriever: Optional[Retriever] = None,
    **kwargs: Any,
) -> RedisRetriever:
    """Factory function to create a Redis retriever.

    Args:
        mcp_config: MCP server configuration for Redis.
        base_retriever: Optional base retriever to cache results from.
        **kwargs: Additional arguments for RedisRetriever.

    Returns:
        A configured RedisRetriever instance.
    """
    return RedisRetriever(mcp_config=mcp_config, base_retriever=base_retriever, **kwargs)
