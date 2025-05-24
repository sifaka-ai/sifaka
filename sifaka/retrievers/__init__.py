"""Retrievers for Sifaka.

This package provides retriever implementations for finding relevant documents
and context for text generation and improvement. Retrievers are used by the
Chain to provide context to models and critics.

Available retrievers:
- MockRetriever: Returns predefined documents for testing
- InMemoryRetriever: Simple keyword-based retrieval from in-memory documents
- RedisRetriever: Redis-based caching retriever with optional base retriever (uses MCP internally)

Vector Database Retrievers:
- MilvusRetriever: Milvus-based semantic search using vector embeddings (uses MCP internally)

Example:
    ```python
    from sifaka.retrievers import (
        MockRetriever, InMemoryRetriever, RedisRetriever, MilvusRetriever,
        MCPServerConfig, MCPTransportType
    )

    # Create basic retrievers
    mock_retriever = MockRetriever()
    memory_retriever = InMemoryRetriever()
    memory_retriever.add_document("doc1", "This is about AI.")

    # MCP server configurations
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/milvus"
    )

    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/redis"
    )

    # Retrievers using MCP internally
    milvus_retriever = MilvusRetriever(
        mcp_config=milvus_config,
        collection_name="documents",
        embedding_model="BAAI/bge-m3"
    )

    redis_retriever = RedisRetriever(
        mcp_config=redis_config,
        base_retriever=memory_retriever
    )
    ```
"""

# Import base retrievers
from sifaka.retrievers.base import MockRetriever, InMemoryRetriever

# Import Redis retriever with error handling
__all__ = ["MockRetriever", "InMemoryRetriever"]

try:
    from sifaka.retrievers.redis import RedisRetriever, create_redis_retriever

    __all__.extend(["RedisRetriever", "create_redis_retriever"])
except ImportError:
    # Redis not available
    pass

# Import Vector Database retrievers with error handling
try:
    from sifaka.retrievers.milvus import MilvusRetriever
    from sifaka.retrievers.redis import RedisRetriever, create_redis_retriever

    __all__.extend(
        [
            "MilvusRetriever",
            "RedisRetriever",
            "create_redis_retriever",
        ]
    )
except ImportError:
    # Vector DB dependencies not available
    pass

# Import MCP base classes for configuration
try:
    from sifaka.retrievers.base import MCPServerConfig, MCPTransportType, MCPClient

    __all__.extend(
        [
            "MCPServerConfig",
            "MCPTransportType",
            "MCPClient",
        ]
    )
except ImportError:
    # MCP dependencies not available
    pass

# Import specialized retrievers if they exist
# Note: specialized module may not exist yet - commented out for now
# try:
#     from sifaka.retrievers.specialized import *
# except ImportError:
#     pass
