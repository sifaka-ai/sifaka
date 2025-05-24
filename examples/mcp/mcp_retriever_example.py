#!/usr/bin/env python3
"""Example demonstrating MCP-based retrievers in Sifaka.

This example shows how to use the new MCP (Model Context Protocol) based
retrievers for Milvus and Redis. MCP provides a standardized protocol
for communication between AI applications and data sources.

Note: This example requires MCP servers to be running. In a real deployment,
you would have separate MCP servers for Milvus and Redis running on different
ports or endpoints.

For demonstration purposes, this example shows the configuration and usage
patterns without requiring actual MCP servers to be running.
"""

import asyncio
import os
from typing import Dict, Any

from sifaka.retrievers import MilvusRetriever, RedisRetriever, MCPServerConfig, MCPTransportType
from sifaka.core.thought import Thought


def create_mcp_configs() -> Dict[str, MCPServerConfig]:
    """Create MCP server configurations.

    Returns:
        Dictionary of MCP server configurations.
    """
    configs = {
        "milvus": MCPServerConfig(
            name="milvus-server",
            transport_type=MCPTransportType.WEBSOCKET,
            url="ws://localhost:8080/mcp/milvus",
            timeout=30.0,
            retry_attempts=3,
            retry_delay=1.0,
            capabilities=["query", "add_document", "initialize_collection", "clear_collection"],
        ),
        "redis": MCPServerConfig(
            name="redis-server",
            transport_type=MCPTransportType.WEBSOCKET,
            url="ws://localhost:8081/mcp/redis",
            timeout=30.0,
            retry_attempts=3,
            retry_delay=1.0,
            capabilities=["query", "add_document", "get_document", "clear_cache"],
        ),
    }

    return configs


async def demonstrate_mcp_milvus_retriever():
    """Demonstrate MCP-based Milvus retriever."""
    print("=== MCP Milvus Retriever Demo ===")

    # Create MCP configuration
    configs = create_mcp_configs()
    milvus_config = configs["milvus"]

    # Create Milvus retriever (uses MCP internally)
    retriever = MilvusRetriever(
        mcp_config=milvus_config,
        collection_name="demo_collection",
        embedding_model="BAAI/bge-m3",
        dimension=1024,
        max_results=3,
    )

    print(f"Created MCP Milvus retriever: {retriever.mcp_retriever.config.name}")
    print(f"Collection: {retriever.collection_name}")
    print(f"Embedding model: {retriever.embedding_model}")
    print(f"Dimension: {retriever.dimension}")

    # Note: In a real scenario, you would:
    # 1. Initialize the collection
    # await retriever.initialize_collection()

    # 2. Add documents
    # await retriever.add_document("doc1", "This is about artificial intelligence.")
    # await retriever.add_document("doc2", "This is about machine learning.")

    # 3. Retrieve documents
    # results = await retriever.retrieve("Tell me about AI")
    # print(f"Retrieved {len(results)} documents")

    # 4. Use with Thought container
    # thought = Thought(prompt="What is artificial intelligence?")
    # enhanced_thought = await retriever.retrieve_for_thought(thought)

    print("MCP Milvus retriever configuration completed.")
    print()


async def demonstrate_mcp_redis_retriever():
    """Demonstrate MCP-based Redis retriever."""
    print("=== MCP Redis Retriever Demo ===")

    # Create MCP configuration
    configs = create_mcp_configs()
    redis_config = configs["redis"]

    # Create Redis retriever (uses MCP internally)
    retriever = RedisRetriever(
        mcp_config=redis_config, cache_ttl=3600, key_prefix="sifaka:demo", max_results=5  # 1 hour
    )

    print(f"Created MCP Redis retriever: {retriever.mcp_retriever.config.name}")
    print(f"Cache TTL: {retriever.cache_ttl} seconds")
    print(f"Key prefix: {retriever.key_prefix}")
    print(f"Max results: {retriever.max_results}")

    # Note: In a real scenario, you would:
    # 1. Add documents to cache
    # await retriever.add_document("doc1", "Cached document about AI", ttl=7200)

    # 2. Retrieve documents (with caching)
    # results = await retriever.retrieve("artificial intelligence")
    # print(f"Retrieved {len(results)} documents from cache")

    # 3. Get cache statistics
    # stats = await retriever.get_cache_stats()
    # print(f"Cache stats: {stats}")

    # 4. Clear cache if needed
    # deleted_count = await retriever.clear_cache()
    # print(f"Cleared {deleted_count} cache entries")

    print("MCP Redis retriever configuration completed.")
    print()


async def demonstrate_mcp_retriever_composition():
    """Demonstrate using multiple MCP retrievers together."""
    print("=== MCP Retriever Composition Demo ===")

    configs = create_mcp_configs()

    # Create both retrievers
    milvus_retriever = MilvusRetriever(
        mcp_config=configs["milvus"],
        collection_name="knowledge_base",
        embedding_model="BAAI/bge-m3",
        max_results=3,
    )

    redis_retriever = RedisRetriever(
        mcp_config=configs["redis"],
        cache_ttl=1800,  # 30 minutes
        key_prefix="sifaka:cache",
        max_results=5,
    )

    print("Created MCP retriever composition:")
    print(
        f"- Milvus: {milvus_retriever.mcp_retriever.config.name} ({milvus_retriever.collection_name})"
    )
    print(f"- Redis: {redis_retriever.mcp_retriever.config.name} ({redis_retriever.key_prefix})")

    # Example usage pattern:
    # 1. Check Redis cache first for fast retrieval
    # 2. If cache miss, query Milvus for semantic search
    # 3. Cache Milvus results in Redis for future queries

    print("\nUsage pattern:")
    print("1. Query Redis cache for fast retrieval")
    print("2. On cache miss, query Milvus for semantic search")
    print("3. Cache Milvus results in Redis for future queries")
    print("4. This provides both semantic search and caching benefits")
    print()


def demonstrate_synchronous_usage():
    """Demonstrate synchronous usage of MCP retrievers."""
    print("=== Synchronous MCP Retriever Usage Demo ===")

    configs = create_mcp_configs()

    # Create retrievers (they handle async internally)
    milvus_retriever = MilvusRetriever(
        mcp_config=configs["milvus"], collection_name="sync_collection", max_results=3
    )

    redis_retriever = RedisRetriever(mcp_config=configs["redis"], cache_ttl=3600, max_results=5)

    print("Created MCP retrievers with synchronous interfaces:")
    print(f"- Milvus: {milvus_retriever.mcp_retriever.config.name}")
    print(f"- Redis: {redis_retriever.mcp_retriever.config.name}")

    # Note: These retrievers provide synchronous interfaces
    # that handle async operations internally via event loops

    # Example usage (would work if MCP servers were running):
    # results = milvus_retriever.retrieve("query text")
    # thought = Thought(prompt="test prompt")
    # enhanced_thought = milvus_retriever.retrieve_for_thought(thought)

    print("Retrievers provide synchronous interfaces with MCP internally.")
    print()


async def main():
    """Main demonstration function."""
    print("MCP-Based Retriever Examples for Sifaka")
    print("=" * 50)
    print()

    print("This example demonstrates the new MCP (Model Context Protocol)")
    print("based retrievers that replace the direct database connections.")
    print()
    print("Benefits of MCP-based retrievers:")
    print("- Standardized communication protocol")
    print("- Better error handling and fallback mechanisms")
    print("- Multi-server composition capabilities")
    print("- Transport abstraction (WebSocket, HTTP, etc.)")
    print("- Improved scalability and maintainability")
    print()

    # Run demonstrations
    await demonstrate_mcp_milvus_retriever()
    await demonstrate_mcp_redis_retriever()
    await demonstrate_mcp_retriever_composition()
    demonstrate_synchronous_usage()

    print("=" * 50)
    print("MCP Retriever demonstrations completed!")
    print()
    print("Next steps:")
    print("1. Set up MCP servers for Milvus and Redis")
    print("2. Configure server endpoints and authentication")
    print("3. Use MCP retrievers in your Sifaka chains")
    print("4. Monitor server health and performance")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
