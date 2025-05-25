"""Base storage classes for unified 3-tier architecture.

This module provides the foundational storage abstractions that implement
the unified memory → cache → persistence pattern across all Sifaka components.
"""

import asyncio
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sifaka.mcp import MCPServerConfig, MCPClient, MCPRequest
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class StorageError(Exception):
    """Base exception for storage operations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        storage_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.operation = operation
        self.storage_type = storage_type
        self.metadata = metadata or {}

        # Build full error message
        full_message = message
        if storage_type:
            full_message = f"[{storage_type}] {full_message}"
        if operation:
            full_message = f"{full_message} (during {operation})"

        super().__init__(full_message)


class InMemoryStorage:
    """L1: Fast in-memory storage with LRU eviction.

    Provides microsecond-level access to frequently used data with automatic
    LRU eviction to prevent memory bloat. This is the fastest tier in the
    3-tier storage architecture.

    Attributes:
        max_size: Maximum number of items to store before eviction.
        data: Ordered dictionary for LRU tracking.
        access_times: Track access patterns for debugging.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize in-memory storage.

        Args:
            max_size: Maximum number of items before LRU eviction.
        """
        self.max_size = max_size
        self.data: OrderedDict[str, Any] = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0

        logger.debug(f"Initialized InMemoryStorage with max_size={max_size}")

    def get(self, key: str) -> Optional[Any]:
        """Get item from memory with LRU tracking.

        Args:
            key: Storage key.

        Returns:
            Stored value or None if not found.
        """
        if key in self.data:
            # Move to end (most recently used)
            value = self.data.pop(key)
            self.data[key] = value
            self.access_times[key] = time.time()
            self.hits += 1
            logger.debug(f"Memory hit for key: {key[:50]}...")
            return value

        self.misses += 1
        logger.debug(f"Memory miss for key: {key[:50]}...")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set item in memory with LRU eviction.

        Args:
            key: Storage key.
            value: Value to store.
        """
        # Remove if already exists (to update position)
        if key in self.data:
            del self.data[key]

        # Add to end (most recently used)
        self.data[key] = value
        self.access_times[key] = time.time()

        # Evict oldest if over capacity
        while len(self.data) > self.max_size:
            oldest_key = next(iter(self.data))
            del self.data[oldest_key]
            del self.access_times[oldest_key]
            logger.debug(f"Evicted oldest key from memory: {oldest_key[:50]}...")

        logger.debug(f"Stored in memory: {key[:50]}... (size: {len(self.data)})")

    def clear(self) -> None:
        """Clear all data from memory."""
        count = len(self.data)
        self.data.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0
        logger.debug(f"Cleared {count} items from memory")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "type": "memory",
            "size": len(self.data),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "utilization": len(self.data) / self.max_size,
        }


class RedisCache:
    """L2: Redis caching layer via MCP.

    Provides fast network-based caching with TTL expiration and cross-process
    sharing. This is the middle tier in the 3-tier storage architecture.

    Attributes:
        mcp_client: MCP client for Redis communication.
        ttl: Default time-to-live for cached items.
        key_prefix: Prefix for all Redis keys.
    """

    def __init__(self, mcp_config: MCPServerConfig, ttl: int = 3600, key_prefix: str = "sifaka"):
        """Initialize Redis cache.

        Args:
            mcp_config: MCP server configuration for Redis.
            ttl: Default TTL in seconds.
            key_prefix: Prefix for Redis keys.
        """
        self.mcp_client = MCPClient(mcp_config)
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.hits = 0
        self.misses = 0
        self._connected = False

        logger.debug(f"Initialized RedisCache with TTL={ttl}, prefix={key_prefix}")

    async def _ensure_connected(self) -> None:
        """Ensure MCP client is connected."""
        if not self._connected:
            await self.mcp_client.connect()
            self._connected = True

    def _run_async(self, coro: Any) -> Any:
        """Run async operation in event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task if loop is already running
                task = asyncio.create_task(coro)
                return task
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)

    async def _get_async(self, key: str) -> Optional[Any]:
        """Async get from Redis."""
        await self._ensure_connected()

        redis_key = f"{self.key_prefix}:{key}"
        request = MCPRequest(
            method="tools/call", params={"name": "get", "arguments": {"key": redis_key}}
        )

        try:
            response = await self.mcp_client.transport.send_request(request)
            if response.error:
                logger.warning(f"Redis get error: {response.error}")
                return None

            # Parse response content
            if hasattr(response, "result") and response.result:
                content = response.result.get("content", [])
                if content and len(content) > 0:
                    text_content = content[0].get("text", "")
                    if text_content and text_content != "null":
                        return pickle.loads(text_content.encode("latin1"))

            return None

        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            return None

    def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache.

        Args:
            key: Storage key.

        Returns:
            Cached value or None if not found.
        """
        result = self._run_async(self._get_async(key))

        if result is not None:
            self.hits += 1
            logger.debug(f"Redis hit for key: {key[:50]}...")
            return result

        self.misses += 1
        logger.debug(f"Redis miss for key: {key[:50]}...")
        return None

    async def _set_async(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Async set to Redis."""
        await self._ensure_connected()

        redis_key = f"{self.key_prefix}:{key}"
        serialized_value = pickle.dumps(value).decode("latin1")

        request = MCPRequest(
            method="tools/call",
            params={
                "name": "set",
                "arguments": {
                    "key": redis_key,
                    "value": serialized_value,
                    "expireSeconds": ttl or self.ttl,
                },
            },
        )

        try:
            response = await self.mcp_client.transport.send_request(request)
            if response.error:
                logger.warning(f"Redis set error: {response.error}")
            else:
                logger.debug(f"Stored in Redis: {key[:50]}... (TTL: {ttl or self.ttl}s)")
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in Redis cache.

        Args:
            key: Storage key.
            value: Value to cache.
            ttl: Optional TTL override.
        """
        # Run async operation
        self._run_async(self._set_async(key, value, ttl))

    def clear(self, pattern: str = "*") -> None:
        """Clear items matching pattern from Redis.

        Args:
            pattern: Key pattern to match (default: all keys with prefix).
        """
        # Implementation would use Redis SCAN + DEL
        # For now, just log the operation
        logger.debug(f"Redis clear requested for pattern: {self.key_prefix}:{pattern}")

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "type": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl": self.ttl,
            "key_prefix": self.key_prefix,
        }


class MilvusStorage:
    """L3: Vector-based persistent storage via MCP.

    Provides persistent storage with semantic search capabilities using vector
    embeddings. This is the slowest but most capable tier in the 3-tier
    storage architecture.

    Attributes:
        mcp_client: MCP client for Milvus communication.
        collection_name: Name of the Milvus collection.
        embedding_function: Function to generate embeddings.
    """

    def __init__(
        self, mcp_config: MCPServerConfig, collection_name: str, embedding_function: Any = None
    ):
        """Initialize Milvus storage.

        Args:
            mcp_config: MCP server configuration for Milvus.
            collection_name: Name of the collection to use.
            embedding_function: Function to generate embeddings.
        """
        self.mcp_client = MCPClient(mcp_config)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or self._default_embedding
        self._connected = False

        logger.debug(f"Initialized MilvusStorage for collection: {collection_name}")

    def _default_embedding(self, text: str) -> List[float]:
        """Default embedding function (simple hash-based for testing)."""
        # In production, this would use a real embedding model
        import hashlib

        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        # Convert to 128-dimensional vector
        return [
            float(int(hash_hex[i : i + 2], 16)) / 255.0 for i in range(0, min(len(hash_hex), 32), 2)
        ]

    async def _ensure_connected(self) -> None:
        """Ensure MCP client is connected."""
        if not self._connected:
            await self.mcp_client.connect()
            self._connected = True

    def _run_async(self, coro: Any) -> Any:
        """Run async operation in event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.create_task(coro)
                return task
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    async def _get_async(self, key: str) -> Optional[Any]:
        """Async get from Milvus by key."""
        await self._ensure_connected()

        # Use Milvus query to find by key
        request = MCPRequest(
            method="tools/call",
            params={
                "name": "milvus_query",
                "arguments": {
                    "collection_name": self.collection_name,
                    "filter": f"key == '{key}'",
                    "output_fields": ["key", "data"],
                    "limit": 1,
                },
            },
        )

        try:
            response = await self.mcp_client.transport.send_request(request)
            if response.error:
                logger.warning(f"Milvus get error: {response.error}")
                return None

            # Parse response
            if hasattr(response, "result") and response.result:
                content = response.result.get("content", [])
                if content and len(content) > 0:
                    text_content = content[0].get("text", "")
                    if text_content and "data" in text_content:
                        # Extract data field from query result
                        import re

                        match = re.search(r'"data":\s*"([^"]*)"', text_content)
                        if match:
                            return pickle.loads(match.group(1).encode("latin1"))

            return None

        except Exception as e:
            logger.warning(f"Milvus get failed for key {key}: {e}")
            return None

    def get(self, key: str) -> Optional[Any]:
        """Get item from Milvus by key.

        Args:
            key: Storage key.

        Returns:
            Stored value or None if not found.
        """
        result = self._run_async(self._get_async(key))
        if result is not None:
            logger.debug(f"Milvus hit for key: {key[:50]}...")
        else:
            logger.debug(f"Milvus miss for key: {key[:50]}...")
        return result

    async def _set_async(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Async set to Milvus."""
        await self._ensure_connected()

        # Generate embedding for the value
        text_repr = str(value)
        if hasattr(value, "prompt"):
            text_repr = f"{value.prompt}\n{getattr(value, 'text', '')}"

        vector = self.embedding_function(text_repr)
        serialized_value = pickle.dumps(value).decode("latin1")

        # Prepare data for insertion
        data = {
            "key": [key],
            "vector": vector,
            "data": [serialized_value],
            "text": [text_repr[:1000]],  # Truncated text for search
            **(metadata or {}),
        }

        request = MCPRequest(
            method="tools/call",
            params={
                "name": "milvus_insert_data",
                "arguments": {"collection_name": self.collection_name, "data": data},
            },
        )

        try:
            response = await self.mcp_client.transport.send_request(request)
            if response.error:
                logger.warning(f"Milvus set error: {response.error}")
            else:
                logger.debug(f"Stored in Milvus: {key[:50]}...")
        except Exception as e:
            logger.warning(f"Milvus set failed for key {key}: {e}")

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set item in Milvus storage.

        Args:
            key: Storage key.
            value: Value to store.
            metadata: Optional metadata fields.
        """
        self._run_async(self._set_async(key, value, metadata))

    async def _search_similar_async(self, query: str, limit: int = 5) -> List[Any]:
        """Async search for similar items."""
        await self._ensure_connected()

        # Generate query vector
        query_vector = self.embedding_function(query)

        request = MCPRequest(
            method="tools/call",
            params={
                "name": "milvus_vector_search",
                "arguments": {
                    "collection_name": self.collection_name,
                    "vector": query_vector,
                    "vector_field": "vector",
                    "limit": limit,
                    "output_fields": ["key", "data", "text"],
                    "metric_type": "COSINE",
                },
            },
        )

        try:
            response = await self.mcp_client.transport.send_request(request)
            if response.error:
                logger.warning(f"Milvus search error: {response.error}")
                return []

            # Parse search results
            results = []
            if hasattr(response, "result") and response.result:
                content = response.result.get("content", [])
                if content and len(content) > 0:
                    text_content = content[0].get("text", "")
                    # Parse multiple results from text content
                    # This is a simplified parser - in production would be more robust
                    import re

                    data_matches = re.findall(r'"data":\s*"([^"]*)"', text_content)
                    for data_match in data_matches[:limit]:
                        try:
                            value = pickle.loads(data_match.encode("latin1"))
                            results.append(value)
                        except Exception as e:
                            logger.warning(f"Failed to deserialize search result: {e}")

            return results

        except Exception as e:
            logger.warning(f"Milvus search failed for query {query}: {e}")
            return []

    def search_similar(self, query: str, limit: int = 5) -> List[Any]:
        """Search for similar items using vector similarity.

        Args:
            query: Query text for similarity search.
            limit: Maximum number of results.

        Returns:
            List of similar items.
        """
        results = self._run_async(self._search_similar_async(query, limit))
        if isinstance(results, list):
            logger.debug(
                f"Milvus search returned {len(results)} results for query: {query[:50]}..."
            )
            return results
        else:
            logger.debug(f"Milvus search returned non-list result for query: {query[:50]}...")
            return []

    def clear(self) -> None:
        """Clear all data from Milvus collection."""
        # Implementation would drop and recreate collection
        logger.debug(f"Milvus clear requested for collection: {self.collection_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get Milvus storage statistics."""
        return {
            "type": "milvus",
            "collection_name": self.collection_name,
            "connected": self._connected,
        }


class CachedStorage:
    """Unified 3-tier storage combining all layers.

    Implements the complete memory → cache → persistence pattern with
    automatic tier management and async persistence.

    Attributes:
        memory: L1 in-memory storage.
        cache: L2 Redis cache.
        persistence: L3 Milvus storage.
    """

    def __init__(self, memory: InMemoryStorage, cache: RedisCache, persistence: MilvusStorage):
        """Initialize cached storage.

        Args:
            memory: In-memory storage instance.
            cache: Redis cache instance.
            persistence: Milvus storage instance.
        """
        self.memory = memory
        self.cache = cache
        self.persistence = persistence

        logger.debug("Initialized CachedStorage with 3-tier architecture")

    def get(self, key: str) -> Optional[Any]:
        """Get item using 3-tier lookup: L1 → L2 → L3.

        Args:
            key: Storage key.

        Returns:
            Stored value or None if not found.
        """
        # L1: Check memory first (fastest)
        value = self.memory.get(key)
        if value is not None:
            return value

        # L2: Check Redis cache
        value = self.cache.get(key)
        if value is not None:
            # Cache in memory for next time
            self.memory.set(key, value)
            return value

        # L3: Check persistent storage
        value = self.persistence.get(key)
        if value is not None:
            # Cache in both faster tiers
            self.memory.set(key, value)
            # Async cache in Redis (don't block)
            asyncio.create_task(self.cache._set_async(key, value))
            return value

        return None

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set item in all tiers: immediate L1, async L2+L3.

        Args:
            key: Storage key.
            value: Value to store.
            metadata: Optional metadata for L3.
        """
        # L1: Immediate save to memory
        self.memory.set(key, value)

        # L2+L3: Async save to cache and persistence (don't block)
        asyncio.create_task(self._async_persist(key, value, metadata))

    async def _async_persist(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Async persistence to L2 and L3."""
        try:
            # Save to both cache and persistence concurrently
            await asyncio.gather(
                self.cache._set_async(key, value),
                self.persistence._set_async(key, value, metadata),
                return_exceptions=True,
            )
        except Exception as e:
            logger.warning(f"Async persistence failed for key {key}: {e}")

    def search_similar(self, query: str, limit: int = 5) -> List[Any]:
        """Search for similar items using L3 vector search.

        Args:
            query: Query text for similarity search.
            limit: Maximum number of results.

        Returns:
            List of similar items.
        """
        # Use L3 for semantic search
        results = self.persistence.search_similar(query, limit)

        # Cache results in faster tiers
        for result in results:
            if hasattr(result, "id"):
                key = f"search_result:{result.id}"
                self.memory.set(key, result)

        return results

    def clear(self) -> None:
        """Clear all tiers."""
        self.memory.clear()
        self.cache.clear()
        self.persistence.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all tiers."""
        return {
            "memory": self.memory.get_stats(),
            "cache": self.cache.get_stats(),
            "persistence": self.persistence.get_stats(),
        }
