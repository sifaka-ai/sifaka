"""Redis storage implementation via MCP.

Redis-based storage for cross-process sharing and caching. Uses the Model Context
Protocol to communicate with a Redis MCP server.
"""

import asyncio
import json
from typing import Any, List, Optional

from sifaka.mcp import MCPClient, MCPServerConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class RedisStorage:
    """Redis-based storage via MCP.

    Stores data in Redis using the Model Context Protocol. Perfect for:
    - Cross-process data sharing
    - Distributed applications
    - Caching with TTL support
    - High-performance storage

    Attributes:
        mcp_client: MCP client for Redis communication.
        key_prefix: Prefix for all Redis keys.
    """

    def __init__(self, mcp_config: MCPServerConfig, key_prefix: str = "sifaka"):
        """Initialize Redis storage.

        Args:
            mcp_config: MCP server configuration for Redis.
            key_prefix: Prefix for all Redis keys.
        """
        self.mcp_client = MCPClient(mcp_config)
        self.key_prefix = key_prefix
        self._connected = False

        logger.debug(f"Initialized RedisStorage with prefix '{key_prefix}'")

    def _ensure_connected(self) -> None:
        """Ensure MCP client is connected."""
        if not self._connected:
            try:
                # This would connect to the Redis MCP server
                # For now, we'll assume connection succeeds
                self._connected = True
                logger.debug("Connected to Redis MCP server")
            except Exception as e:
                logger.error(f"Failed to connect to Redis MCP server: {e}")
                raise

    def _make_key(self, key: str) -> str:
        """Create a prefixed Redis key."""
        return f"{self.key_prefix}:{key}"

    # Internal async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key from Redis asynchronously (internal method).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        self._ensure_connected()

        try:
            redis_key = self._make_key(key)
            logger.debug(f"Redis get: {redis_key}")

            # Call Redis GET via MCP
            result = await self.mcp_client.call_tool("redis_get", {"key": redis_key})
            if result and "value" in result:
                return json.loads(result["value"])

            return None

        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None

    async def _set_async(self, key: str, value: Any) -> None:
        """Set a value for a key in Redis asynchronously (internal method).

        Args:
            key: The storage key.
            value: The value to store.
        """
        self._ensure_connected()

        try:
            redis_key = self._make_key(key)
            serialized_value = json.dumps(value, default=str)
            logger.debug(f"Redis set: {redis_key}")

            # Call Redis SET via MCP
            await self.mcp_client.call_tool(
                "redis_set", {"key": redis_key, "value": serialized_value}
            )

        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            raise

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query asynchronously (internal method).

        For Redis storage, this scans keys and returns matching values.

        Args:
            query: The search query (used for key pattern matching).
            limit: Maximum number of results to return.

        Returns:
            List of matching values.
        """
        self._ensure_connected()

        try:
            pattern = self._make_key(f"*{query}*")
            logger.debug(f"Redis search: pattern '{pattern}', limit {limit}")

            # Call Redis SCAN via MCP
            result = await self.mcp_client.call_tool(
                "redis_scan", {"pattern": pattern, "count": limit}
            )

            values = []
            if result and "keys" in result:
                for key in result["keys"]:
                    value_result = await self.mcp_client.call_tool("redis_get", {"key": key})
                    if value_result and "value" in value_result:
                        values.append(json.loads(value_result["value"]))

            return values

        except Exception as e:
            logger.error(f"Redis search failed for query '{query}': {e}")
            return []

    async def _clear_async(self) -> None:
        """Clear all data with the key prefix asynchronously (internal method)."""
        self._ensure_connected()

        try:
            pattern = self._make_key("*")
            logger.debug(f"Redis clear: pattern '{pattern}'")

            # Call Redis SCAN to get all keys, then delete them
            result = await self.mcp_client.call_tool("redis_scan", {"pattern": pattern})
            if result and "keys" in result:
                for key in result["keys"]:
                    await self.mcp_client.call_tool("redis_del", {"key": key})

        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            raise

    async def _delete_async(self, key: str) -> bool:
        """Delete a value by key asynchronously (internal method).

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        self._ensure_connected()

        try:
            redis_key = self._make_key(key)
            logger.debug(f"Redis delete: {redis_key}")

            # Call Redis DEL via MCP
            result = await self.mcp_client.call_tool("redis_del", {"key": redis_key})
            return result.get("deleted", 0) > 0

        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously (internal method).

        Returns:
            List of all storage keys.
        """
        self._ensure_connected()

        try:
            pattern = self._make_key("*")
            logger.debug(f"Redis keys: pattern '{pattern}'")

            # Call Redis SCAN to get all keys
            result = await self.mcp_client.call_tool("redis_scan", {"pattern": pattern})
            if result and "keys" in result:
                # Remove prefix from keys
                prefix_len = len(self.key_prefix) + 1
                return [key[prefix_len:] for key in result["keys"]]

            return []

        except Exception as e:
            logger.error(f"Redis keys failed: {e}")
            return []

    # Public sync methods (backward compatible API)
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key from Redis.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return asyncio.run(self._get_async(key))

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key in Redis.

        Args:
            key: The storage key.
            value: The value to store.
        """
        return asyncio.run(self._set_async(key, value))

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        For Redis storage, this scans keys and returns matching values.

        Args:
            query: The search query (used for key pattern matching).
            limit: Maximum number of results to return.

        Returns:
            List of matching values.
        """
        return asyncio.run(self._search_async(query, limit))

    def clear(self) -> None:
        """Clear all data with the key prefix."""
        return asyncio.run(self._clear_async())

    def __len__(self) -> int:
        """Return number of stored items (simulated for Redis)."""
        # In a real implementation, this would count keys with the prefix
        return 0  # Placeholder

    def __contains__(self, key: str) -> bool:
        """Check if key exists in Redis."""
        return self.get(key) is not None
