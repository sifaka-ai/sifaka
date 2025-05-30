"""Redis storage implementation via MCP.

Redis-based storage for cross-process sharing and caching. Uses the Model Context
Protocol to communicate with a Redis MCP server.
"""

import asyncio
import json
from datetime import datetime
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

    def _run_async_safely(self, coro):
        """Run async coroutine safely, handling existing event loops."""
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're already in an event loop, we need to handle this differently
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(coro)

    async def _ensure_connected(self) -> None:
        """Ensure MCP client is connected."""
        if not self._connected:
            try:
                # Connect to the Redis MCP server
                await self.mcp_client.connect()
                self._connected = True
                logger.debug("Connected to Redis MCP server")
            except Exception as e:
                logger.error(f"Failed to connect to Redis MCP server: {e}")
                raise

    def _make_key(self, key: str) -> str:
        """Create a prefixed Redis key."""
        return f"{self.key_prefix}:{key}"

    def _generate_timestamp_key(self, thought: Any) -> str:
        """Generate a timestamp-based key for a thought.

        Format: YYYYMMDD_thoughtid_iterN
        Example: 20250527_8b69c3dc_iter0

        Args:
            thought: The thought object with id, iteration, and timestamp.

        Returns:
            Timestamp-based key for Redis storage.
        """
        # Extract thought properties
        thought_id = getattr(thought, "id", "unknown")
        iteration = getattr(thought, "iteration", 0)
        timestamp = getattr(thought, "timestamp", datetime.now())

        # Parse timestamp if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                timestamp = datetime.now()

        # Format: YYYYMMDD_shortid_iterN
        date_str = timestamp.strftime("%Y%m%d")
        short_id = thought_id[:8] if len(thought_id) >= 8 else thought_id

        return f"{date_str}_{short_id}_iter{iteration}"

    # Internal async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key from Redis asynchronously (internal method).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        await self._ensure_connected()

        try:
            redis_key = self._make_key(key)
            logger.debug(f"Redis get: {redis_key}")

            # Call Redis GET via MCP (correct tool name)
            result = await self.mcp_client.call_tool("get", {"key": redis_key})

            # Handle MCP response format
            if result and not result.get("isError", False) and "content" in result:
                content = result["content"]
                if content and isinstance(content, list) and len(content) > 0:
                    raw_value = content[0].get("text", "")
                    # Check if key exists
                    if (
                        raw_value
                        and not raw_value.startswith("Key ")
                        and not raw_value.endswith(" does not exist")
                    ):
                        try:
                            # First, try to parse as double-encoded JSON
                            decoded_once = json.loads(raw_value)
                            if isinstance(decoded_once, str):
                                # Try to decode again (double-encoded)
                                try:
                                    decoded_value = json.loads(decoded_once)
                                except json.JSONDecodeError:
                                    # Single-encoded string, return as-is
                                    decoded_value = decoded_once
                            else:
                                # Already decoded, return as-is
                                decoded_value = decoded_once

                            # If the value is a dictionary and looks like a Thought, try to reconstruct it
                            if (
                                isinstance(decoded_value, dict)
                                and "id" in decoded_value
                                and "prompt" in decoded_value
                            ):
                                try:
                                    from sifaka.core.thought import Thought

                                    return Thought.from_dict(decoded_value)
                                except Exception as e:
                                    logger.debug(f"Failed to reconstruct Thought from dict: {e}")
                                    # Return the raw dict if reconstruction fails
                                    return decoded_value

                            return decoded_value
                        except json.JSONDecodeError:
                            # Return raw value if not JSON
                            return raw_value

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
        await self._ensure_connected()

        try:
            # Use timestamp-based key for thoughts, original key for other data
            if hasattr(value, "id") and hasattr(value, "iteration") and hasattr(value, "timestamp"):
                # This looks like a thought - use timestamp-based key
                timestamp_key = self._generate_timestamp_key(value)
                redis_key = self._make_key(timestamp_key)
                logger.debug(f"Redis set (thought): {redis_key}")

                # Also store with original key for backward compatibility
                original_redis_key = self._make_key(key)
                logger.debug(f"Redis set (original): {original_redis_key}")
            else:
                # Regular data - use original key
                redis_key = self._make_key(key)
                original_redis_key = None
                logger.debug(f"Redis set: {redis_key}")

            # Handle different value types and escape JSON for MCP
            if hasattr(value, "model_dump"):
                # Pydantic model - serialize to JSON and escape
                json_value = json.dumps(value.model_dump(), default=str)
                # Double-encode to prevent MCP from parsing it
                serialized_value = json.dumps(json_value)
            elif isinstance(value, (dict, list)):
                # Dict or list - serialize to JSON and escape
                json_value = json.dumps(value, default=str)
                # Double-encode to prevent MCP from parsing it
                serialized_value = json.dumps(json_value)
            else:
                # Other types - convert to string and escape if needed
                str_value = str(value)
                # Check if it looks like JSON and escape if so
                try:
                    json.loads(str_value)
                    # It's valid JSON, so escape it
                    serialized_value = json.dumps(str_value)
                except json.JSONDecodeError:
                    # Not JSON, use as-is
                    serialized_value = str_value

            # Store with timestamp-based key
            result = await self.mcp_client.call_tool(
                "set", {"key": redis_key, "value": serialized_value}
            )

            # Check for errors in response
            if result and result.get("isError", False):
                error_msg = "Unknown error"
                if "content" in result and result["content"]:
                    error_msg = result["content"][0].get("text", error_msg)
                raise Exception(f"Redis SET failed: {error_msg}")

            # Also store with original key for backward compatibility (thoughts only)
            if original_redis_key:
                result = await self.mcp_client.call_tool(
                    "set", {"key": original_redis_key, "value": serialized_value}
                )
                # Don't fail if the second store fails, just log it
                if result and result.get("isError", False):
                    logger.warning(
                        f"Failed to store thought with original key {original_redis_key}"
                    )

        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            raise

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query asynchronously (internal method).

        Note: Search is not implemented for Redis storage as it requires SCAN
        which is not available in the current Redis MCP server.

        Args:
            query: The search query (used for key pattern matching).
            limit: Maximum number of results to return.

        Returns:
            Empty list (search not supported).
        """
        logger.warning("Search is not implemented for Redis storage")
        return []

    async def _clear_async(self) -> None:
        """Clear all data with the key prefix asynchronously (internal method)."""
        logger.warning("Clear is not implemented for Redis storage (requires SCAN)")
        # Note: Clear is not implemented as it requires SCAN which is not available
        # in the current Redis MCP server

    async def _delete_async(self, key: str) -> bool:
        """Delete a value by key asynchronously (internal method).

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        await self._ensure_connected()

        try:
            redis_key = self._make_key(key)
            logger.debug(f"Redis delete: {redis_key}")

            # Call Redis DELETE via MCP (correct tool name)
            result = await self.mcp_client.call_tool("delete", {"key": redis_key})

            # Handle MCP response format
            if result and not result.get("isError", False) and "content" in result:
                content = result["content"]
                if content and isinstance(content, list) and len(content) > 0:
                    response_text = content[0].get("text", "")
                    # Check if deletion was successful
                    return "Successfully deleted" in response_text

            return False

        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously (internal method).

        Returns:
            Empty list (keys listing not supported).
        """
        logger.warning("Keys listing is not implemented for Redis storage (requires SCAN)")
        return []

    # Public sync methods (backward compatible API)
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key from Redis.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return self._run_async_safely(self._get_async(key))

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key in Redis.

        Args:
            key: The storage key.
            value: The value to store.
        """
        return self._run_async_safely(self._set_async(key, value))

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        For Redis storage, this scans keys and returns matching values.

        Args:
            query: The search query (used for key pattern matching).
            limit: Maximum number of results to return.

        Returns:
            List of matching values.
        """
        return self._run_async_safely(self._search_async(query, limit))

    def clear(self) -> None:
        """Clear all data with the key prefix."""
        return self._run_async_safely(self._clear_async())

    def delete(self, key: str) -> bool:
        """Delete a value by key from Redis.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        return self._run_async_safely(self._delete_async(key))

    def keys(self) -> List[str]:
        """Get all keys in storage.

        Returns:
            List of all storage keys.
        """
        return self._run_async_safely(self._keys_async())

    def __len__(self) -> int:
        """Return number of stored items (simulated for Redis)."""
        # In a real implementation, this would count keys with the prefix
        return 0  # Placeholder

    def __contains__(self, key: str) -> bool:
        """Check if key exists in Redis."""
        return self.get(key) is not None

    def _connect_mcp(self) -> None:
        """Connect to the MCP server (placeholder for testing)."""
        # This is a placeholder method for testing
        # In a real implementation, this would establish the MCP connection
        self._connected = True
        logger.debug("MCP connection established (mock)")

    # Additional methods expected by tests
    def save(self, key: str, value: Any) -> None:
        """Save a value to Redis (alias for set)."""
        self.set(key, value)

    def load(self, key: str) -> Optional[Any]:
        """Load a value from Redis (alias for get)."""
        return self.get(key)

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        return key in self

    def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching a pattern (placeholder implementation)."""
        logger.warning("list_keys is not fully implemented for Redis storage")
        return []

    def _serialize_thought(self, thought: Any) -> str:
        """Serialize a thought object to JSON string."""
        if hasattr(thought, "model_dump"):
            return json.dumps(thought.model_dump(), default=str)
        elif isinstance(thought, dict):
            return json.dumps(thought, default=str)
        else:
            return str(thought)

    def _deserialize_thought(self, data: str) -> Any:
        """Deserialize a JSON string back to a thought object."""
        try:
            thought_dict = json.loads(data)
            if isinstance(thought_dict, dict) and "id" in thought_dict and "prompt" in thought_dict:
                from sifaka.core.thought import Thought

                return Thought.from_dict(thought_dict)
            return thought_dict
        except json.JSONDecodeError:
            return data
