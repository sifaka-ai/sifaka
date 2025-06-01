"""Simple Redis storage implementation via MCP.

Redis-based storage for cross-process sharing and caching. Uses PydanticAI's
native MCP client to communicate with the MCP Redis server.
"""

import json
from typing import Any, List, Optional

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class RedisStorage:
    """Simple Redis-based storage via MCP.

    Stores data directly in Redis using MCP tools for persistence and cross-process sharing.
    """

    def __init__(self, redis_mcp_server: Optional[Any] = None, key_prefix: str = "sifaka"):
        """Initialize Redis storage.

        Args:
            redis_mcp_server: PydanticAI MCPServerStdio instance for Redis (optional).
            key_prefix: Prefix for all Redis keys.
        """
        self.redis_mcp_server = redis_mcp_server
        self.key_prefix = key_prefix
        self._redis_enabled = redis_mcp_server is not None

        logger.debug(
            f"Initialized RedisStorage with prefix '{key_prefix}', Redis enabled: {self._redis_enabled}"
        )

    def _make_key(self, key: str) -> str:
        """Create a prefixed Redis key."""
        return f"{self.key_prefix}:{key}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value for Redis storage."""
        if hasattr(value, "model_dump"):
            # Pydantic model - serialize to JSON
            return json.dumps(value.model_dump(), default=str)
        elif isinstance(value, (dict, list)):
            # Dict or list - serialize to JSON
            return json.dumps(value, default=str)
        else:
            # Other types - convert to string
            return str(value)

    def _deserialize_value(self, data: str) -> Any:
        """Deserialize a value from Redis storage."""
        if not data or "does not exist" in data:
            return None

        # Try to parse as JSON if it looks like structured data
        if data.startswith("{") or data.startswith("[") or data.startswith('"'):
            try:
                decoded_value = json.loads(data)

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
                        return decoded_value

                return decoded_value
            except json.JSONDecodeError:
                return data

        return data

    # Internal async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key from Redis asynchronously."""
        if not self._redis_enabled:
            return None

        try:
            redis_key = self._make_key(key)

            # Use MCP server directly with JSON tools
            async with self.redis_mcp_server as mcp_client:
                # Call json_get tool for complex objects
                result = await mcp_client.call_tool(
                    "json_get", arguments={"name": redis_key, "path": "$"}
                )

                if not result or not hasattr(result, "content"):
                    return None

                # Extract the JSON value from the result
                data = str(result.content[0].text) if result.content else None

                if not data:
                    return None

                # Check if key doesn't exist
                if "No data found" in data or "does not exist" in data:
                    return None

                # For JSON tools, the data is already deserialized
                try:
                    import json

                    json_data = json.loads(data) if isinstance(data, str) else data

                    # If it's a Thought object, try to reconstruct it
                    if isinstance(json_data, dict) and "id" in json_data and "prompt" in json_data:
                        try:
                            from sifaka.core.thought import Thought

                            return Thought.from_dict(json_data)
                        except Exception as e:
                            logger.debug(f"Failed to reconstruct Thought from dict: {e}")
                            return json_data

                    return json_data
                except (json.JSONDecodeError, TypeError):
                    return data

        except Exception as e:
            logger.warning(f"Failed to get from Redis for key {key}: {e}")
            return None

    async def _set_async(self, key: str, value: Any) -> None:
        """Set a value for a key in Redis asynchronously."""
        print(f"🔥 REDIS _set_async called with key={key}, value type={type(value)}")

        if not self._redis_enabled:
            print(f"🔥 REDIS not enabled, returning")
            return

        try:
            redis_key = self._make_key(key)

            logger.debug(f"Setting Redis JSON key: {redis_key}")
            logger.debug(f"Original value type: {type(value)}")

            # Convert value to JSON-compatible format
            if hasattr(value, "model_dump"):
                # Pydantic model - use model_dump for JSON compatibility
                json_value = value.model_dump()
            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                # Already JSON-compatible
                json_value = value
            else:
                # Convert to dict representation
                json_value = {"data": str(value), "type": type(value).__name__}

            logger.debug(f"JSON value type: {type(json_value)}")

            # Use MCP server directly with JSON tools
            async with self.redis_mcp_server as mcp_client:
                # Call json_set tool for complex objects
                result = await mcp_client.call_tool(
                    "json_set", arguments={"name": redis_key, "path": "$", "value": json_value}
                )
                logger.debug(f"Redis JSON set result for {key}: {result}")

        except Exception as e:
            logger.warning(f"Failed to set Redis key {key}: {e}")
            raise

    # Public sync methods (required by Storage protocol)
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key (sync wrapper)."""
        import asyncio

        try:
            return asyncio.run(self._get_async(key))
        except RuntimeError:
            # Already in an event loop, use run_until_complete
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._get_async(key))

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key (sync wrapper)."""
        import asyncio

        try:
            asyncio.run(self._set_async(key, value))
        except RuntimeError:
            # Already in an event loop, use run_until_complete
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._set_async(key, value))

    def __contains__(self, key: str) -> bool:
        """Check if key exists (sync)."""
        result = self.get(key)
        return result is not None

    def __len__(self) -> int:
        """Return number of stored items (placeholder)."""
        return 0  # Redis doesn't easily support counting all keys

    def __bool__(self) -> bool:
        """Return True if the storage is available/configured."""
        return self._redis_enabled

    # Async methods
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        result = await self._get_async(key)
        return result is not None

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self._redis_enabled:
            return False

        try:
            redis_key = self._make_key(key)

            # Use MCP server directly with JSON tools
            async with self.redis_mcp_server as mcp_client:
                # Call json_del tool
                result = await mcp_client.call_tool(
                    "json_del", arguments={"name": redis_key, "path": "$"}
                )
                logger.debug(f"Redis JSON delete result for {key}: {result}")
                return True

        except Exception as e:
            logger.warning(f"Failed to delete Redis key {key}: {e}")
            return False

    # Placeholder methods for compatibility
    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search not implemented for Redis storage."""
        logger.warning("Search is not implemented for Redis storage")
        return []

    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys not fully implemented for Redis storage."""
        logger.warning("list_keys is not fully implemented for Redis storage")
        return []
