"""Redis persistence implementation for Sifaka using MCP.

This module provides Redis-based storage via MCP server integration,
offering production-ready persistence with cross-process sharing.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic_ai.mcp import MCPServerStdio

from .base import SifakaBasePersistence
from sifaka.utils import get_logger

logger = get_logger(__name__)


class RedisPersistence(SifakaBasePersistence):
    """Redis-based persistence via MCP server.
    
    This implementation uses the Redis MCP server for production-ready
    storage with features like:
    
    - Cross-process data sharing
    - Configurable TTL for automatic cleanup
    - Redis-specific optimizations (key patterns, indexing)
    - JSON-based storage for complex objects
    - Atomic operations
    
    Attributes:
        mcp_server: The Redis MCP server instance
        ttl_seconds: Time-to-live for stored data (None for no expiration)
    """
    
    def __init__(
        self, 
        mcp_server: MCPServerStdio,
        key_prefix: str = "sifaka",
        ttl_seconds: Optional[int] = None
    ):
        """Initialize Redis persistence.
        
        Args:
            mcp_server: Redis MCP server instance
            key_prefix: Prefix for all Redis keys
            ttl_seconds: Time-to-live for stored data (None for no expiration)
        """
        super().__init__(key_prefix)
        self.mcp_server = mcp_server
        self.ttl_seconds = ttl_seconds
        logger.debug(f"Initialized RedisPersistence with TTL: {ttl_seconds}")
    
    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data at the given key using Redis JSON.
        
        Args:
            key: Storage key
            data: Raw data to store (JSON string)
        """
        try:
            # Parse JSON data for Redis JSON storage
            json_data = json.loads(data)
            
            async with self.mcp_server as mcp_client:
                # Use json_set tool for complex objects
                result = await mcp_client.call_tool(
                    "json_set", 
                    arguments={
                        "name": key, 
                        "path": "$", 
                        "value": json_data
                    }
                )
                
                # Set TTL if configured
                if self.ttl_seconds:
                    await mcp_client.call_tool(
                        "expire",
                        arguments={
                            "name": key,
                            "seconds": self.ttl_seconds
                        }
                    )
                
                logger.debug(f"Redis stored key: {key} (TTL: {self.ttl_seconds})")
                
        except Exception as e:
            logger.error(f"Failed to store Redis key {key}: {e}")
            raise
    
    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data from the given key using Redis JSON.
        
        Args:
            key: Storage key
            
        Returns:
            Raw data if found, None otherwise
        """
        try:
            async with self.mcp_server as mcp_client:
                # Use json_get tool for complex objects
                result = await mcp_client.call_tool(
                    "json_get", 
                    arguments={
                        "name": key, 
                        "path": "$"
                    }
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
                
                # Parse and re-serialize to ensure consistent format
                try:
                    json_data = json.loads(data) if isinstance(data, str) else data
                    serialized = json.dumps(json_data)
                    logger.debug(f"Redis retrieved key: {key}")
                    return serialized
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid JSON data for key {key}: {data}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Failed to retrieve Redis key {key}: {e}")
            return None
    
    async def _delete_raw(self, key: str) -> bool:
        """Delete data at the given key using Redis JSON.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted, False if key didn't exist
        """
        try:
            async with self.mcp_server as mcp_client:
                # Use json_del tool
                result = await mcp_client.call_tool(
                    "json_del", 
                    arguments={
                        "name": key, 
                        "path": "$"
                    }
                )
                
                logger.debug(f"Redis deleted key: {key}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to delete Redis key {key}: {e}")
            return False
    
    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys matching the given pattern.
        
        Args:
            pattern: Key pattern to match (Redis pattern syntax)
            
        Returns:
            List of matching keys
        """
        try:
            async with self.mcp_server as mcp_client:
                # Use keys command to list matching keys
                result = await mcp_client.call_tool(
                    "keys",
                    arguments={"pattern": pattern}
                )
                
                if not result or not hasattr(result, "content"):
                    return []
                
                # Parse the result to get key list
                keys_data = str(result.content[0].text) if result.content else "[]"
                
                try:
                    keys = json.loads(keys_data) if isinstance(keys_data, str) else keys_data
                    if isinstance(keys, list):
                        logger.debug(f"Redis listed {len(keys)} keys matching pattern: {pattern}")
                        return keys
                    else:
                        return []
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid keys data: {keys_data}")
                    return []
                    
        except Exception as e:
            logger.warning(f"Failed to list Redis keys with pattern {pattern}: {e}")
            return []
    
    async def clear_namespace(self) -> None:
        """Clear all data in this persistence namespace.
        
        This removes all keys with the configured prefix.
        """
        try:
            pattern = f"{self.key_prefix}:*"
            keys = await self._list_keys(pattern)
            
            if not keys:
                logger.debug("No keys to clear in Redis namespace")
                return
            
            async with self.mcp_server as mcp_client:
                # Delete all keys in batches
                for key in keys:
                    await self._delete_raw(key)
            
            logger.debug(f"Cleared {len(keys)} keys from Redis namespace")
            
        except Exception as e:
            logger.error(f"Failed to clear Redis namespace: {e}")
            raise
    
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information.
        
        Returns:
            Dictionary with Redis server info
        """
        try:
            async with self.mcp_server as mcp_client:
                result = await mcp_client.call_tool("info", arguments={})
                
                if result and hasattr(result, "content"):
                    info_data = str(result.content[0].text) if result.content else "{}"
                    try:
                        return json.loads(info_data) if isinstance(info_data, str) else info_data
                    except (json.JSONDecodeError, TypeError):
                        return {"raw_info": info_data}
                
                return {}
                
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {e}")
            return {"error": str(e)}
    
    # PydanticAI BaseStatePersistence interface implementation
    async def snapshot_node(self, state: "SifakaThought", next_node: str) -> None:
        """Snapshot the current state before executing a node.
        
        Args:
            state: Current thought state
            next_node: Name of the next node to execute
        """
        try:
            # Store the thought with a snapshot key
            snapshot_key = f"{self.key_prefix}:snapshot:{state.id}:{next_node}"
            data = await self.serialize_state(state)
            await self._store_raw(snapshot_key, data)
            
            # Also store as regular thought
            await self.store_thought(state)
            
            logger.debug(f"Redis snapshotted state for thought {state.id} before node {next_node}")
            
        except Exception as e:
            logger.error(f"Failed to snapshot state for thought {state.id}: {e}")
            raise
    
    async def load_state(self, state_id: str) -> Optional["SifakaThought"]:
        """Load a previously saved state.
        
        Args:
            state_id: The state ID to load
            
        Returns:
            The loaded state if found, None otherwise
        """
        return await self.retrieve_thought(state_id)
