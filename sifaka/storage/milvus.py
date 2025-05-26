"""Milvus storage implementation via MCP.

Vector storage with semantic search capabilities using Milvus via the Model Context
Protocol. Perfect for storing and searching thoughts, documents, and other text data.
"""

import asyncio
import json
from typing import Any, List, Optional

from sifaka.mcp import MCPClient, MCPServerConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MilvusStorage:
    """Milvus vector storage via MCP.

    Stores data in Milvus with vector embeddings for semantic search. Perfect for:
    - Semantic search over thoughts and documents
    - Vector similarity search
    - Large-scale text storage and retrieval
    - AI-powered search capabilities

    Attributes:
        mcp_client: MCP client for Milvus communication.
        collection_name: Name of the Milvus collection.
    """

    def __init__(self, mcp_config: MCPServerConfig, collection_name: str = "sifaka_storage"):
        """Initialize Milvus storage.

        Args:
            mcp_config: MCP server configuration for Milvus.
            collection_name: Name of the Milvus collection.
        """
        self.mcp_client = MCPClient(mcp_config)
        self.collection_name = collection_name
        self._connected = False

        logger.debug(f"Initialized MilvusStorage with collection '{collection_name}'")

    def _ensure_connected(self) -> None:
        """Ensure MCP client is connected and collection exists."""
        if not self._connected:
            try:
                # This would connect to the Milvus MCP server and ensure collection exists
                # For now, we'll assume connection succeeds
                self._connected = True
                logger.debug(f"Connected to Milvus MCP server, collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus MCP server: {e}")
                raise

    # Internal async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key from Milvus asynchronously (internal method).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        self._ensure_connected()

        try:
            logger.debug(f"Milvus get: {key}")

            # Call Milvus search for exact key match via MCP
            result = await self.mcp_client.call_tool(
                "search_nodes",
                {
                    "collection_name": self.collection_name,
                    "query": key,
                    "limit": 1,
                    "filter": f"key == '{key}'",
                },
            )

            if result and "nodes" in result and result["nodes"]:
                node = result["nodes"][0]
                return json.loads(node.get("content", "null"))

            return None

        except Exception as e:
            logger.error(f"Milvus get failed for key {key}: {e}")
            return None

    async def _set_async(self, key: str, value: Any) -> None:
        """Set a value for a key in Milvus asynchronously (internal method).

        Args:
            key: The storage key.
            value: The value to store.
        """
        self._ensure_connected()

        try:
            # Serialize the value
            serialized_value = json.dumps(value, default=str)
            logger.debug(f"Milvus set: {key}")

            # Call Milvus create_entities via MCP
            await self.mcp_client.call_tool(
                "create_entities",
                {
                    "collection_name": self.collection_name,
                    "entities": [
                        {
                            "key": key,
                            "content": serialized_value,
                            "text": str(value),  # For embedding generation
                        }
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Milvus set failed for key {key}: {e}")
            raise

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query using vector similarity asynchronously (internal method).

        This is where Milvus shines - semantic search over stored content.

        Args:
            query: The search query for semantic search.
            limit: Maximum number of results to return.

        Returns:
            List of matching values, ranked by semantic similarity.
        """
        self._ensure_connected()

        try:
            logger.debug(f"Milvus search: '{query}', limit {limit}")

            # Call Milvus vector search via MCP
            result = await self.mcp_client.call_tool(
                "search_nodes",
                {"collection_name": self.collection_name, "query": query, "limit": limit},
            )

            values = []
            if result and "nodes" in result:
                for node in result["nodes"]:
                    content = node.get("content")
                    if content:
                        try:
                            values.append(json.loads(content))
                        except json.JSONDecodeError:
                            values.append(content)

            return values

        except Exception as e:
            logger.error(f"Milvus search failed for query '{query}': {e}")
            return []

    async def _clear_async(self) -> None:
        """Clear all data from the Milvus collection asynchronously (internal method)."""
        self._ensure_connected()

        try:
            logger.debug(f"Milvus clear: collection '{self.collection_name}'")

            # Call Milvus delete_entities to clear all data via MCP
            await self.mcp_client.call_tool(
                "delete_entities",
                {
                    "collection_name": self.collection_name,
                    "filter": "key != ''",  # Delete all entities
                },
            )

        except Exception as e:
            logger.error(f"Milvus clear failed: {e}")
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
            logger.debug(f"Milvus delete: {key}")

            # Call Milvus delete_entities for specific key via MCP
            result = await self.mcp_client.call_tool(
                "delete_entities",
                {"collection_name": self.collection_name, "filter": f"key == '{key}'"},
            )
            return bool(result.get("deleted_count", 0) > 0)

        except Exception as e:
            logger.error(f"Milvus delete failed for key {key}: {e}")
            return False

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously (internal method).

        Returns:
            List of all storage keys.
        """
        self._ensure_connected()

        try:
            logger.debug(f"Milvus keys: collection '{self.collection_name}'")

            # Call Milvus search to get all entities and extract keys via MCP
            result = await self.mcp_client.call_tool(
                "search_nodes",
                {
                    "collection_name": self.collection_name,
                    "query": "",  # Empty query to get all
                    "limit": 10000,  # Large limit to get all keys
                },
            )

            keys = []
            if result and "nodes" in result:
                for node in result["nodes"]:
                    key = node.get("key")
                    if key:
                        keys.append(key)

            return keys

        except Exception as e:
            logger.error(f"Milvus keys failed: {e}")
            return []

    # Public sync methods (backward compatible API)
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key from Milvus.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return asyncio.run(self._get_async(key))

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key in Milvus.

        Args:
            key: The storage key.
            value: The value to store.
        """
        return asyncio.run(self._set_async(key, value))

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query using vector similarity.

        This is where Milvus shines - semantic search over stored content.

        Args:
            query: The search query for semantic search.
            limit: Maximum number of results to return.

        Returns:
            List of matching values, ranked by semantic similarity.
        """
        return asyncio.run(self._search_async(query, limit))

    def clear(self) -> None:
        """Clear all data from the Milvus collection."""
        return asyncio.run(self._clear_async())

    def delete(self, key: str) -> bool:
        """Delete a value by key from Milvus.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        return asyncio.run(self._delete_async(key))

    def keys(self) -> List[str]:
        """Get all keys in storage.

        Returns:
            List of all storage keys.
        """
        return asyncio.run(self._keys_async())

    def __len__(self) -> int:
        """Return number of stored items (simulated for Milvus)."""
        # In a real implementation, this would count entities in the collection
        return 0  # Placeholder

    def __contains__(self, key: str) -> bool:
        """Check if key exists in Milvus."""
        return self.get(key) is not None
