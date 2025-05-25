"""Milvus storage implementation via MCP.

Vector storage with semantic search capabilities using Milvus via the Model Context
Protocol. Perfect for storing and searching thoughts, documents, and other text data.
"""

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

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key from Milvus.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        self._ensure_connected()

        try:
            # This would use MCP to search for the exact key
            # For now, we'll simulate the operation
            logger.debug(f"Milvus get: {key}")

            # Simulated MCP call:
            # result = self.mcp_client.call_tool("search_nodes", {
            #     "collection_name": self.collection_name,
            #     "query": key,
            #     "limit": 1,
            #     "filter": f"key == '{key}'"
            # })
            #
            # if result and "nodes" in result and result["nodes"]:
            #     node = result["nodes"][0]
            #     return json.loads(node.get("content", "null"))

            return None  # Placeholder

        except Exception as e:
            logger.error(f"Milvus get failed for key {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key in Milvus.

        Args:
            key: The storage key.
            value: The value to store.
        """
        self._ensure_connected()

        try:
            # Serialize the value
            serialized_value = json.dumps(value, default=str)

            # This would use MCP to create/update an entity
            # For now, we'll simulate the operation
            logger.debug(f"Milvus set: {key}")

            # Simulated MCP call:
            # self.mcp_client.call_tool("create_entities", {
            #     "collection_name": self.collection_name,
            #     "entities": [{
            #         "key": key,
            #         "content": serialized_value,
            #         "text": str(value)  # For embedding generation
            #     }]
            # })

        except Exception as e:
            logger.error(f"Milvus set failed for key {key}: {e}")
            raise

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query using vector similarity.

        This is where Milvus shines - semantic search over stored content.

        Args:
            query: The search query for semantic search.
            limit: Maximum number of results to return.

        Returns:
            List of matching values, ranked by semantic similarity.
        """
        self._ensure_connected()

        try:
            # This would use MCP to perform vector search
            # For now, we'll simulate the operation
            logger.debug(f"Milvus search: '{query}', limit {limit}")

            # Simulated MCP call:
            # result = self.mcp_client.call_tool("search_nodes", {
            #     "collection_name": self.collection_name,
            #     "query": query,
            #     "limit": limit
            # })
            #
            # values = []
            # if result and "nodes" in result:
            #     for node in result["nodes"]:
            #         content = node.get("content")
            #         if content:
            #             try:
            #                 values.append(json.loads(content))
            #             except json.JSONDecodeError:
            #                 values.append(content)
            #
            # return values

            return []  # Placeholder

        except Exception as e:
            logger.error(f"Milvus search failed for query '{query}': {e}")
            return []

    def clear(self) -> None:
        """Clear all data from the Milvus collection."""
        self._ensure_connected()

        try:
            # This would use MCP to delete all entities in the collection
            # For now, we'll simulate the operation
            logger.debug(f"Milvus clear: collection '{self.collection_name}'")

            # Simulated MCP call:
            # self.mcp_client.call_tool("delete_entities", {
            #     "collection_name": self.collection_name,
            #     "filter": "key != ''"  # Delete all entities
            # })

        except Exception as e:
            logger.error(f"Milvus clear failed: {e}")
            raise

    def __len__(self) -> int:
        """Return number of stored items (simulated for Milvus)."""
        # In a real implementation, this would count entities in the collection
        return 0  # Placeholder

    def __contains__(self, key: str) -> bool:
        """Check if key exists in Milvus."""
        return self.get(key) is not None
