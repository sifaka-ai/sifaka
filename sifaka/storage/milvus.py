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

        # Hybrid approach: Store metadata separately due to MCP validation limitations
        # In production, this would be a separate database or Redis
        self._metadata_store = {}

        logger.debug(f"Initialized MilvusStorage with collection '{collection_name}'")

    async def _ensure_connected(self) -> None:
        """Ensure MCP client is connected and collection exists."""
        if not self._connected:
            try:
                # Connect to the Milvus MCP server
                await self.mcp_client.connect()

                # Check if collection exists, create if it doesn't
                await self._ensure_collection_exists()

                self._connected = True
                logger.debug(f"Connected to Milvus MCP server, collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus MCP server: {e}")
                raise

    async def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create it if it doesn't."""
        try:
            # List collections to check if ours exists
            result = await self.mcp_client.call_tool("milvus_list_collections", {})

            # Parse the response to check if our collection exists
            if result and "content" in result and result["content"]:
                collections_text = result["content"][0].get("text", "")
                if self.collection_name not in collections_text:
                    # Collection doesn't exist, create it
                    await self._create_collection()

        except Exception as e:
            logger.warning(f"Failed to check/create collection: {e}")
            # Continue anyway, collection might exist

    async def _create_collection(self) -> None:
        """Create the collection with explicit schema for key-value storage."""
        try:
            # Define explicit schema with all fields defined upfront
            schema = {
                "dimension": 384,  # Default embedding dimension
                "primary_field": "id",
                "id_type": "int",
                "vector_field": "vector",
                "metric_type": "COSINE",
                "auto_id": True,
                "enable_dynamic_field": True,  # Enable dynamic fields for our custom fields
                "other_fields": [
                    {"name": "key", "type": "VARCHAR", "max_length": 512},
                    {"name": "content", "type": "VARCHAR", "max_length": 65535},
                    {"name": "text", "type": "VARCHAR", "max_length": 65535},
                ],
            }

            await self.mcp_client.call_tool(
                "milvus_create_collection",
                {"collection_name": self.collection_name, "collection_schema": schema},
            )

            # Load the collection into memory
            await self.mcp_client.call_tool(
                "milvus_load_collection", {"collection_name": self.collection_name}
            )

            logger.info(f"Created and loaded collection: {self.collection_name}")

        except Exception as e:
            logger.warning(f"Failed to create collection {self.collection_name}: {e}")
            # Continue anyway, collection might already exist

    # Internal async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key from Milvus asynchronously (internal method).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        await self._ensure_connected()

        try:
            logger.debug(f"Milvus get: {key}")

            # Use Milvus query for exact key match via MCP
            result = await self.mcp_client.call_tool(
                "milvus_query",
                {
                    "collection_name": self.collection_name,
                    "filter_expr": f"key == '{key}'",  # Key is stored as direct string
                    "output_fields": ["key", "content"],
                    "limit": 1,
                },
            )

            logger.debug(f"Milvus query result: {result}")

            # Handle MCP response format
            if result and not result.get("isError", False) and "content" in result:
                content = result["content"]
                if content and isinstance(content, list) and len(content) > 0:
                    response_text = content[0].get("text", "")
                    # Parse the response to extract the content
                    if "Query results" in response_text and response_text.strip():
                        # Try to extract JSON data from the response
                        lines = response_text.split("\n")
                        for line in lines:
                            if line.strip() and "{" in line:
                                try:
                                    # Try to parse as JSON first
                                    try:
                                        data = json.loads(line.strip())
                                    except json.JSONDecodeError:
                                        # If JSON fails, try to evaluate as Python dict
                                        data = eval(line.strip())

                                    if isinstance(data, dict) and "content" in data:
                                        stored_content = data["content"]
                                        # Content is now stored as direct string (not array)
                                        # Parse the stored content
                                        try:
                                            decoded_value = json.loads(stored_content)

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
                                                    logger.debug(
                                                        f"Failed to reconstruct Thought from dict: {e}"
                                                    )
                                                    return decoded_value

                                            return decoded_value
                                        except json.JSONDecodeError:
                                            return stored_content
                                except json.JSONDecodeError:
                                    continue

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
        await self._ensure_connected()

        try:
            logger.debug(f"Milvus set: {key}")

            # Handle different value types
            if hasattr(value, "model_dump"):
                # Pydantic model - serialize to JSON
                serialized_value = json.dumps(value.model_dump(), default=str)
                text_value = str(value.text) if hasattr(value, "text") else str(value)
            elif isinstance(value, (dict, list)):
                # Dict or list - serialize to JSON
                serialized_value = json.dumps(value, default=str)
                text_value = str(value)
            else:
                # Other types - convert to string
                serialized_value = str(value)
                text_value = str(value)

            # First, delete any existing entry with this key
            try:
                await self.mcp_client.call_tool(
                    "milvus_delete_entities",
                    {"collection_name": self.collection_name, "filter_expr": f"key == '{key}'"},
                )
            except Exception:
                # Ignore errors if entity doesn't exist
                pass

            # Insert new data using Milvus insert_data via MCP
            # Generate a simple vector from the text content for search
            # In a real implementation, you'd use proper embeddings
            dummy_vector = [float(abs(hash(text_value + str(i))) % 100) / 100.0 for i in range(384)]

            # Ensure all vector elements are proper floats
            dummy_vector = [float(x) for x in dummy_vector]

            # Try the format that worked in our tests - single record insertion
            # Based on our successful test, this should work
            # Single record as batch operation: vector stays as list, other fields become single-element lists
            # Single record insertion: ALL fields as direct values
            # MCP requires field arrays format: vector as direct list, other fields as single-element lists
            # Field arrays format (accepted by both MCP and pymilvus)
            # Proper list-of-dicts format (now supported by fixed MCP server)
            data = [
                {
                    "vector": dummy_vector,  # Vector as direct list
                    "key": key,  # Direct string value
                    "content": serialized_value,  # Direct string value
                    "text": text_value,  # Direct string value
                }
            ]

            result = await self.mcp_client.call_tool(
                "milvus_insert_data", {"collection_name": self.collection_name, "data": data}
            )

            # Check for errors in response
            if result and result.get("isError", False):
                error_msg = "Unknown error"
                if "content" in result and result["content"]:
                    error_msg = result["content"][0].get("text", error_msg)
                raise Exception(f"Milvus INSERT failed: {error_msg}")

        except Exception as e:
            logger.error(f"Milvus set failed for key {key}: {e}")
            raise

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query using text search asynchronously (internal method).

        This is where Milvus shines - semantic search over stored content.

        Args:
            query: The search query for text search.
            limit: Maximum number of results to return.

        Returns:
            List of matching values, ranked by relevance.
        """
        await self._ensure_connected()

        try:
            logger.debug(f"Milvus search: '{query}', limit {limit}")

            # Use Milvus text search via MCP
            result = await self.mcp_client.call_tool(
                "milvus_text_search",
                {
                    "collection_name": self.collection_name,
                    "query_text": query,
                    "limit": limit,
                    "output_fields": ["key", "content", "text"],
                },
            )

            values = []
            if result and not result.get("isError", False) and "content" in result:
                content = result["content"]
                if content and isinstance(content, list) and len(content) > 0:
                    response_text = content[0].get("text", "")

                    # Parse the response to extract search results
                    if "Search results" in response_text:
                        lines = response_text.split("\n")
                        for line in lines:
                            if line.strip() and "{" in line:
                                try:
                                    # Parse the result line as JSON
                                    data = json.loads(line.strip())
                                    if isinstance(data, dict) and "content" in data:
                                        stored_content = data["content"]
                                        # Parse the stored content
                                        try:
                                            decoded_value = json.loads(stored_content)

                                            # If the value is a dictionary and looks like a Thought, try to reconstruct it
                                            if (
                                                isinstance(decoded_value, dict)
                                                and "id" in decoded_value
                                                and "prompt" in decoded_value
                                            ):
                                                try:
                                                    from sifaka.core.thought import Thought

                                                    values.append(Thought.from_dict(decoded_value))
                                                except Exception as e:
                                                    logger.debug(
                                                        f"Failed to reconstruct Thought from dict: {e}"
                                                    )
                                                    values.append(decoded_value)
                                            else:
                                                values.append(decoded_value)
                                        except json.JSONDecodeError:
                                            values.append(stored_content)
                                except json.JSONDecodeError:
                                    continue

            return values

        except Exception as e:
            logger.error(f"Milvus search failed for query '{query}': {e}")
            return []

    async def _clear_async(self) -> None:
        """Clear all data from the Milvus collection asynchronously (internal method)."""
        await self._ensure_connected()

        try:
            logger.debug(f"Milvus clear: collection '{self.collection_name}'")

            # Call Milvus delete_entities to clear all data via MCP
            result = await self.mcp_client.call_tool(
                "milvus_delete_entities",
                {
                    "collection_name": self.collection_name,
                    "filter_expr": "key != ''",  # Delete all entities with keys
                },
            )

            # Check for errors in response
            if result and result.get("isError", False):
                error_msg = "Unknown error"
                if "content" in result and result["content"]:
                    error_msg = result["content"][0].get("text", error_msg)
                raise Exception(f"Milvus CLEAR failed: {error_msg}")

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
        await self._ensure_connected()

        try:
            logger.debug(f"Milvus delete: {key}")

            # Call Milvus delete_entities for specific key via MCP
            result = await self.mcp_client.call_tool(
                "milvus_delete_entities",
                {"collection_name": self.collection_name, "filter_expr": f"key == '{key}'"},
            )

            # Handle MCP response format
            if result and not result.get("isError", False) and "content" in result:
                content = result["content"]
                if content and isinstance(content, list) and len(content) > 0:
                    response_text = content[0].get("text", "")
                    # Check if deletion was successful
                    return "deleted" in response_text.lower()

            return False

        except Exception as e:
            logger.error(f"Milvus delete failed for key {key}: {e}")
            return False

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously (internal method).

        Returns:
            List of all storage keys.
        """
        await self._ensure_connected()

        try:
            logger.debug(f"Milvus keys: collection '{self.collection_name}'")

            # Use Milvus query to get all entities and extract keys via MCP
            result = await self.mcp_client.call_tool(
                "milvus_query",
                {
                    "collection_name": self.collection_name,
                    "filter_expr": "key != ''",  # Get all entities with non-empty keys
                    "output_fields": ["key"],
                    "limit": 10000,  # Large limit to get all keys
                },
            )

            keys = []
            if result and not result.get("isError", False) and "content" in result:
                content = result["content"]
                if content and isinstance(content, list) and len(content) > 0:
                    response_text = content[0].get("text", "")

                    # Parse the response to extract keys
                    if "Query results" in response_text:
                        lines = response_text.split("\n")
                        for line in lines:
                            if line.strip() and "{" in line:
                                try:
                                    # Parse the result line as JSON
                                    data = json.loads(line.strip())
                                    if isinstance(data, dict) and "key" in data:
                                        key = data["key"]
                                        if key and key not in keys:
                                            keys.append(key)
                                except json.JSONDecodeError:
                                    continue

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
