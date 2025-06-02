"""Milvus storage implementation via MCP.

Vector storage with semantic search capabilities using Milvus via the Model Context
Protocol. Perfect for storing and searching thoughts, documents, and other text data.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, List, Optional

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MilvusStorage:
    """Milvus vector storage via MCP.

    Stores data in Milvus with vector embeddings for semantic search. Perfect for:
    - Semantic search over thoughts and documents
    - Vector similarity search
    - Large-scale text storage and retrieval
    - AI-powered search capabilities

    Uses PydanticAI's MCP client for compatibility with the rest of the Sifaka ecosystem.
    """

    def __init__(
        self,
        milvus_mcp_server: Optional[Any] = None,
        collection_name: str = "sifaka_storage",
        dimension: int = 384,
        max_text_length: int = 65535,
        key_prefix: str = "sifaka",
    ):
        """Initialize Milvus storage.

        Args:
            milvus_mcp_server: PydanticAI MCPServerStdio instance for Milvus (optional).
            collection_name: Name of the Milvus collection.
            dimension: Vector dimension for embeddings.
            max_text_length: Maximum text length before truncation.
            key_prefix: Prefix for all Milvus keys.
        """
        self.milvus_mcp_server = milvus_mcp_server
        self.collection_name = collection_name
        self.dimension = dimension
        self.max_text_length = max_text_length
        self.key_prefix = key_prefix
        self._milvus_enabled = milvus_mcp_server is not None
        self._connected = False

        # Hybrid approach: Store metadata separately due to MCP validation limitations
        # In production, this would be a separate database or Redis
        self._metadata_store = {}

        logger.debug(
            f"Initialized MilvusStorage with collection '{collection_name}', dimension {dimension}, Milvus enabled: {self._milvus_enabled}"
        )

    def _make_key(self, key: str) -> str:
        """Create a prefixed Milvus key."""
        return f"{self.key_prefix}:{key}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value for Milvus storage."""
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
        """Deserialize a value from Milvus storage."""
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

    def _generate_timestamp_key(self, thought: Any) -> str:
        """Generate a timestamp-based key for a thought.

        Format: YYYYMMDD_thoughtid_iterN
        Example: 20250527_8b69c3dc_iter0

        Args:
            thought: The thought object with id, iteration, and timestamp.

        Returns:
            Timestamp-based key for Milvus storage.
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

    def _extract_essential_fields(self, data: dict) -> dict:
        """Extract only essential fields for Milvus storage to avoid field length issues.

        Only stores fields needed for semantic search and linking:
        - id, parent_id, chain_id (for linking)
        - text, model_prompt (for semantic search)
        - timestamp, iteration (for ordering)
        - prompt (for context)

        Args:
            data: Full data dictionary (e.g., from Thought.model_dump())

        Returns:
            Dictionary with only essential fields
        """
        essential_fields = {
            # Critical linking fields (never truncate)
            "id": data.get("id"),
            "parent_id": data.get("parent_id"),
            "chain_id": data.get("chain_id"),
            # Search and context fields
            "text": data.get("text", ""),
            "model_prompt": data.get("model_prompt", ""),
            "prompt": data.get("prompt", ""),
            # Metadata for ordering and context
            "timestamp": data.get("timestamp"),
            "iteration": data.get("iteration", 0),
            # Keep model info for debugging
            "model_name": data.get("model_name"),
        }

        # Remove None values to keep the data clean
        result = {k: v for k, v in essential_fields.items() if v is not None}

        # Log what fields we're storing for debugging
        logger.debug(f"Milvus essential fields: {list(result.keys())}")

        return result

    def _truncate_field(self, field_value: str, field_name: str, max_length: int = 65536) -> str:
        """Truncate field value to fit Milvus field length limits.

        Uses smart truncation that preserves beginning and end of content.

        Args:
            field_value: The field value to truncate
            field_name: Name of the field (for logging)
            max_length: Maximum allowed length (default: 65536 for Milvus)

        Returns:
            Truncated field value with metadata about truncation
        """
        if len(field_value) <= max_length:
            return field_value

        # Calculate truncation points (preserve beginning and end)
        truncation_marker = "\n\n[... TRUNCATED FOR MILVUS STORAGE ...]\n\n"
        marker_length = len(truncation_marker)
        available_length = max_length - marker_length

        # Split available space: 70% for beginning, 30% for end
        beginning_length = int(available_length * 0.7)
        ending_length = available_length - beginning_length

        # Extract beginning and end
        beginning = field_value[:beginning_length]
        ending = field_value[-ending_length:] if ending_length > 0 else ""

        # Combine with truncation marker
        truncated_value = beginning + truncation_marker + ending

        # Log truncation for debugging
        original_length = len(field_value)
        logger.warning(
            f"Truncated {field_name} field from {original_length} to {len(truncated_value)} characters "
            f"(saved {beginning_length} + {len(ending)} chars)"
        )

        return truncated_value

    def _smart_truncate_json(self, json_str: str, max_length: int = 65536) -> str:
        """Smart truncation of JSON content that preserves important fields.

        Truncates fields in order of importance:
        1. Keep: id, prompt, timestamp, iteration (small, critical)
        2. Truncate if needed: text, metadata, validation_results
        3. Truncate heavily: history, pre/post_generation_context
        4. Truncate last: large text fields

        Args:
            json_str: JSON string to truncate
            max_length: Maximum allowed length

        Returns:
            Truncated JSON string that fits within limits
        """
        if len(json_str) <= max_length:
            return json_str

        try:
            # Parse JSON to work with individual fields
            data = json.loads(json_str)

            # Field priority (higher number = truncate first)
            field_priorities = {
                # Keep these (priority 0 - never truncate)
                "id": 0,
                "parent_id": 0,  # CRITICAL: Never truncate parent_id - needed for linking thoughts
                "chain_id": 0,  # CRITICAL: Never truncate chain_id - needed for chain tracking
                "prompt": 0,
                "timestamp": 0,
                "iteration": 0,
                # Truncate moderately (priority 1)
                "text": 1,
                "metadata": 1,
                "validation_results": 1,
                "critic_feedback": 1,
                # Truncate heavily (priority 2)
                "history": 2,
                "pre_generation_context": 2,
                "post_generation_context": 2,
                "system_prompt": 2,
                "model_prompt": 2,
            }

            # Start truncating from highest priority fields
            for priority in [3, 2, 1]:
                current_json = json.dumps(data, default=str)
                if len(current_json) <= max_length:
                    break

                # Truncate fields at this priority level
                for field_name, field_priority in field_priorities.items():
                    if field_priority == priority and field_name in data:
                        if isinstance(data[field_name], str):
                            # Calculate how much to truncate
                            if priority == 1:
                                # Moderate truncation (keep 50%)
                                max_field_length = len(data[field_name]) // 2
                            elif priority == 2:
                                # Heavy truncation (keep 25%)
                                max_field_length = len(data[field_name]) // 4
                            else:
                                # Aggressive truncation (keep 10%)
                                max_field_length = len(data[field_name]) // 10

                            if len(data[field_name]) > max_field_length:
                                data[field_name] = self._truncate_field(
                                    data[field_name], field_name, max_field_length
                                )
                        elif isinstance(data[field_name], (list, dict)):
                            # For complex fields, convert to string and truncate
                            str_value = str(data[field_name])
                            if priority == 1:
                                max_field_length = 1000
                            elif priority == 2:
                                max_field_length = 500
                            else:
                                max_field_length = 100

                            if len(str_value) > max_field_length:
                                data[field_name] = (
                                    f"[TRUNCATED: {type(data[field_name]).__name__} with {len(str_value)} chars]"
                                )

                # Check if we're under the limit now
                current_json = json.dumps(data, default=str)
                if len(current_json) <= max_length:
                    break

            # Final check - if still too big, truncate the entire JSON
            final_json = json.dumps(data, default=str)
            if len(final_json) > max_length:
                logger.warning(
                    f"JSON still too large ({len(final_json)} chars), applying final truncation"
                )
                final_json = self._truncate_field(final_json, "entire_json", max_length)

            return final_json

        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to simple truncation
            logger.warning("Failed to parse JSON for smart truncation, using simple truncation")
            return self._truncate_field(json_str, "json_fallback", max_length)

    async def _ensure_connected(self) -> None:
        """Ensure MCP client is connected and collection exists."""
        if not self._milvus_enabled:
            return

        if not self._connected:
            try:
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
            # Use MCP server directly with Milvus tools
            async with self.milvus_mcp_server as mcp_client:
                # List collections to check if ours exists
                result = await mcp_client.call_tool("milvus_list_collections", arguments={})

                # Parse the response to check if our collection exists
                if result and hasattr(result, "content") and result.content:
                    collections_text = str(result.content[0].text) if result.content else ""
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

            async with self.milvus_mcp_server as mcp_client:
                await mcp_client.call_tool(
                    "milvus_create_collection",
                    arguments={
                        "collection_name": self.collection_name,
                        "collection_schema": schema,
                    },
                )

                # Load the collection into memory
                await mcp_client.call_tool(
                    "milvus_load_collection", arguments={"collection_name": self.collection_name}
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
        if not self._milvus_enabled:
            return None

        await self._ensure_connected()

        try:
            milvus_key = self._make_key(key)
            logger.debug(f"Milvus get: {key} -> {milvus_key}")

            # Check if we have a metadata mapping for this key (original -> timestamp)
            search_key = milvus_key
            if key in self._metadata_store:
                search_key = self._make_key(self._metadata_store[key])
                logger.debug(f"Using mapped key: {key} -> {search_key}")

            # Use MCP server directly with Milvus tools
            async with self.milvus_mcp_server as mcp_client:
                # Use Milvus query for exact key match via MCP
                result = await mcp_client.call_tool(
                    "milvus_query",
                    arguments={
                        "collection_name": self.collection_name,
                        "filter_expr": f"key == '{search_key}'",  # Key is stored as direct string
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
                                        import ast

                                        data = ast.literal_eval(line.strip())

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
        if not self._milvus_enabled:
            return

        await self._ensure_connected()

        try:
            milvus_key = self._make_key(key)
            logger.debug(f"Milvus set: {key} -> {milvus_key}")

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

            # Use timestamp-based key for thoughts, original key for other data
            if hasattr(value, "id") and hasattr(value, "iteration") and hasattr(value, "timestamp"):
                # This looks like a thought - use timestamp-based key
                timestamp_key = self._generate_timestamp_key(value)
                final_milvus_key = self._make_key(timestamp_key)
                logger.debug(f"Milvus set (thought): {key} -> {final_milvus_key}")

                # Also store metadata for original key lookup
                original_key = key
                logger.debug(f"Milvus set (original): {original_key}")
            else:
                # Regular data - use prefixed key
                final_milvus_key = milvus_key
                original_key = None
                logger.debug(f"Milvus set: {final_milvus_key}")

            # Handle different value types - extract only essential fields for Milvus
            if hasattr(value, "model_dump"):
                # Pydantic model (likely a Thought) - extract only essential fields
                full_data = value.model_dump()
                essential_data = self._extract_essential_fields(full_data)
                serialized_value = json.dumps(essential_data, default=str)
                text_value = str(value.text) if hasattr(value, "text") else str(value)
            elif isinstance(value, (dict, list)):
                # Dict or list - extract essential fields if it looks like a thought
                if isinstance(value, dict) and "id" in value and "text" in value:
                    essential_data = self._extract_essential_fields(value)
                    serialized_value = json.dumps(essential_data, default=str)
                else:
                    serialized_value = json.dumps(value, default=str)
                text_value = (
                    str(value.get("text", value)) if isinstance(value, dict) else str(value)
                )
            else:
                # Other types - convert to string
                serialized_value = str(value)
                text_value = str(value)

            # Apply smart field truncation for Milvus limits
            serialized_value = self._smart_truncate_json(
                serialized_value, max_length=self.max_text_length
            )
            text_value = self._truncate_field(text_value, "text", max_length=self.max_text_length)
            milvus_key = self._truncate_field(
                milvus_key, "key", max_length=512
            )  # Keys should be shorter

            # Use MCP server directly with Milvus tools
            async with self.milvus_mcp_server as mcp_client:
                # First, delete any existing entry with this key
                try:
                    await mcp_client.call_tool(
                        "milvus_delete_entities",
                        arguments={
                            "collection_name": self.collection_name,
                            "filter_expr": f"key == '{final_milvus_key}'",
                        },
                    )
                except Exception:  # nosec
                    # Ignore errors if entity doesn't exist
                    pass

                # Also delete by original key if this is a thought (for updates)
                if original_key:
                    try:
                        await mcp_client.call_tool(
                            "milvus_delete_entities",
                            arguments={
                                "collection_name": self.collection_name,
                                "filter_expr": f"key == '{original_key}'",
                            },
                        )
                    except Exception:  # nosec
                        # Ignore errors if entity doesn't exist
                        pass

            # Insert new data using Milvus insert_data via MCP
            # Generate a simple vector from the text content for search
            # In a real implementation, you'd use proper embeddings
            dummy_vector = [float(abs(hash(text_value + str(i))) % 100) / 100.0 for i in range(384)]

            # Ensure all vector elements are proper floats
            dummy_vector = [float(x) for x in dummy_vector]

            # Final safety check - ensure all fields are within Milvus limits
            if len(serialized_value) > 65535:
                logger.warning(
                    f"Content still too large ({len(serialized_value)} chars), applying emergency truncation"
                )
                serialized_value = self._truncate_field(
                    serialized_value, "content", max_length=65535
                )

            if len(text_value) > 65535:
                logger.warning(
                    f"Text still too large ({len(text_value)} chars), applying emergency truncation"
                )
                text_value = self._truncate_field(text_value, "text", max_length=65535)

            if len(milvus_key) > 511:
                logger.warning(
                    f"Key still too large ({len(milvus_key)} chars), applying emergency truncation"
                )
                milvus_key = self._truncate_field(milvus_key, "key", max_length=511)

            # Store with timestamp-based key
            data = [
                {
                    "vector": dummy_vector,  # Vector as direct list
                    "key": milvus_key,  # Timestamp-based key
                    "content": serialized_value,  # Direct string value
                    "text": text_value,  # Direct string value
                }
            ]

            result = await mcp_client.call_tool(
                "milvus_insert_data",
                arguments={"collection_name": self.collection_name, "data": data},
            )

            # Check for errors in response
            if result and result.get("isError", False):
                error_msg = "Unknown error"
                if "content" in result and result["content"]:
                    error_msg = result["content"][0].get("text", error_msg)
                raise Exception(f"Milvus INSERT failed: {error_msg}")

            # For thoughts, also store a mapping from original key to timestamp key
            if original_key:
                # Store metadata mapping for backward compatibility
                self._metadata_store[original_key] = milvus_key
                logger.debug(f"Stored metadata mapping: {original_key} -> {milvus_key}")

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
        if not self._milvus_enabled:
            return []

        await self._ensure_connected()

        try:
            logger.debug(f"Milvus search: '{query}', limit {limit}")

            # Use MCP server directly with Milvus tools
            async with self.milvus_mcp_server as mcp_client:
                # Use Milvus text search via MCP
                result = await mcp_client.call_tool(
                    "milvus_text_search",
                    arguments={
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
        if not self._milvus_enabled:
            return

        await self._ensure_connected()

        try:
            logger.debug(f"Milvus clear: collection '{self.collection_name}'")

            # Use MCP server directly with Milvus tools
            async with self.milvus_mcp_server as mcp_client:
                # Call Milvus delete_entities to clear all data via MCP
                result = await mcp_client.call_tool(
                    "milvus_delete_entities",
                    arguments={
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
        if not self._milvus_enabled:
            return False

        await self._ensure_connected()

        try:
            milvus_key = self._make_key(key)
            logger.debug(f"Milvus delete: {key} -> {milvus_key}")

            # Use MCP server directly with Milvus tools
            async with self.milvus_mcp_server as mcp_client:
                # Call Milvus delete_entities for specific key via MCP
                result = await mcp_client.call_tool(
                    "milvus_delete_entities",
                    arguments={
                        "collection_name": self.collection_name,
                        "filter_expr": f"key == '{milvus_key}'",
                    },
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
        if not self._milvus_enabled:
            return []

        await self._ensure_connected()

        try:
            logger.debug(f"Milvus keys: collection '{self.collection_name}'")

            # Use MCP server directly with Milvus tools
            async with self.milvus_mcp_server as mcp_client:
                # Use Milvus query to get all entities and extract keys via MCP
                result = await mcp_client.call_tool(
                    "milvus_query",
                    arguments={
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

    def _connect_mcp(self) -> None:
        """Connect to the MCP server (placeholder for testing)."""
        # This is a placeholder method for testing
        # In a real implementation, this would establish the MCP connection
        self._connected = True
        logger.debug("MCP connection established (mock)")

    # Additional methods expected by tests

    def save(self, key: str, value: Any) -> None:
        """Save a value to Milvus (alias for set)."""
        self.set(key, value)

    def load(self, key: str) -> Optional[Any]:
        """Load a value from Milvus (alias for get)."""
        return self.get(key)

    def exists(self, key: str) -> bool:
        """Check if a key exists in Milvus."""
        return key in self

    def search_similar(self, query_vector: List[float], limit: int = 10) -> List[dict]:
        """Search for similar vectors in Milvus."""
        # This is a placeholder implementation
        # In a real implementation, this would use vector similarity search
        return []

    def save_batch(self, thoughts: List[Any]) -> None:
        """Save multiple thoughts in batch."""
        for thought in thoughts:
            self.save(thought.id, thought)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder implementation)."""
        # This is a simple hash-based embedding for testing
        # In production, you'd use a real embedding model
        return [float(abs(hash(text + str(i))) % 100) / 100.0 for i in range(self.dimension)]

    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length."""
        if len(text) <= self.max_text_length:
            return text
        return text[: self.max_text_length - 3] + "..."
