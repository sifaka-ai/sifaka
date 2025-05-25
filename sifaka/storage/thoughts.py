"""Thought storage implementation using unified 3-tier architecture.

This module provides CachedThoughtStorage which replaces the legacy JSON
persistence with a unified memory → cache → persistence pattern that includes
vector search capabilities.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought
from sifaka.persistence.base import ThoughtStorage, ThoughtQuery, ThoughtQueryResult
from sifaka.utils.logging import get_logger

from .base import CachedStorage, StorageError

logger = get_logger(__name__)


class CachedThoughtStorage(ThoughtStorage):
    """Unified thought storage: memory + cache + vectors.

    This class implements the ThoughtStorage interface using the unified
    3-tier storage architecture. It provides:

    - Fast access to recent thoughts (memory)
    - Cross-process caching (Redis)
    - Semantic search capabilities (Milvus)
    - Automatic persistence without blocking

    Attributes:
        storage: Underlying CachedStorage instance.
    """

    def __init__(self, storage: CachedStorage):
        """Initialize cached thought storage.

        Args:
            storage: CachedStorage instance for 3-tier storage.
        """
        self.storage = storage
        logger.debug("Initialized CachedThoughtStorage")

    def save_thought(self, thought: Thought) -> None:
        """Save a thought to storage.

        The thought is immediately saved to memory (L1) and asynchronously
        persisted to cache (L2) and vector storage (L3).

        Args:
            thought: The thought to save.

        Raises:
            StorageError: If the save operation fails.
        """
        try:
            key = f"thought:{thought.id}"

            # Prepare metadata for vector storage
            metadata = {
                "chain_id": [thought.chain_id] if thought.chain_id else [""],
                "iteration": [thought.iteration],
                "timestamp": [thought.timestamp.isoformat()],
                "has_text": [bool(thought.text)],
                "validation_passed": [self._all_validations_passed(thought)],
                "critic_count": [len(thought.critic_feedback or [])],
            }

            # Save to all tiers
            self.storage.set(key, thought, metadata)

            logger.debug(f"Saved thought {thought.id} (iteration {thought.iteration})")

        except Exception as e:
            raise StorageError(
                f"Failed to save thought {thought.id}",
                operation="save_thought",
                storage_type="CachedThoughtStorage",
                metadata={"thought_id": thought.id, "iteration": thought.iteration},
            ) from e

    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """Retrieve a thought by ID.

        Uses 3-tier lookup: memory → cache → persistence.

        Args:
            thought_id: The ID of the thought to retrieve.

        Returns:
            The thought if found, None otherwise.

        Raises:
            StorageError: If the retrieval operation fails.
        """
        try:
            key = f"thought:{thought_id}"
            thought = self.storage.get(key)

            if thought:
                logger.debug(f"Retrieved thought {thought_id}")
            else:
                logger.debug(f"Thought {thought_id} not found")

            return thought

        except Exception as e:
            raise StorageError(
                f"Failed to retrieve thought {thought_id}",
                operation="get_thought",
                storage_type="CachedThoughtStorage",
                metadata={"thought_id": thought_id},
            ) from e

    def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought from storage.

        Note: This implementation doesn't actually delete from persistent
        storage to preserve data integrity. It only removes from cache layers.

        Args:
            thought_id: The ID of the thought to delete.

        Returns:
            True if the thought was found and removed from cache, False otherwise.

        Raises:
            StorageError: If the delete operation fails.
        """
        try:
            key = f"thought:{thought_id}"

            # Check if thought exists
            thought = self.storage.get(key)
            if not thought:
                return False

            # Remove from memory and cache (but not persistence for data integrity)
            self.storage.memory.data.pop(key, None)
            # Note: Redis cache will expire naturally

            logger.debug(f"Deleted thought {thought_id} from cache layers")
            return True

        except Exception as e:
            raise StorageError(
                f"Failed to delete thought {thought_id}",
                operation="delete_thought",
                storage_type="CachedThoughtStorage",
                metadata={"thought_id": thought_id},
            ) from e

    def query_thoughts(self, query: Optional[ThoughtQuery] = None) -> ThoughtQueryResult:
        """Query thoughts based on criteria.

        Uses vector similarity search for semantic queries and filters
        results based on the query parameters.

        Args:
            query: Query parameters, or None for recent thoughts.

        Returns:
            Query result containing matching thoughts and metadata.

        Raises:
            StorageError: If the query operation fails.
        """
        try:
            if query is None:
                query = ThoughtQuery(limit=10)

            # Use semantic search if query text is provided
            if query.text_contains:
                thoughts = self.find_similar_thoughts_by_text(
                    query.text_contains, query.limit or 10
                )
            else:
                # For now, return recent thoughts from memory
                # In a full implementation, this would query persistence layer
                thoughts = self._get_recent_thoughts(query.limit or 10)

            # Apply filters
            filtered_thoughts = self._apply_filters(thoughts, query)

            # Sort results
            if query.sort_by:
                sort_desc = query.sort_order == "desc"
                filtered_thoughts = self._sort_thoughts(filtered_thoughts, query.sort_by, sort_desc)

            # Apply pagination
            start_idx = query.offset or 0
            end_idx = start_idx + (query.limit or 10)
            paginated_thoughts = filtered_thoughts[start_idx:end_idx]

            result = ThoughtQueryResult(
                thoughts=paginated_thoughts,
                total_count=len(filtered_thoughts),
                query=query,
                execution_time_ms=0.0,  # TODO: Add timing
            )

            logger.debug(
                f"Query returned {len(paginated_thoughts)} thoughts (total: {len(filtered_thoughts)})"
            )
            return result

        except Exception as e:
            raise StorageError(
                "Failed to query thoughts",
                operation="query_thoughts",
                storage_type="CachedThoughtStorage",
                metadata={"query": query.model_dump() if query else None},
            ) from e

    def find_similar_thoughts(self, thought: Thought, limit: int = 5) -> List[Thought]:
        """Find thoughts similar to the given thought.

        Uses vector similarity search on the thought's prompt and text content.

        Args:
            thought: The thought to find similar thoughts for.
            limit: Maximum number of similar thoughts to return.

        Returns:
            List of similar thoughts.
        """
        query_text = f"{thought.prompt}\n{thought.text or ''}"
        return self.find_similar_thoughts_by_text(query_text, limit)

    def find_similar_thoughts_by_text(self, query_text: str, limit: int = 5) -> List[Thought]:
        """Find thoughts similar to the given text.

        Args:
            query_text: Text to search for similar thoughts.
            limit: Maximum number of similar thoughts to return.

        Returns:
            List of similar thoughts.
        """
        try:
            similar_items = self.storage.search_similar(query_text, limit)
            # Filter to only return Thought objects
            thoughts = [item for item in similar_items if isinstance(item, Thought)]

            logger.debug(f"Found {len(thoughts)} similar thoughts for query: {query_text[:50]}...")
            return thoughts

        except Exception as e:
            logger.warning(f"Similar thoughts search failed: {e}")
            return []

    def get_thought_history(self, thought_id: str) -> List[Thought]:
        """Get the complete history of a thought.

        Args:
            thought_id: The ID of the thought.

        Returns:
            List of thoughts in the history chain, ordered by iteration.
        """
        # Get the thought first
        thought = self.get_thought(thought_id)
        if not thought:
            return []

        # Get all thoughts for this chain
        return self.get_chain_thoughts(thought.chain_id or "")

    def get_chain_thoughts(self, chain_id: str) -> List[Thought]:
        """Get all thoughts for a specific chain.

        Args:
            chain_id: The chain ID to get thoughts for.

        Returns:
            List of thoughts for the chain, sorted by iteration.
        """
        # This would ideally use a more efficient query on the persistence layer
        # For now, we'll use a text search approach
        query_text = f"chain_id:{chain_id}"
        thoughts = self.find_similar_thoughts_by_text(query_text, limit=100)

        # Filter and sort by chain_id and iteration
        chain_thoughts = [t for t in thoughts if t.chain_id == chain_id]
        chain_thoughts.sort(key=lambda t: t.iteration)

        return chain_thoughts

    def count_thoughts(self, query: Optional[ThoughtQuery] = None) -> int:
        """Count thoughts matching query criteria.

        Args:
            query: Query parameters, or None for all thoughts.

        Returns:
            Number of thoughts matching the criteria.
        """
        try:
            # Use query_thoughts and count the results
            result = self.query_thoughts(query)
            return result.total_count
        except Exception as e:
            logger.warning(f"Failed to count thoughts: {e}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the storage backend.

        Returns:
            Dictionary containing health status and metrics.
        """
        try:
            # Test basic operations
            test_thought = Thought(prompt="health_check_test")

            # Try to save and retrieve
            self.save_thought(test_thought)
            retrieved = self.get_thought(test_thought.id)

            # Clean up test thought
            self.delete_thought(test_thought.id)

            # Get storage stats
            stats = self.get_stats()

            return {
                "status": "healthy" if retrieved is not None else "degraded",
                "test_passed": retrieved is not None,
                "storage_stats": stats,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

    def _all_validations_passed(self, thought: Thought) -> bool:
        """Check if all validations passed for a thought."""
        if not thought.validation_results:
            return False
        return all(result.passed for result in thought.validation_results.values())

    def _get_recent_thoughts(self, limit: int) -> List[Thought]:
        """Get recent thoughts from memory storage."""
        # Get thoughts from memory storage
        thoughts = []
        for key, value in self.storage.memory.data.items():
            if key.startswith("thought:") and isinstance(value, Thought):
                thoughts.append(value)

        # Sort by timestamp (most recent first)
        thoughts.sort(key=lambda t: t.timestamp, reverse=True)
        return thoughts[:limit]

    def _apply_filters(self, thoughts: List[Thought], query: ThoughtQuery) -> List[Thought]:
        """Apply query filters to thoughts."""
        filtered = thoughts

        if query.chain_ids:
            filtered = [t for t in filtered if t.chain_id in query.chain_ids]

        if query.start_date:
            filtered = [t for t in filtered if t.timestamp >= query.start_date]

        if query.end_date:
            filtered = [t for t in filtered if t.timestamp <= query.end_date]

        if query.has_context is not None:
            filtered = [
                t
                for t in filtered
                if bool(t.pre_generation_context or t.post_generation_context) == query.has_context
            ]

        if query.has_validation_results is not None:
            filtered = [
                t for t in filtered if bool(t.validation_results) == query.has_validation_results
            ]

        return filtered

    def _sort_thoughts(
        self, thoughts: List[Thought], sort_by: str, desc: bool = False
    ) -> List[Thought]:
        """Sort thoughts by the specified field."""
        if sort_by == "timestamp":
            thoughts.sort(key=lambda t: t.timestamp, reverse=desc)
        elif sort_by == "iteration":
            thoughts.sort(key=lambda t: t.iteration, reverse=desc)
        elif sort_by == "chain_id":
            thoughts.sort(key=lambda t: t.chain_id or "", reverse=desc)

        return thoughts

    def clear(self) -> None:
        """Clear all thought storage."""
        self.storage.clear()
        logger.debug("Cleared all thought storage")

    def get_stats(self) -> Dict[str, Any]:
        """Get thought storage statistics."""
        base_stats = self.storage.get_stats()

        # Add thought-specific stats
        thought_count = sum(
            1 for key in self.storage.memory.data.keys() if key.startswith("thought:")
        )

        return {
            **base_stats,
            "thought_count_in_memory": thought_count,
            "storage_type": "CachedThoughtStorage",
        }
