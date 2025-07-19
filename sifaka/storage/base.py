"""Abstract base class for Sifaka storage backends.

This module defines the interface that all storage backends must implement.
Storage backends handle persistence of SifakaResult objects, allowing for:
- Saving improvement sessions for later analysis
- Loading previous results for continuation or review
- Searching through historical improvements
- Building datasets of text improvements

Built-in implementations include FileStorage, RedisStorage, and PostgresStorage.
Custom backends can be created by inheriting from StorageBackend."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.models import SifakaResult
from ..core.validation import validate_result_id, validate_sifaka_result


class StorageBackend(ABC):
    """Abstract base class for persistent storage of Sifaka results.

    Storage backends provide a way to persist SifakaResult objects beyond
    the lifetime of a single improvement session. This enables features like:
    - Analyzing improvement patterns over time
    - Building training datasets from improvements
    - Debugging and auditing text transformations
    - Sharing results across systems or teams

    All methods are async to support both local and remote storage systems.

    Example:
        >>> class MyStorage(StorageBackend):
        ...     async def save(self, result):
        ...         # Implementation here
        ...         return result.id
        >>>
        >>> storage = MyStorage()
        >>> result = await improve("text", storage=storage)
    """

    @abstractmethod
    async def save(self, result: SifakaResult) -> str:
        """Save a SifakaResult to persistent storage.

        Implementations should serialize the entire result object including
        all generations, critiques, and metadata. The returned ID should be
        unique and suitable for later retrieval.

        Args:
            result: The complete SifakaResult object to persist. Contains
                the original text, final text, all intermediate generations,
                critiques, validations, and metadata.

        Returns:
            A unique string identifier for retrieving this result later.
            This could be the result.id, a database primary key, or any
            other unique identifier.

        Raises:
            StorageError: If the save operation fails

        Note:
            Implementations should handle serialization of datetime objects,
            deque collections, and nested Pydantic models appropriately.
        """

    @abstractmethod
    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a previously saved SifakaResult from storage.

        Implementations should deserialize the complete result object
        with all its history and metadata intact.

        Args:
            result_id: The unique identifier returned by save(). This is
                used to locate the specific result in storage.

        Returns:
            The complete SifakaResult object if found, with all generations,
            critiques, and metadata restored. Returns None if no result
            exists with the given ID.

        Raises:
            StorageError: If the load operation fails (not including not found)

        Note:
            The loaded result should be identical to what was saved, including
            preservation of deque maxlen settings and all timestamps.
        """

    @abstractmethod
    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List the IDs of stored results with pagination.

        Useful for browsing available results or building indexes.
        Results should be returned in a consistent order (e.g., by
        creation time or ID).

        Args:
            limit: Maximum number of IDs to return. Implementations should
                enforce reasonable limits to prevent memory issues.
            offset: Number of results to skip from the beginning. Used for
                pagination through large result sets.

        Returns:
            List of result IDs that can be passed to load(). The list may
            be shorter than 'limit' if fewer results are available.

        Example:
            >>> # Get first 10 results
            >>> ids = await storage.list(limit=10, offset=0)
            >>> # Get next 10 results
            >>> more_ids = await storage.list(limit=10, offset=10)
        """

    @abstractmethod
    async def delete(self, result_id: str) -> bool:
        """Delete a stored result from persistent storage.

        Implementations should remove all data associated with the result,
        including any indexes or metadata.

        Args:
            result_id: The unique identifier of the result to delete

        Returns:
            True if a result was found and deleted, False if no result
            existed with that ID. Should not raise an error for missing IDs.

        Raises:
            StorageError: If the delete operation fails for other reasons

        Note:
            Deletes should be permanent. Implementations may choose to
            support soft deletes internally but this should be transparent
            to the caller.
        """

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search stored results by text content.

        Implementations should search through both original and final text,
        and potentially critique feedback. The search algorithm is
        implementation-specific (could be simple substring, full-text, or
        semantic search).

        Args:
            query: The search query string. Interpretation is implementation-
                specific but typically matches against text content.
            limit: Maximum number of matching results to return

        Returns:
            List of result IDs that match the query, ordered by relevance
            if the implementation supports ranking. Empty list if no matches.

        Example:
            >>> # Find results about machine learning
            >>> ml_ids = await storage.search("machine learning", limit=5)
            >>> for id in ml_ids:
            ...     result = await storage.load(id)
            ...     print(result.final_text[:100])
        """

    async def get_metadata(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get summary metadata for a result without loading the full object.

        This method provides a lightweight way to get basic information about
        a stored result. Useful for building indexes or summary views.

        Args:
            result_id: The unique identifier of the result

        Returns:
            Dictionary containing summary information about the result, or
            None if the result doesn't exist. Default implementation includes:
            - id: The result ID
            - original_text: First 100 chars of original text
            - final_text: First 100 chars of final text
            - iteration: Number of improvement iterations
            - processing_time: Total processing time in seconds
            - created_at: When the result was created
            - updated_at: When the result was last modified
            - num_generations: Count of text generations
            - num_validations: Count of validation runs
            - num_critiques: Count of critique evaluations

        Note:
            This default implementation loads the full result. Storage backends
            may override this with more efficient implementations that query
            just the metadata.
        """
        result = await self.load(result_id)
        if not result:
            return None

        return {
            "id": result.id,
            "original_text": (
                result.original_text[:100] + "..."
                if len(result.original_text) > 100
                else result.original_text
            ),
            "final_text": (
                result.final_text[:100] + "..."
                if len(result.final_text) > 100
                else result.final_text
            ),
            "iteration": result.iteration,
            "processing_time": result.processing_time,
            "created_at": result.created_at,
            "updated_at": result.updated_at,
            "num_generations": len(result.generations),
            "num_validations": len(result.validations),
            "num_critiques": len(result.critiques),
        }

    def _validate_result_before_save(self, result: SifakaResult) -> SifakaResult:
        """Validate SifakaResult before saving with enhanced validation.

        Args:
            result: The result to validate

        Returns:
            The validated result

        Raises:
            ValueError: If the result is invalid
        """
        return validate_sifaka_result(result)

    def _validate_result_id_format(self, result_id: str) -> str:
        """Validate result ID format with enhanced validation.

        Args:
            result_id: The result ID to validate

        Returns:
            The validated result ID

        Raises:
            ValueError: If the result ID is invalid
        """
        return validate_result_id(result_id)
