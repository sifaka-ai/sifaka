"""In-memory storage backend for temporary result storage.

This module provides a simple dictionary-based storage backend that keeps
SifakaResult objects in memory. Data is lost when the process ends, making
this suitable for:

- Development and testing
- Temporary result caching
- Situations where persistence isn't required
- Quick prototyping

## Characteristics:

- **Fast**: No I/O overhead, direct memory access
- **Simple**: No configuration or setup required
- **Limited**: Constrained by available memory
- **Volatile**: Data lost on process termination

## Usage:

    >>> from sifaka import improve, MemoryStorage
    >>> storage = MemoryStorage()
    >>>
    >>> # Use with improve
    >>> result = await improve("text", storage=storage)
    >>>
    >>> # Later retrieval
    >>> loaded = await storage.load(result.id)
    >>>
    >>> # Check storage size
    >>> print(f"Storing {storage.size()} results")

## Limitations:

- No persistence across restarts
- No concurrent access from multiple processes
- Memory usage grows with stored results
- No automatic cleanup or expiration
"""

from typing import Dict, List, Optional

from ..core.exceptions import StorageError
from ..core.models import SifakaResult
from .base import StorageBackend


class MemoryStorage(StorageBackend):
    """Non-persistent storage backend using Python dictionaries.

    MemoryStorage provides the simplest possible storage implementation,
    keeping all results in a dictionary. It's the default storage backend
    when none is specified, offering fast access without any external
    dependencies.

    The storage is process-local and volatile - all data is lost when
    the Python process ends. This makes it ideal for development, testing,
    and scenarios where persistence isn't needed.

    Example:
        >>> # Create storage
        >>> storage = MemoryStorage()
        >>>
        >>> # Save a result
        >>> result_id = await storage.save(result)
        >>>
        >>> # List all stored results
        >>> all_ids = await storage.list()
        >>>
        >>> # Search for specific content
        >>> matching_ids = await storage.search("machine learning")
        >>>
        >>> # Clear all storage
        >>> storage.clear()

    Thread Safety:
        The basic dictionary operations are thread-safe in CPython due to
        the GIL, but concurrent access from async coroutines is safe as
        long as operations are atomic.
    """

    def __init__(self) -> None:
        """Initialize empty in-memory storage.

        Creates an empty dictionary to store SifakaResult objects keyed
        by their IDs. No configuration is needed.
        """
        self._storage: Dict[str, SifakaResult] = {}

    async def save(self, result: SifakaResult) -> str:
        """Save a result to memory storage.

        Stores the complete SifakaResult object in the internal dictionary.
        The result's ID is used as the key.

        Args:
            result: Complete SifakaResult to store

        Returns:
            The result's ID for later retrieval

        Raises:
            StorageError: If save fails (unlikely for memory storage)

        Note:
            Results are stored by reference, so modifications to the result
            object after saving will be reflected in storage.
        """
        try:
            self._storage[result.id] = result
            return result.id
        except Exception as e:
            raise StorageError(
                f"Failed to save result {result.id}",
                storage_type="memory",
                operation="save",
            ) from e

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a result from memory storage.

        Retrieves a previously saved result by its ID.

        Args:
            result_id: Unique identifier of the result to load

        Returns:
            The stored SifakaResult if found, None otherwise

        Raises:
            StorageError: If load operation fails

        Note:
            Returns the actual stored object, not a copy. Modifications
            will affect the stored version.
        """
        try:
            return self._storage.get(result_id)
        except Exception as e:
            raise StorageError(
                f"Failed to load result {result_id}",
                storage_type="memory",
                operation="load",
            ) from e

    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List IDs of all stored results with pagination.

        Returns result IDs in the order they were added (dictionary
        insertion order is preserved in Python 3.7+).

        Args:
            limit: Maximum number of IDs to return
            offset: Number of IDs to skip from the beginning

        Returns:
            List of result IDs, may be empty

        Example:
            >>> # Get first 10 results
            >>> first_page = await storage.list(limit=10)
            >>> # Get next 10
            >>> second_page = await storage.list(limit=10, offset=10)
        """
        ids = list(self._storage.keys())
        return ids[offset : offset + limit]

    async def delete(self, result_id: str) -> bool:
        """Delete a result from memory storage.

        Removes the result from storage, freeing its memory.

        Args:
            result_id: ID of the result to delete

        Returns:
            True if a result was deleted, False if ID not found

        Note:
            This is immediate and permanent within the process lifetime.
        """
        if result_id in self._storage:
            del self._storage[result_id]
            return True
        return False

    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search for results containing specific text.

        Performs case-insensitive substring search in both original
        and final text of stored results.

        Args:
            query: Text to search for (case-insensitive)
            limit: Maximum number of matching IDs to return

        Returns:
            List of IDs for results containing the query text

        Raises:
            StorageError: If search operation fails

        Example:
            >>> # Find results about AI
            >>> ai_results = await storage.search("artificial intelligence")
            >>> for result_id in ai_results:
            ...     result = await storage.load(result_id)
            ...     print(result.final_text[:100])
        """
        try:
            matches = []
            query_lower = query.lower()

            for result_id, result in self._storage.items():
                # Simple text search in original and final text
                if (
                    query_lower in result.original_text.lower()
                    or query_lower in result.final_text.lower()
                ):
                    matches.append(result_id)

                if len(matches) >= limit:
                    break

            return matches
        except Exception as e:
            raise StorageError(
                f"Search failed for query: {query}",
                storage_type="memory",
                operation="search",
            ) from e

    def clear(self) -> None:
        """Remove all results from storage.

        Clears the internal dictionary, removing all stored results.
        This operation cannot be undone.

        Example:
            >>> print(f"Before: {storage.size()} results")
            >>> storage.clear()
            >>> print(f"After: {storage.size()} results")  # Will be 0
        """
        self._storage.clear()

    def size(self) -> int:
        """Get the number of results currently in storage.

        Returns:
            Count of stored results

        Example:
            >>> if storage.size() > 1000:
            ...     print("Warning: High memory usage")
            ...     storage.clear()
        """
        return len(self._storage)
