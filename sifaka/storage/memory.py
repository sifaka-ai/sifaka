"""In-memory storage implementation.

This is the default storage backend that keeps everything in memory with no persistence.
Perfect for development, testing, and simple use cases.
"""

from typing import Any, Dict, List, Optional

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryStorage:
    """Simple in-memory storage with no persistence.

    This is the default storage backend for Sifaka. It stores everything in a
    Python dictionary and provides no persistence across process restarts.

    Perfect for:
    - Development and testing
    - Simple scripts and experiments
    - Cases where persistence is not needed

    Attributes:
        data: Dictionary storing all key-value pairs.
    """

    def __init__(self) -> None:
        """Initialize empty memory storage."""
        self.data: Dict[str, Any] = {}
        logger.debug("Initialized MemoryStorage")

    # Async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key asynchronously."""
        return self.get(key)

    async def _set_async(self, key: str, value: Any) -> None:
        """Set a value for a key asynchronously."""
        self.set(key, value)

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query asynchronously."""
        return self.search(query, limit)

    async def _clear_async(self) -> None:
        """Clear all stored data asynchronously."""
        self.clear()

    async def _delete_async(self, key: str) -> bool:
        """Delete a value by key asynchronously."""
        return self.delete(key)

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously."""
        return self.keys()

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        value = self.data.get(key)
        logger.debug(f"Memory get: {key} -> {'found' if value is not None else 'not found'}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key.

        Args:
            key: The storage key.
            value: The value to store.
        """
        self.data[key] = value
        logger.debug(f"Memory set: {key} -> stored")

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        For memory storage, this just returns all values (no semantic search).

        Args:
            query: The search query (ignored for memory storage).
            limit: Maximum number of results to return.

        Returns:
            List of all stored values, limited by the limit parameter.
        """
        values = list(self.data.values())[:limit]
        logger.debug(f"Memory search: '{query}' -> {len(values)} results")
        return values

    def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        if key in self.data:
            del self.data[key]
            logger.debug(f"Memory delete: {key} -> deleted")
            return True
        else:
            logger.debug(f"Memory delete: {key} -> not found")
            return False

    def clear(self) -> None:
        """Clear all stored data."""
        count = len(self.data)
        self.data.clear()
        logger.debug(f"Memory clear: removed {count} items")

    def keys(self) -> List[str]:
        """Get all keys in storage.

        Returns:
            List of all storage keys.
        """
        return list(self.data.keys())

    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        return key in self.data

    def save(self, key: str, value: Any) -> None:
        """Save a value for a key (alias for set).

        Args:
            key: The storage key.
            value: The value to store.
        """
        self.set(key, value)

    def exists(self, key: str) -> bool:
        """Check if key exists in storage (alias for __contains__).

        Args:
            key: The storage key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self.data

    def load(self, key: str) -> Optional[Any]:
        """Load a value by key (alias for get).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return self.get(key)
