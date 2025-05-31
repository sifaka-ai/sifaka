"""Storage protocol for Sifaka.

This module defines the simple Storage protocol that all storage implementations
must follow. The protocol is designed to be minimal and easy to implement.

All storage implementations are now async-only for consistency with PydanticAI.
"""

from typing import Any, List, Optional, Protocol


class Storage(Protocol):
    """Simple storage protocol for Sifaka.

    All storage implementations must provide these four methods:
    - get: Retrieve a value by key
    - set: Store a value with a key
    - search: Search for items (for vector/semantic search)
    - clear: Remove all stored data

    The protocol supports both sync and async implementations internally.
    """

    # Internal async methods (to be implemented by storage backends)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key asynchronously (internal method).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        ...

    async def _set_async(self, key: str, value: Any) -> None:
        """Set a value for a key asynchronously (internal method).

        Args:
            key: The storage key.
            value: The value to store.
        """
        ...

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query asynchronously (internal method).

        For simple storage backends, this might just return all values.
        For vector storage backends, this performs semantic search.

        Args:
            query: The search query.
            limit: Maximum number of results to return.

        Returns:
            List of matching items.
        """
        ...

    async def _clear_async(self) -> None:
        """Clear all stored data asynchronously (internal method)."""
        ...

    async def _delete_async(self, key: str) -> bool:
        """Delete a value by key asynchronously (internal method).

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        ...

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously (internal method).

        Returns:
            List of all storage keys.
        """
        ...

    # Public async-only API
    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return await self._get_async(key)

    async def set(self, key: str, value: Any) -> None:
        """Set a value for a key.

        Args:
            key: The storage key.
            value: The value to store.
        """
        return await self._set_async(key, value)

    async def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        For simple storage backends, this might just return all values.
        For vector storage backends, this performs semantic search.

        Args:
            query: The search query.
            limit: Maximum number of results to return.

        Returns:
            List of matching items.
        """
        return await self._search_async(query, limit)

    async def clear(self) -> None:
        """Clear all stored data."""
        return await self._clear_async()

    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        return await self._delete_async(key)

    async def keys(self) -> List[str]:
        """Get all keys in storage.

        Returns:
            List of all storage keys.
        """
        return await self._keys_async()

    async def __len__(self) -> int:
        """Return the number of items in storage.

        Returns:
            The number of items in storage.
        """
        keys = await self.keys()
        return len(keys)
