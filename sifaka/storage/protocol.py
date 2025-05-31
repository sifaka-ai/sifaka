"""Storage protocol for Sifaka.

This module defines the simple Storage protocol that all storage implementations
must follow. The protocol is designed to be minimal and easy to implement.

The protocol supports both sync and async implementations internally, with sync
methods wrapping async implementations using asyncio.run() for backward compatibility.
"""

import asyncio
from typing import Any, List, Optional, Protocol


def _run_async_safely(coro):
    """Run an async coroutine safely, handling existing event loops.

    This function detects if we're already in an event loop (like when
    PydanticAI is running) and handles the async operation appropriately.
    """
    try:
        # Try to get the current event loop
        asyncio.get_running_loop()
        # We're in an async context - this is not supported for sync methods
        # The caller should use the async version instead
        raise RuntimeError(
            "Cannot call sync storage methods from within an async context. "
            "Use the async methods instead."
        )
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)


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

    # Public sync methods (backward compatible API)
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return _run_async_safely(self._get_async(key))

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key.

        Args:
            key: The storage key.
            value: The value to store.
        """
        return _run_async_safely(self._set_async(key, value))

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        For simple storage backends, this might just return all values.
        For vector storage backends, this performs semantic search.

        Args:
            query: The search query.
            limit: Maximum number of results to return.

        Returns:
            List of matching items.
        """
        return _run_async_safely(self._search_async(query, limit))

    def clear(self) -> None:
        """Clear all stored data."""
        return _run_async_safely(self._clear_async())

    def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        return _run_async_safely(self._delete_async(key))

    def keys(self) -> List[str]:
        """Get all keys in storage.

        Returns:
            List of all storage keys.
        """
        return _run_async_safely(self._keys_async())

    def __len__(self) -> int:
        """Return the number of items in storage.

        Returns:
            The number of items in storage.
        """
        return len(self.keys())
