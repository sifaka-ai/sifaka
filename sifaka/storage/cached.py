"""Cached storage implementation.

Combines multiple storage backends for layered caching. Typically used to combine
fast in-memory storage with persistent storage backends.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from sifaka.storage.protocol import Storage

logger = get_logger(__name__)


class CachedStorage:
    """Async layered storage combining memory cache with persistent storage.

    This storage implementation combines a fast cache (usually MemoryStorage)
    with a persistent backend (FileStorage or RedisStorage).

    All operations are async to properly handle modern storage backends.

    Read strategy:
    1. Check cache first (fast)
    2. If not found, check persistence (slower)
    3. If found in persistence, cache it for next time

    Write strategy:
    1. Write to cache immediately (fast)
    2. Write to persistence (may be slower)

    Perfect for:
    - Combining speed with persistence
    - Frequently accessed data that needs to survive restarts
    - Applications that need both fast access and durability

    Attributes:
        cache: Fast storage backend (usually MemoryStorage).
        persistence: Persistent storage backend.
    """

    def __init__(self, cache: Optional["Storage"], persistence: Optional["Storage"]):
        """Initialize cached storage.

        Args:
            cache: Fast cache storage (e.g., MemoryStorage).
            persistence: Persistent storage (e.g., FileStorage, RedisStorage).
        """
        self.cache = cache
        self.persistence = persistence

        if cache is None and persistence is None:
            raise ValueError("At least one of cache or persistence must be provided")

        backends = []
        if cache:
            backends.append("cache")
        if persistence:
            backends.append("persistence")

        logger.debug(f"Initialized CachedStorage with: {', '.join(backends)}")

    # Internal async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key asynchronously, checking cache first then persistence.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        # Try cache first (fast) - async only
        if self.cache is not None:
            if hasattr(self.cache, "_get_async"):
                value = await self.cache._get_async(key)
            elif hasattr(self.cache, "get"):
                # Try async get method
                result = self.cache.get(key)
                if hasattr(result, "__await__"):
                    value = await result
                else:
                    value = result
            else:
                value = None
            if value is not None:
                logger.debug(f"Cached get async: {key} -> cache hit")
                return value

        # Try persistence (slower) - async only
        if self.persistence is not None:
            if hasattr(self.persistence, "_get_async"):
                value = await self.persistence._get_async(key)
            elif hasattr(self.persistence, "get"):
                # Try async get method
                result = self.persistence.get(key)
                if hasattr(result, "__await__"):
                    value = await result
                else:
                    value = result
            else:
                value = None
            if value is not None:
                # Cache the value for next time - async only
                if self.cache is not None:
                    if hasattr(self.cache, "_set_async"):
                        await self.cache._set_async(key, value)
                    elif hasattr(self.cache, "set"):
                        # Try async set method
                        result = self.cache.set(key, value)
                        if hasattr(result, "__await__"):
                            await result
                logger.debug(f"Cached get async: {key} -> persistence hit, cached")
                return value

        logger.debug(f"Cached get async: {key} -> miss")
        return None

    async def _set_async(self, key: str, value: Any) -> None:
        """Set a value for a key asynchronously in both cache and persistence.

        Args:
            key: The storage key.
            value: The value to store.
        """
        # Write to cache immediately (fast) - async only
        if self.cache is not None:
            if hasattr(self.cache, "_set_async"):
                await self.cache._set_async(key, value)
            elif hasattr(self.cache, "set"):
                # Try async set method
                result = self.cache.set(key, value)
                if hasattr(result, "__await__"):
                    await result
            else:
                logger.warning(f"Cache {type(self.cache)} has no set method")

        # Write to persistence (may be slower) - async only
        if self.persistence is not None:
            if hasattr(self.persistence, "_set_async"):
                await self.persistence._set_async(key, value)
            elif hasattr(self.persistence, "set"):
                # Try async set method
                result = self.persistence.set(key, value)
                if hasattr(result, "__await__"):
                    await result
            else:
                logger.warning(f"Persistence {type(self.persistence)} has no set method")

        logger.debug(f"Cached set async: {key} -> stored in available backends")

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query asynchronously.

        Args:
            query: The search query.
            limit: Maximum number of results to return.

        Returns:
            List of matching items.
        """
        # Prefer persistence for search (especially vector search)
        if self.persistence is not None:
            if hasattr(self.persistence, "_search_async"):
                results = await self.persistence._search_async(query, limit)
            else:
                results = self.persistence.search(query, limit)
            logger.debug(
                f"Cached search async: '{query}' -> {len(results)} results from persistence"
            )
            return results

        # Fall back to cache
        if self.cache is not None:
            if hasattr(self.cache, "_search_async"):
                results = await self.cache._search_async(query, limit)
            else:
                results = self.cache.search(query, limit)
            logger.debug(f"Cached search async: '{query}' -> {len(results)} results from cache")
            return results

        logger.debug(f"Cached search async: '{query}' -> no backends available")
        return []

    async def _clear_async(self) -> None:
        """Clear all stored data asynchronously from both cache and persistence."""
        if self.cache is not None:
            if hasattr(self.cache, "_clear_async"):
                await self.cache._clear_async()
            else:
                self.cache.clear()

        if self.persistence is not None:
            if hasattr(self.persistence, "_clear_async"):
                await self.persistence._clear_async()
            else:
                self.persistence.clear()

        logger.debug("Cached clear async: cleared all backends")

    async def _delete_async(self, key: str) -> bool:
        """Delete a value by key asynchronously from both cache and persistence.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted from at least one backend, False otherwise.
        """
        deleted = False

        if self.cache is not None:
            if hasattr(self.cache, "_delete_async"):
                cache_deleted = await self.cache._delete_async(key)
            else:
                cache_deleted = self.cache.delete(key) if hasattr(self.cache, "delete") else False
            deleted = deleted or cache_deleted

        if self.persistence is not None:
            if hasattr(self.persistence, "_delete_async"):
                persistence_deleted = await self.persistence._delete_async(key)
            else:
                persistence_deleted = (
                    self.persistence.delete(key) if hasattr(self.persistence, "delete") else False
                )
            deleted = deleted or persistence_deleted

        logger.debug(f"Cached delete async: {key} -> {'deleted' if deleted else 'not found'}")
        return deleted

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously from both cache and persistence.

        Returns:
            List of all storage keys (deduplicated).
        """
        all_keys = set()

        if self.cache is not None:
            if hasattr(self.cache, "_keys_async"):
                cache_keys = await self.cache._keys_async()
            else:
                cache_keys = self.cache.keys() if hasattr(self.cache, "keys") else []
            all_keys.update(cache_keys)

        if self.persistence is not None:
            if hasattr(self.persistence, "_keys_async"):
                persistence_keys = await self.persistence._keys_async()
            else:
                persistence_keys = (
                    self.persistence.keys() if hasattr(self.persistence, "keys") else []
                )
            all_keys.update(persistence_keys)

        return list(all_keys)

    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key asynchronously, checking cache first then persistence.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return await self._get_async(key)

    async def set(self, key: str, value: Any) -> None:
        """Set a value for a key asynchronously in both cache and persistence.

        Args:
            key: The storage key.
            value: The value to store.
        """
        await self._set_async(key, value)

    async def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query asynchronously.

        Args:
            query: The search query.
            limit: Maximum number of results to return.

        Returns:
            List of matching items.
        """
        return await self._search_async(query, limit)

    async def clear(self) -> None:
        """Clear all data from both cache and persistence asynchronously."""
        await self._clear_async()

    def __len__(self) -> int:
        """Return number of items in cache, or persistence if no cache."""
        if self.cache and hasattr(self.cache, "__len__"):
            try:
                return len(self.cache)
            except TypeError:
                # Handle async __len__ methods
                return 0
        elif self.persistence and hasattr(self.persistence, "__len__"):
            try:
                result = len(self.persistence)
                # Check if it's a coroutine (async method)
                if hasattr(result, "__await__"):
                    return 0  # Can't await in sync method
                return result
            except TypeError:
                # Handle async __len__ methods
                return 0
        else:
            return 0

    def __contains__(self, key: str) -> bool:
        """Check if key exists in either cache or persistence."""
        if self.cache and hasattr(self.cache, "__contains__"):
            if key in self.cache:
                return True

        if self.persistence and hasattr(self.persistence, "__contains__"):
            return key in self.persistence

        # Fall back to checking if we can get the value
        # Note: This is a sync method so we can't await, return False for safety
        return False

    async def save(self, key: str, value: Any) -> None:
        """Save a value for a key asynchronously (same as set).

        Args:
            key: The storage key.
            value: The value to store.
        """
        await self.set(key, value)

    def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: The storage key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self

    async def load(self, key: str) -> Optional[Any]:
        """Load a value by key asynchronously (alias for get).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return await self.get(key)

    async def delete(self, key: str) -> bool:
        """Delete a value by key asynchronously from both cache and persistence.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted from at least one backend, False otherwise.
        """
        return await self._delete_async(key)
