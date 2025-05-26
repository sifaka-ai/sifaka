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
    """Layered storage combining memory cache with persistent storage.

    This storage implementation combines a fast cache (usually MemoryStorage)
    with a persistent backend (FileStorage, RedisStorage, or MilvusStorage).

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

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key, checking cache first then persistence.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        # Try cache first (fast)
        if self.cache is not None:
            value = self.cache.get(key)
            if value is not None:
                logger.debug(f"Cached get: {key} -> cache hit")
                return value

        # Try persistence (slower)
        if self.persistence is not None:
            value = self.persistence.get(key)
            if value is not None:
                # Cache the value for next time
                if self.cache is not None:
                    self.cache.set(key, value)
                logger.debug(f"Cached get: {key} -> persistence hit, cached")
                return value

        logger.debug(f"Cached get: {key} -> miss")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key in both cache and persistence.

        Args:
            key: The storage key.
            value: The value to store.
        """
        # Write to cache immediately (fast)
        if self.cache is not None:
            self.cache.set(key, value)

        # Write to persistence (may be slower)
        if self.persistence is not None:
            self.persistence.set(key, value)

        logger.debug(f"Cached set: {key} -> stored in available backends")

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        Prefers persistence for search (especially for vector search),
        but falls back to cache if persistence is not available.

        Args:
            query: The search query.
            limit: Maximum number of results to return.

        Returns:
            List of matching items.
        """
        # Prefer persistence for search (especially vector search)
        if self.persistence is not None:
            results = self.persistence.search(query, limit)
            logger.debug(f"Cached search: '{query}' -> {len(results)} results from persistence")
            return results

        # Fall back to cache
        if self.cache is not None:
            results = self.cache.search(query, limit)
            logger.debug(f"Cached search: '{query}' -> {len(results)} results from cache")
            return results

        logger.debug(f"Cached search: '{query}' -> no backends available")
        return []

    def clear(self) -> None:
        """Clear all data from both cache and persistence."""
        cleared = []

        if self.cache is not None:
            self.cache.clear()
            cleared.append("cache")

        if self.persistence is not None:
            self.persistence.clear()
            cleared.append("persistence")

        logger.debug(f"Cached clear: cleared {', '.join(cleared)}")

    def __len__(self) -> int:
        """Return number of items in cache, or persistence if no cache."""
        if self.cache and hasattr(self.cache, "__len__"):
            return len(self.cache)
        elif self.persistence and hasattr(self.persistence, "__len__"):
            return len(self.persistence)
        else:
            return 0

    def __contains__(self, key: str) -> bool:
        """Check if key exists in either cache or persistence."""
        if self.cache and hasattr(self.cache, "__contains__"):
            if key in self.cache:
                return True

        if self.persistence and hasattr(self.persistence, "__contains__"):
            return key in self.persistence

        # Fall back to get() method
        return self.get(key) is not None

    def save(self, key: str, value: Any) -> None:
        """Save a value for a key (same as set).

        Args:
            key: The storage key.
            value: The value to store.
        """
        self.set(key, value)

    def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: The storage key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self
