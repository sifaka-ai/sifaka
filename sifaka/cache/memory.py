"""
Memory cache implementation for Sifaka.

This module provides a memory-based cache implementation.
"""

import time
from typing import Any, Dict, Optional, Tuple

from sifaka.cache.base import AbstractCache


class MemoryCache(AbstractCache):
    """Memory-based cache implementation.
    
    This cache stores values in memory. Values are lost when the process exits.
    
    Attributes:
        namespace: The namespace to use for cache keys.
        max_size: The maximum number of items to store in the cache.
        eviction_policy: The policy to use when the cache is full.
    """
    
    def __init__(
        self,
        namespace: str = "sifaka",
        max_size: Optional[int] = None,
        eviction_policy: str = "lru",
    ):
        """Initialize the memory cache.
        
        Args:
            namespace: The namespace to use for cache keys.
            max_size: The maximum number of items to store in the cache.
                If None, the cache has no size limit.
            eviction_policy: The policy to use when the cache is full.
                Supported policies: "lru" (least recently used), "fifo" (first in, first out).
        
        Raises:
            ValueError: If the eviction policy is not supported.
        """
        super().__init__(namespace)
        
        if eviction_policy not in ["lru", "fifo"]:
            raise ValueError(f"Unsupported eviction policy: {eviction_policy}")
        
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self._cache: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._access_times: Dict[str, float] = {}
        self._insertion_order: list = []
    
    def _get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The key to get the value for.
            
        Returns:
            The cached value, or None if the key is not in the cache or has expired.
        """
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        # Check if the value has expired
        if expiry is not None and time.time() > expiry:
            # Value has expired, remove it from the cache
            self._delete(key)
            return None
        
        # Update access time for LRU
        if self.eviction_policy == "lru":
            self._access_times[key] = time.time()
        
        return value
    
    def _set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to set the value for.
            value: The value to cache.
            ttl: Time to live in seconds. If None, the value will not expire.
        """
        # Calculate expiry time if TTL is provided
        expiry = time.time() + ttl if ttl is not None else None
        
        # Check if we need to evict an item
        if self.max_size is not None and len(self._cache) >= self.max_size and key not in self._cache:
            self._evict()
        
        # Store the value
        self._cache[key] = (value, expiry)
        
        # Update metadata for eviction policies
        if self.eviction_policy == "lru":
            self._access_times[key] = time.time()
        
        if self.eviction_policy == "fifo" and key not in self._insertion_order:
            self._insertion_order.append(key)
    
    def _delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: The key to delete.
        """
        if key in self._cache:
            del self._cache[key]
        
        # Clean up metadata
        if key in self._access_times:
            del self._access_times[key]
        
        if key in self._insertion_order:
            self._insertion_order.remove(key)
    
    def _clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_times.clear()
        self._insertion_order.clear()
    
    def _contains(self, key: str) -> bool:
        """Check if a key is in the cache.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key is in the cache and has not expired, False otherwise.
        """
        if key not in self._cache:
            return False
        
        _, expiry = self._cache[key]
        
        # Check if the value has expired
        if expiry is not None and time.time() > expiry:
            # Value has expired, remove it from the cache
            self._delete(key)
            return False
        
        return True
    
    def _evict(self) -> None:
        """Evict an item from the cache based on the eviction policy."""
        if not self._cache:
            return
        
        if self.eviction_policy == "lru":
            # Find the least recently used item
            lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            self._delete(lru_key)
        elif self.eviction_policy == "fifo":
            # Remove the first item inserted
            if self._insertion_order:
                fifo_key = self._insertion_order[0]
                self._delete(fifo_key)
