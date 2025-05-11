"""
Cache Manager Module

This module provides a cache manager for the Sifaka chain system.
It handles caching of chain results to improve performance.

## Components
1. **CacheManager**: Manages caching of chain results

## Usage Examples
```python
from sifaka.chain.managers.cache import CacheManager
from sifaka.utils.state import StateManager

# Create cache manager
cache_manager = CacheManager(
    state_manager=StateManager(),
    cache_enabled=True,
    cache_size=100
)

# Check if result is in cache
if cache_manager.has_cached_result(prompt):
    result = cache_manager.get_cached_result(prompt)
    print(f"Cache hit: {result.output}")
else:
    # Generate result and cache it
    result = generate_result(prompt)
    cache_manager.cache_result(prompt, result)
```
"""

from typing import Dict, Optional, Any
import time

from ...utils.state import StateManager
from ...utils.logging import get_logger
from ..result import ChainResult

# Configure logger
logger = get_logger(__name__)


class CacheManager:
    """Manages caching of chain results."""

    def __init__(
        self,
        state_manager: StateManager,
        cache_enabled: bool = True,
        cache_size: int = 100,
    ):
        """
        Initialize the cache manager.

        Args:
            state_manager: State manager for state management
            cache_enabled: Whether caching is enabled
            cache_size: Maximum number of cached results
        """
        self._state_manager = state_manager
        self._cache_enabled = cache_enabled
        self._cache_size = cache_size

        # Initialize cache in state
        if not self._state_manager.get("result_cache"):
            self._state_manager.update("result_cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "cache_manager")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("cache_enabled", cache_enabled)
        self._state_manager.set_metadata("cache_size", cache_size)

    @property
    def cache_enabled(self) -> bool:
        """Get whether caching is enabled."""
        return self._cache_enabled

    @property
    def cache_size(self) -> int:
        """Get the maximum cache size."""
        return self._cache_size

    def has_cached_result(self, prompt: str) -> bool:
        """
        Check if a result is cached for the given prompt.

        Args:
            prompt: The prompt to check

        Returns:
            True if a result is cached, False otherwise
        """
        if not self._cache_enabled:
            return False

        cache = self._state_manager.get("result_cache", {})
        return prompt in cache

    def get_cached_result(self, prompt: str) -> Optional[ChainResult]:
        """
        Get a cached result for the given prompt.

        Args:
            prompt: The prompt to get the result for

        Returns:
            The cached result, or None if not found
        """
        if not self._cache_enabled:
            return None

        cache = self._state_manager.get("result_cache", {})
        result = cache.get(prompt)

        if result:
            # Track cache hit
            self._state_manager.set_metadata("cache_hit", True)
            cache_hits = self._state_manager.get_metadata("cache_hits", 0)
            self._state_manager.set_metadata("cache_hits", cache_hits + 1)
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")

        return result

    def cache_result(self, prompt: str, result: ChainResult) -> None:
        """
        Cache a result for the given prompt.

        Args:
            prompt: The prompt to cache the result for
            result: The result to cache
        """
        if not self._cache_enabled:
            return

        cache = self._state_manager.get("result_cache", {})

        # If cache is full, remove oldest entry
        if len(cache) >= self._cache_size:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
            logger.debug(f"Cache full, removed oldest entry: {oldest_key[:50]}...")

        # Add result to cache
        cache[prompt] = result
        self._state_manager.update("result_cache", cache)
        logger.debug(f"Cached result for prompt: {prompt[:50]}...")

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._state_manager.update("result_cache", {})
        logger.debug("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache = self._state_manager.get("result_cache", {})
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": self._cache_size,
            "cache_entries": len(cache),
            "cache_hits": self._state_manager.get_metadata("cache_hits", 0),
            "cache_misses": self._state_manager.get_metadata("cache_misses", 0),
        }
