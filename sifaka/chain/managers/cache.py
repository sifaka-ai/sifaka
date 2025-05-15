"""
Cache Manager Module

This module provides a cache manager for the Sifaka chain system.
It handles caching of chain results to improve performance.

## Overview
The cache manager provides a standardized way to cache chain results,
reducing redundant computation and improving response times for repeated
prompts. It uses the state management system to store and retrieve cached
results, with configurable cache size and enabling/disabling options.

## Components
1. **CacheManager**: Manages caching of chain results with configurable
   cache size and statistics tracking

## Usage Examples
```python
from sifaka.chain.managers.cache import CacheManager

# Create cache manager
cache_manager = CacheManager(
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

# Get cache statistics
stats = cache_manager.get_cache_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache entries: {stats['cache_entries']}")

# Clear cache when needed
cache_manager.clear_cache()
```

## Error Handling
The cache manager is designed to be robust and handle edge cases gracefully:
- Cache misses return None rather than raising exceptions
- Cache operations are no-ops when caching is disabled
- Cache size limits are enforced by removing oldest entries

## Configuration
The cache manager can be configured with the following options:
- cache_enabled: Whether caching is enabled (default: True)
- cache_size: Maximum number of cached results (default: 100)
"""

from typing import Dict, Optional, Any
import time
from pydantic import PrivateAttr
from ...utils.state import StateManager, create_manager_state
from ...utils.logging import get_logger
from ...core.results import ChainResult
from ...utils.mixins import InitializeStateMixin

logger = get_logger(__name__)


class CacheManager(InitializeStateMixin):
    """
    Manages caching of chain results.

    This class provides a standardized way to cache chain results, reducing
    redundant computation and improving response times for repeated prompts.
    It uses the state management system to store and retrieve cached results,
    with configurable cache size and enabling/disabling options.

    ## Architecture
    The CacheManager uses the StateManager for storing cache data, ensuring
    consistent state management across the system. It implements an LRU-like
    cache eviction policy, removing the oldest entries when the cache is full.

    ## Lifecycle
    1. **Initialization**: Set up cache with state manager and configuration
    2. **Operation**: Cache results and retrieve cached results
    3. **Maintenance**: Clear cache and track statistics

    ## Error Handling
    The CacheManager handles edge cases gracefully:
    - Cache misses return None rather than raising exceptions
    - Cache operations are no-ops when caching is disabled
    - Cache size limits are enforced by removing oldest entries

    ## Examples
    ```python
    # Create cache manager
    cache_manager = CacheManager(
        cache_enabled=True,
        cache_size=100
    )

    # Use cache for results
    if cache_manager.has_cached_result(prompt):
        result = cache_manager.get_cached_result(prompt)
    else:
        result = generate_result(prompt)
        cache_manager.cache_result(prompt, result)
    ```
    """

    _state_manager: StateManager = PrivateAttr(default_factory=create_manager_state)

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_size: int = 100,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the cache manager.

        This method initializes the cache manager with the provided configuration
        options. It sets up the initial cache state and metadata.

        Args:
            cache_enabled (bool, optional): Whether caching is enabled. Defaults to True.
            cache_size (int, optional): Maximum number of cached results. Defaults to 100.
            state_manager (Optional[StateManager], optional): State manager for state management.
                If None, a new state manager will be created. Defaults to None.

        Raises:
            None: This method does not raise exceptions

        Example:
            ```python
            from sifaka.chain.managers.cache import CacheManager

            # Create cache manager
            cache_manager = CacheManager(
                cache_enabled=True,
                cache_size=100
            )
            ```
        """
        self._cache_enabled = cache_enabled
        self._cache_size = cache_size

        # Support both dependency injection and auto-creation patterns
        if state_manager is not None:
            object.__setattr__(self, "_state_manager", state_manager)

        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize the cache manager state."""
        # Check if super has _initialize_state method before calling it
        if hasattr(super(), "_initialize_state"):
            super()._initialize_state()

        if not self._state_manager.get("result_cache"):
            self._state_manager.update("result_cache", {})

        self._state_manager.update("initialized", True)
        self._state_manager.set_metadata("component_type", "cache_manager")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("cache_enabled", self._cache_enabled)
        self._state_manager.set_metadata("cache_size", self._cache_size)

    @property
    def cache_enabled(self) -> bool:
        """
        Get whether caching is enabled.

        Returns:
            bool: True if caching is enabled, False otherwise

        Example:
            ```python
            if cache_manager.cache_enabled:
                print("Caching is enabled")
            ```
        """
        return self._cache_enabled

    @property
    def cache_size(self) -> int:
        """
        Get the maximum cache size.

        Returns:
            int: The maximum number of entries the cache can hold

        Example:
            ```python
            print(f"Cache can hold up to {cache_manager.cache_size} entries")
            ```
        """
        return self._cache_size

    def has_cached_result(self, prompt: str) -> bool:
        """
        Check if a result is cached for the given prompt.

        This method checks if a result is already cached for the given prompt.
        It returns False if caching is disabled.

        Args:
            prompt (str): The prompt to check for in the cache

        Returns:
            bool: True if a result is cached, False otherwise

        Example:
            ```python
            if cache_manager.has_cached_result(prompt):
                result = cache_manager.get_cached_result(prompt)
            else:
                result = generate_result(prompt)
            ```
        """
        if not self._cache_enabled:
            return False
        cache = self._state_manager.get("result_cache", {})
        return prompt in cache

    def get_cached_result(self, prompt: str) -> Optional[ChainResult]:
        """
        Get a cached result for the given prompt.

        This method retrieves a cached result for the given prompt if available.
        It returns None if caching is disabled or if the prompt is not in the cache.
        It also tracks cache hit statistics.

        Args:
            prompt (str): The prompt to get the result for

        Returns:
            Optional[ChainResult]: The cached result, or None if not found

        Raises:
            None: This method does not raise exceptions

        Example:
            ```python
            result = cache_manager.get_cached_result(prompt)
            if result:
                print(f"Using cached result: {result.output}")
            else:
                print("No cached result found")
            ```
        """
        if not self._cache_enabled:
            return None
        cache = self._state_manager.get("result_cache", {})
        result = cache.get(prompt)
        if result:
            self._state_manager.set_metadata("cache_hit", True)
            cache_hits = self._state_manager.get_metadata("cache_hits", 0)
            self._state_manager.set_metadata("cache_hits", cache_hits + 1)
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")

            # Ensure we're returning a ChainResult
            if isinstance(result, ChainResult):
                return result
            else:
                logger.warning(f"Cached result for prompt '{prompt[:50]}...' is not a ChainResult")
                return None
        return None

    def cache_result(self, prompt: str, result: ChainResult) -> None:
        """
        Cache a result for the given prompt.

        This method caches a result for the given prompt. If the cache is full,
        it removes the oldest entry before adding the new one. It does nothing
        if caching is disabled.

        Args:
            prompt (str): The prompt to cache the result for
            result (ChainResult): The result to cache

        Raises:
            None: This method does not raise exceptions

        Example:
            ```python
            result = generate_result(prompt)
            cache_manager.cache_result(prompt, result)
            ```
        """
        if not self._cache_enabled:
            return
        cache = self._state_manager.get("result_cache", {})
        if len(cache) >= self._cache_size:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
            logger.debug(f"Cache full, removed oldest entry: {oldest_key[:50]}...")
        cache[prompt] = result
        self._state_manager.update("result_cache", cache)
        logger.debug(f"Cached result for prompt: {prompt[:50]}...")

    def clear_cache(self) -> None:
        """
        Clear the cache.

        This method removes all entries from the cache, effectively
        resetting it to an empty state.

        Returns:
            None

        Example:
            ```python
            # Clear the cache when needed
            cache_manager.clear_cache()
            ```
        """
        self._state_manager.update("result_cache", {})
        logger.debug("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        This method returns a dictionary with various cache statistics,
        including whether caching is enabled, the maximum cache size,
        the current number of cache entries, and hit/miss counts.

        Returns:
            Dict[str, Any]: Dictionary with cache statistics including:
                - cache_enabled: Whether caching is enabled
                - cache_size: Maximum number of cached results
                - cache_entries: Current number of entries in the cache
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses

        Example:
            ```python
            stats = cache_manager.get_cache_stats()
            print(f"Cache hits: {stats['cache_hits']}")
            print(f"Cache entries: {stats['cache_entries']}")
            print(f"Cache hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.2f}")
            ```
        """
        cache = self._state_manager.get("result_cache", {})
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": self._cache_size,
            "cache_entries": len(cache),
            "cache_hits": self._state_manager.get_metadata("cache_hits", 0),
            "cache_misses": self._state_manager.get_metadata("cache_misses", 0),
        }
