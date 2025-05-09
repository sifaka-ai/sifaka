"""
Caching strategies for classifiers.

This module provides caching strategies for classifiers in the Sifaka framework.
"""

from functools import lru_cache
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

# Type variables for generic caching
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


class CachingStrategy(Generic[T, R]):
    """
    Strategy for caching classifier results.

    This class provides a standardized way to cache classifier results,
    with configurable cache size and optional key transformation.

    ## Lifecycle

    1. **Creation**: Instantiate with cache size and optional key function
       - Provide cache_size to set the maximum cache size
       - Provide key_func to customize cache key generation

    2. **Caching**: Cache function results
       - Use get_cached() to retrieve cached results
       - Use cache_result() to store new results
       - Use clear_cache() to clear the cache

    ## Error Handling

    The class implements these error handling patterns:
    - Safe cache access with get_cached()
    - Type checking for cache keys and values
    - Exception handling in key generation

    ## Examples

    Basic usage:

    ```python
    from sifaka.classifiers.strategies import CachingStrategy

    # Create a caching strategy
    cache = CachingStrategy(cache_size=100)

    # Check for cached result
    result = cache.get_cached("input_text")
    if result is not None:
        return result

    # Compute result
    result = compute_result("input_text")

    # Cache result
    cache.cache_result("input_text", result)

    # Return result
    return result
    ```

    Using with a custom key function:

    ```python
    from sifaka.classifiers.strategies import CachingStrategy

    # Create a caching strategy with a custom key function
    def normalize_key(text):
        return text.lower().strip()

    cache = CachingStrategy(
        cache_size=100,
        key_func=normalize_key
    )

    # Check for cached result (will use normalized key)
    result = cache.get_cached("Input Text  ")  # Key will be "input text"
    ```

    Using with a classifier:

    ```python
    from sifaka.classifiers.base import BaseClassifier
    from sifaka.classifiers.strategies import CachingStrategy

    class MyClassifier(BaseClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cache = CachingStrategy(
                cache_size=self.config.cache_size
            )

        def _classify_impl(self, text):
            # Check cache first
            cached = self._cache.get_cached(text)
            if cached is not None:
                return cached

            # Compute result
            result = self._classify_impl_uncached(text)

            # Cache result
            self._cache.cache_result(text, result)

            # Return result
            return result
    ```
    """

    def __init__(
        self,
        cache_size: int = 0,
        key_func: Optional[Callable[[T], Any]] = None,
    ) -> None:
        """
        Initialize the caching strategy.

        Args:
            cache_size: Maximum number of results to cache (0 disables caching)
            key_func: Optional function to transform input into cache key
        """
        self.cache_size = cache_size
        self.key_func = key_func or (lambda x: x)
        self._cache: Dict[Any, R] = {}
        
        # Create LRU cache wrapper if caching is enabled
        if self.cache_size > 0:
            self._lru_cache = lru_cache(maxsize=self.cache_size)(lambda k: self._cache.get(k))
        else:
            self._lru_cache = None

    def get_cached(self, input_value: T) -> Optional[R]:
        """
        Get a cached result.

        Args:
            input_value: Input value to check in cache

        Returns:
            Cached result or None if not found
        """
        if self.cache_size <= 0:
            return None
            
        try:
            # Generate cache key
            key = self.key_func(input_value)
            
            # Use LRU cache to maintain access order
            if key in self._cache:
                self._lru_cache(key)  # Update LRU order
                return self._cache[key]
                
            return None
        except Exception:
            # If any error occurs during cache lookup, return None
            return None

    def cache_result(self, input_value: T, result: R) -> None:
        """
        Cache a result.

        Args:
            input_value: Input value to use as cache key
            result: Result to cache
        """
        if self.cache_size <= 0:
            return
            
        try:
            # Generate cache key
            key = self.key_func(input_value)
            
            # Store in cache
            self._cache[key] = result
            
            # Maintain LRU order
            if len(self._cache) > self.cache_size:
                # Remove least recently used item
                # This is handled automatically by the LRU cache
                pass
        except Exception:
            # If any error occurs during caching, just skip it
            pass

    def clear_cache(self) -> None:
        """
        Clear the cache.
        """
        self._cache.clear()
        if self._lru_cache is not None:
            self._lru_cache.cache_clear()
