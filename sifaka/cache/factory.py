"""
Cache factory for Sifaka.

This module provides a factory function for creating cache instances.
"""

from typing import Any, Dict, Optional, Union

from sifaka.cache.base import Cache
from sifaka.cache.memory import MemoryCache
from sifaka.cache.disk import DiskCache
from sifaka.errors import ConfigurationError


def create_cache(
    cache_type: str = "memory",
    namespace: str = "sifaka",
    **options: Any,
) -> Cache:
    """Create a cache instance.
    
    Args:
        cache_type: The type of cache to create. Supported types: "memory", "disk".
        namespace: The namespace to use for cache keys.
        **options: Additional options to pass to the cache constructor.
            For memory cache:
                max_size: The maximum number of items to store in the cache.
                eviction_policy: The policy to use when the cache is full.
            For disk cache:
                directory: The directory to store cache files in.
                max_size: The maximum size of the cache in bytes.
    
    Returns:
        A cache instance.
        
    Raises:
        ConfigurationError: If the cache type is not supported.
    """
    if cache_type == "memory":
        return MemoryCache(
            namespace=namespace,
            max_size=options.get("max_size"),
            eviction_policy=options.get("eviction_policy", "lru"),
        )
    elif cache_type == "disk":
        return DiskCache(
            namespace=namespace,
            directory=options.get("directory"),
            max_size=options.get("max_size"),
        )
    else:
        raise ConfigurationError(f"Unsupported cache type: {cache_type}")
