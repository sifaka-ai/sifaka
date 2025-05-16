"""
Base cache interface for Sifaka.

This module defines the base cache interface that all cache implementations must follow.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, TypeVar, Union, Tuple

from sifaka.errors import CacheError


# Type for cache keys
CacheKey = Union[str, Tuple[str, ...], Dict[str, Any]]

# Type for cache values
T = TypeVar("T")


class Cache(Protocol):
    """Protocol defining the interface for caches.
    
    This protocol defines the interface that all cache implementations must follow.
    """
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The key to get the value for.
            
        Returns:
            The cached value, or None if the key is not in the cache.
            
        Raises:
            CacheError: If there is an error getting the value from the cache.
        """
        ...
    
    def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to set the value for.
            value: The value to cache.
            ttl: Time to live in seconds. If None, the value will not expire.
            
        Raises:
            CacheError: If there is an error setting the value in the cache.
        """
        ...
    
    def delete(self, key: CacheKey) -> None:
        """Delete a value from the cache.
        
        Args:
            key: The key to delete.
            
        Raises:
            CacheError: If there is an error deleting the value from the cache.
        """
        ...
    
    def clear(self) -> None:
        """Clear the cache.
        
        Raises:
            CacheError: If there is an error clearing the cache.
        """
        ...
    
    def contains(self, key: CacheKey) -> bool:
        """Check if a key is in the cache.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key is in the cache, False otherwise.
            
        Raises:
            CacheError: If there is an error checking if the key is in the cache.
        """
        ...


class AbstractCache(ABC):
    """Abstract base class for cache implementations.
    
    This class provides common functionality for cache implementations.
    """
    
    def __init__(self, namespace: str = "sifaka"):
        """Initialize the cache.
        
        Args:
            namespace: The namespace to use for cache keys.
        """
        self.namespace = namespace
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The key to get the value for.
            
        Returns:
            The cached value, or None if the key is not in the cache.
            
        Raises:
            CacheError: If there is an error getting the value from the cache.
        """
        try:
            cache_key = self._make_key(key)
            return self._get(cache_key)
        except Exception as e:
            raise CacheError(f"Error getting value from cache: {str(e)}")
    
    def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to set the value for.
            value: The value to cache.
            ttl: Time to live in seconds. If None, the value will not expire.
            
        Raises:
            CacheError: If there is an error setting the value in the cache.
        """
        try:
            cache_key = self._make_key(key)
            self._set(cache_key, value, ttl)
        except Exception as e:
            raise CacheError(f"Error setting value in cache: {str(e)}")
    
    def delete(self, key: CacheKey) -> None:
        """Delete a value from the cache.
        
        Args:
            key: The key to delete.
            
        Raises:
            CacheError: If there is an error deleting the value from the cache.
        """
        try:
            cache_key = self._make_key(key)
            self._delete(cache_key)
        except Exception as e:
            raise CacheError(f"Error deleting value from cache: {str(e)}")
    
    def clear(self) -> None:
        """Clear the cache.
        
        Raises:
            CacheError: If there is an error clearing the cache.
        """
        try:
            self._clear()
        except Exception as e:
            raise CacheError(f"Error clearing cache: {str(e)}")
    
    def contains(self, key: CacheKey) -> bool:
        """Check if a key is in the cache.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key is in the cache, False otherwise.
            
        Raises:
            CacheError: If there is an error checking if the key is in the cache.
        """
        try:
            cache_key = self._make_key(key)
            return self._contains(cache_key)
        except Exception as e:
            raise CacheError(f"Error checking if key is in cache: {str(e)}")
    
    def _make_key(self, key: CacheKey) -> str:
        """Make a cache key from a key.
        
        Args:
            key: The key to make a cache key from.
            
        Returns:
            A string cache key.
        """
        if isinstance(key, str):
            # If the key is already a string, just use it
            serialized_key = key
        else:
            # Otherwise, serialize the key to JSON
            try:
                serialized_key = json.dumps(key, sort_keys=True)
            except (TypeError, ValueError):
                # If the key can't be serialized to JSON, use its string representation
                serialized_key = str(key)
        
        # Hash the key to ensure it's a valid cache key
        hashed_key = hashlib.md5(serialized_key.encode()).hexdigest()
        
        # Add the namespace to the key
        return f"{self.namespace}:{hashed_key}"
    
    @abstractmethod
    def _get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The key to get the value for.
            
        Returns:
            The cached value, or None if the key is not in the cache.
        """
        pass
    
    @abstractmethod
    def _set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to set the value for.
            value: The value to cache.
            ttl: Time to live in seconds. If None, the value will not expire.
        """
        pass
    
    @abstractmethod
    def _delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: The key to delete.
        """
        pass
    
    @abstractmethod
    def _clear(self) -> None:
        """Clear the cache."""
        pass
    
    @abstractmethod
    def _contains(self, key: str) -> bool:
        """Check if a key is in the cache.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key is in the cache, False otherwise.
        """
        pass
