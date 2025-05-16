"""
Disk cache implementation for Sifaka.

This module provides a disk-based cache implementation.
"""

import os
import pickle
import time
import tempfile
import shutil
from typing import Any, Dict, Optional, Tuple

from sifaka.cache.base import AbstractCache
from sifaka.errors import CacheError


class DiskCache(AbstractCache):
    """Disk-based cache implementation.
    
    This cache stores values on disk. Values persist between process restarts.
    
    Attributes:
        namespace: The namespace to use for cache keys.
        directory: The directory to store cache files in.
        max_size: The maximum size of the cache in bytes.
    """
    
    def __init__(
        self,
        namespace: str = "sifaka",
        directory: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        """Initialize the disk cache.
        
        Args:
            namespace: The namespace to use for cache keys.
            directory: The directory to store cache files in.
                If None, a temporary directory is used.
            max_size: The maximum size of the cache in bytes.
                If None, the cache has no size limit.
        
        Raises:
            CacheError: If the cache directory cannot be created.
        """
        super().__init__(namespace)
        
        self.max_size = max_size
        
        # Set up the cache directory
        if directory is None:
            self.directory = os.path.join(tempfile.gettempdir(), f"sifaka_cache_{namespace}")
        else:
            self.directory = directory
        
        # Create the cache directory if it doesn't exist
        try:
            os.makedirs(self.directory, exist_ok=True)
        except Exception as e:
            raise CacheError(f"Error creating cache directory: {str(e)}")
        
        # Create the metadata file if it doesn't exist
        self.metadata_file = os.path.join(self.directory, "metadata.pickle")
        if not os.path.exists(self.metadata_file):
            self._save_metadata({})
    
    def _get_cache_path(self, key: str) -> str:
        """Get the path to a cache file.
        
        Args:
            key: The cache key.
            
        Returns:
            The path to the cache file.
        """
        return os.path.join(self.directory, key)
    
    def _get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The key to get the value for.
            
        Returns:
            The cached value, or None if the key is not in the cache or has expired.
        """
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        # Get metadata for the key
        metadata = self._load_metadata()
        key_metadata = metadata.get(key)
        
        if key_metadata is None:
            # No metadata for this key, remove the file
            try:
                os.remove(cache_path)
            except:
                pass
            return None
        
        # Check if the value has expired
        expiry = key_metadata.get("expiry")
        if expiry is not None and time.time() > expiry:
            # Value has expired, remove it from the cache
            self._delete(key)
            return None
        
        # Load the value from the cache file
        try:
            with open(cache_path, "rb") as f:
                value = pickle.load(f)
            
            # Update access time
            metadata[key]["last_access"] = time.time()
            self._save_metadata(metadata)
            
            return value
        except Exception as e:
            # Error loading the value, remove it from the cache
            self._delete(key)
            return None
    
    def _set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to set the value for.
            value: The value to cache.
            ttl: Time to live in seconds. If None, the value will not expire.
        """
        cache_path = self._get_cache_path(key)
        
        # Calculate expiry time if TTL is provided
        expiry = time.time() + ttl if ttl is not None else None
        
        # Check if we need to evict items
        if self.max_size is not None:
            self._ensure_space(value)
        
        # Save the value to the cache file
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            raise CacheError(f"Error saving value to cache: {str(e)}")
        
        # Update metadata
        metadata = self._load_metadata()
        metadata[key] = {
            "expiry": expiry,
            "created": time.time(),
            "last_access": time.time(),
            "size": os.path.getsize(cache_path),
        }
        self._save_metadata(metadata)
    
    def _delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: The key to delete.
        """
        cache_path = self._get_cache_path(key)
        
        # Remove the cache file
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                raise CacheError(f"Error deleting cache file: {str(e)}")
        
        # Update metadata
        metadata = self._load_metadata()
        if key in metadata:
            del metadata[key]
            self._save_metadata(metadata)
    
    def _clear(self) -> None:
        """Clear the cache."""
        # Remove all cache files
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if os.path.isfile(file_path) and filename != "metadata.pickle":
                try:
                    os.remove(file_path)
                except:
                    pass
        
        # Clear metadata
        self._save_metadata({})
    
    def _contains(self, key: str) -> bool:
        """Check if a key is in the cache.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key is in the cache and has not expired, False otherwise.
        """
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return False
        
        # Get metadata for the key
        metadata = self._load_metadata()
        key_metadata = metadata.get(key)
        
        if key_metadata is None:
            # No metadata for this key, remove the file
            try:
                os.remove(cache_path)
            except:
                pass
            return False
        
        # Check if the value has expired
        expiry = key_metadata.get("expiry")
        if expiry is not None and time.time() > expiry:
            # Value has expired, remove it from the cache
            self._delete(key)
            return False
        
        return True
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from the metadata file.
        
        Returns:
            The metadata dictionary.
        """
        if not os.path.exists(self.metadata_file):
            return {}
        
        try:
            with open(self.metadata_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            # Error loading metadata, start fresh
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> None:
        """Save metadata to the metadata file.
        
        Args:
            metadata: The metadata dictionary.
        """
        try:
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata, f)
        except Exception as e:
            raise CacheError(f"Error saving metadata: {str(e)}")
    
    def _ensure_space(self, value: Any) -> None:
        """Ensure there is enough space in the cache for a new value.
        
        Args:
            value: The value to cache.
        """
        # Get the current cache size
        metadata = self._load_metadata()
        current_size = sum(item.get("size", 0) for item in metadata.values())
        
        # Estimate the size of the new value
        # This is a rough estimate, as the actual size on disk may be different
        try:
            value_size = len(pickle.dumps(value))
        except:
            # If we can't estimate the size, assume it's small
            value_size = 1024
        
        # Check if we need to evict items
        if self.max_size is not None and current_size + value_size > self.max_size:
            # Sort items by last access time
            items = sorted(
                [(k, v) for k, v in metadata.items()],
                key=lambda x: x[1].get("last_access", 0)
            )
            
            # Remove items until we have enough space
            space_needed = current_size + value_size - self.max_size
            space_freed = 0
            
            for key, item in items:
                if space_freed >= space_needed:
                    break
                
                self._delete(key)
                space_freed += item.get("size", 0)
