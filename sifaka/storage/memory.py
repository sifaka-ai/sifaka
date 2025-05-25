"""In-memory storage implementation.

This is the default storage backend that keeps everything in memory with no persistence.
Perfect for development, testing, and simple use cases.
"""

from typing import Any, Dict, List, Optional

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryStorage:
    """Simple in-memory storage with no persistence.
    
    This is the default storage backend for Sifaka. It stores everything in a
    Python dictionary and provides no persistence across process restarts.
    
    Perfect for:
    - Development and testing
    - Simple scripts and experiments  
    - Cases where persistence is not needed
    
    Attributes:
        data: Dictionary storing all key-value pairs.
    """
    
    def __init__(self):
        """Initialize empty memory storage."""
        self.data: Dict[str, Any] = {}
        logger.debug("Initialized MemoryStorage")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key.
        
        Args:
            key: The storage key.
            
        Returns:
            The stored value, or None if not found.
        """
        value = self.data.get(key)
        logger.debug(f"Memory get: {key} -> {'found' if value is not None else 'not found'}")
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a value for a key.
        
        Args:
            key: The storage key.
            value: The value to store.
        """
        self.data[key] = value
        logger.debug(f"Memory set: {key} -> stored")
    
    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.
        
        For memory storage, this just returns all values (no semantic search).
        
        Args:
            query: The search query (ignored for memory storage).
            limit: Maximum number of results to return.
            
        Returns:
            List of all stored values, limited by the limit parameter.
        """
        values = list(self.data.values())[:limit]
        logger.debug(f"Memory search: '{query}' -> {len(values)} results")
        return values
    
    def clear(self) -> None:
        """Clear all stored data."""
        count = len(self.data)
        self.data.clear()
        logger.debug(f"Memory clear: removed {count} items")
    
    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self.data)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        return key in self.data
