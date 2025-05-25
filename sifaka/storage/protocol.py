"""Storage protocol for Sifaka.

This module defines the simple Storage protocol that all storage implementations
must follow. The protocol is designed to be minimal and easy to implement.
"""

from typing import Any, List, Optional, Protocol


class Storage(Protocol):
    """Simple storage protocol for Sifaka.
    
    All storage implementations must provide these four methods:
    - get: Retrieve a value by key
    - set: Store a value with a key
    - search: Search for items (for vector/semantic search)
    - clear: Remove all stored data
    """
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key.
        
        Args:
            key: The storage key.
            
        Returns:
            The stored value, or None if not found.
        """
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set a value for a key.
        
        Args:
            key: The storage key.
            value: The value to store.
        """
        ...
    
    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.
        
        For simple storage backends, this might just return all values.
        For vector storage backends, this performs semantic search.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            
        Returns:
            List of matching items.
        """
        ...
    
    def clear(self) -> None:
        """Clear all stored data."""
        ...
