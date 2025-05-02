"""
Memory manager for critics.

This module provides the MemoryManager class which is responsible for
managing memory for critics.
"""

from collections import deque
from typing import List, Optional

from ...utils.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Manages memory for critics.
    
    This class is responsible for storing and retrieving reflections
    and other memory items for critics.
    """
    
    def __init__(self, buffer_size: int = 5):
        """
        Initialize a MemoryManager instance.
        
        Args:
            buffer_size: The maximum number of items to store in memory
        """
        self._buffer_size = max(1, buffer_size)
        self._memory_buffer: deque = deque(maxlen=self._buffer_size)
        
    def add_to_memory(self, item: str) -> None:
        """
        Add an item to memory.
        
        Args:
            item: The item to add to memory
        """
        if not item or not item.strip():
            return
            
        # Add to memory buffer
        self._memory_buffer.append(item.strip())
        logger.debug(f"Added item to memory buffer (size: {len(self._memory_buffer)})")
        
    def get_memory(self, max_items: Optional[int] = None) -> List[str]:
        """
        Get items from memory.
        
        Args:
            max_items: The maximum number of items to retrieve
            
        Returns:
            A list of memory items
        """
        if not max_items:
            return list(self._memory_buffer)
            
        # Get the most recent items up to max_items
        return list(self._memory_buffer)[-max_items:]
        
    def clear_memory(self) -> None:
        """
        Clear all items from memory.
        """
        self._memory_buffer.clear()
        logger.debug("Cleared memory buffer")
        
    @property
    def memory_size(self) -> int:
        """
        Get the number of items in memory.
        
        Returns:
            The number of items in memory
        """
        return len(self._memory_buffer)
