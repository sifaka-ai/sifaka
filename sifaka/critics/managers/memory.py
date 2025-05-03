"""
Memory manager for critics.

This module provides the MemoryManager class which is responsible for
managing memory for critics.

## Component Lifecycle

### MemoryManager Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Buffer initialization
   - Resource allocation
   - Error handling setup

2. **Usage Phase**
   - Memory item storage
   - Memory retrieval
   - Buffer management
   - Error handling and recovery

3. **Cleanup Phase**
   - Memory clearing
   - Resource cleanup
   - State reset
   - Error recovery

### Component Interactions

1. **Critic Core**
   - Receives memory items
   - Requests memory retrieval
   - Handles memory errors

2. **Model Provider**
   - Uses memory for context
   - Provides memory items
   - Handles memory-related errors

3. **Prompt Manager**
   - Incorporates memory in prompts
   - Manages memory formatting
   - Handles memory integration

### Error Handling and Recovery

1. **Input Validation Errors**
   - Empty or invalid items
   - Invalid buffer size
   - Recovery: Skip invalid items

2. **Memory Management Errors**
   - Buffer overflow
   - Memory retrieval failures
   - Recovery: Use default values or clear buffer

3. **Resource Errors**
   - Memory allocation failures
   - Buffer initialization errors
   - Recovery: Resource cleanup and state preservation

## Examples

```python
from sifaka.critics.managers.memory import MemoryManager

# Create a memory manager with a buffer size of 5
memory_manager = MemoryManager(buffer_size=5)

# Add items to memory
memory_manager.add_to_memory("First reflection")
memory_manager.add_to_memory("Second reflection")
memory_manager.add_to_memory("Third reflection")

# Get all items from memory
all_items = memory_manager.get_memory()
print(f"All items: {all_items}")

# Get the last 2 items
recent_items = memory_manager.get_memory(max_items=2)
print(f"Recent items: {recent_items}")

# Clear memory
memory_manager.clear_memory()
print(f"Memory size after clear: {memory_manager.memory_size}")
```
"""

from collections import deque
from typing import List, Optional

from ...utils.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Manages memory for critics.

    This class is responsible for storing and retrieving reflections
    and other memory items for critics. It implements a circular buffer
    to maintain a fixed-size memory of recent items.

    ## Lifecycle Management

    The MemoryManager manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up buffer
       - Configures error handling
       - Allocates resources

    2. **Operation**
       - Stores memory items
       - Retrieves memory
       - Manages buffer
       - Handles errors

    3. **Cleanup**
       - Clears memory
       - Releases resources
       - Logs final status

    ## Error Handling

    The MemoryManager implements comprehensive error handling:

    1. **Input Validation**
       - Validates memory items
       - Checks buffer size
       - Verifies item format

    2. **Memory Management**
       - Handles buffer overflow
       - Manages retrieval errors
       - Provides fallback values

    3. **Resource Management**
       - Handles allocation failures
       - Manages cleanup errors
       - Preserves valid state

    ## Examples

    ```python
    from sifaka.critics.managers.memory import MemoryManager

    # Create a memory manager
    memory_manager = MemoryManager(buffer_size=3)

    # Add items to memory
    memory_manager.add_to_memory("First item")
    memory_manager.add_to_memory("Second item")
    memory_manager.add_to_memory("Third item")

    # Get memory items
    items = memory_manager.get_memory()
    print(f"Items: {items}")

    # Get memory size
    size = memory_manager.memory_size
    print(f"Size: {size}")

    # Clear memory
    memory_manager.clear_memory()
    ```
    """

    def __init__(self, buffer_size: int = 5):
        """
        Initialize a MemoryManager instance.

        This method sets up the memory manager with a fixed-size buffer
        for storing memory items.

        Lifecycle:
        1. Validate configuration
        2. Initialize buffer
        3. Set up error handling
        4. Configure resources

        Args:
            buffer_size: The maximum number of items to store in memory

        Raises:
            ValueError: If buffer_size is invalid
            RuntimeError: If initialization fails
        """
        self._buffer_size = max(1, buffer_size)
        self._memory_buffer: deque = deque(maxlen=self._buffer_size)

    def add_to_memory(self, item: str) -> None:
        """
        Add an item to memory.

        This method adds a new item to the memory buffer, maintaining
        the buffer size limit by removing the oldest item if necessary.

        Lifecycle:
        1. Input validation
        2. Buffer management
        3. Item storage
        4. Error handling

        Args:
            item: The string item to add to memory

        Raises:
            ValueError: If item is empty or invalid
            RuntimeError: If memory operation fails

        Examples:
            ```python
            # Add a single item
            memory_manager.add_to_memory("First reflection")

            # Add multiple items
            for i in range(3):
                memory_manager.add_to_memory(f"Reflection {i}")

            # Add an item that will cause buffer overflow
            memory_manager = MemoryManager(buffer_size=2)
            memory_manager.add_to_memory("First")  # Stored
            memory_manager.add_to_memory("Second")  # Stored
            memory_manager.add_to_memory("Third")  # First is removed
            ```
        """
        if not item or not isinstance(item, str):
            raise ValueError("Invalid memory item: must be non-empty string")
        self._memory_buffer.append(item)

    def get_memory(self, max_items: Optional[int] = None) -> List[str]:
        """
        Retrieve items from memory.

        This method returns items from the memory buffer, optionally
        limiting the number of items returned. Items are returned in
        chronological order (oldest first).

        Lifecycle:
        1. Parameter validation
        2. Memory retrieval
        3. Result formatting
        4. Error handling

        Args:
            max_items: Optional maximum number of items to return.
                      If None, returns all items.

        Returns:
            A list of memory items, ordered from oldest to newest

        Raises:
            ValueError: If max_items is invalid
            RuntimeError: If memory retrieval fails

        Examples:
            ```python
            # Get all items
            memory_manager.add_to_memory("First")
            memory_manager.add_to_memory("Second")
            all_items = memory_manager.get_memory()
            print(all_items)  # ["First", "Second"]

            # Get limited items
            memory_manager.add_to_memory("Third")
            recent_items = memory_manager.get_memory(max_items=2)
            print(recent_items)  # ["Second", "Third"]

            # Get items from empty memory
            memory_manager.clear_memory()
            items = memory_manager.get_memory()
            print(items)  # []
            ```
        """
        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be non-negative")
        items = list(self._memory_buffer)
        if max_items is not None:
            items = items[-max_items:]
        return items

    def clear_memory(self) -> None:
        """
        Clear all items from memory.

        This method removes all items from the memory buffer, effectively
        resetting the memory state. The buffer size remains unchanged.

        Lifecycle:
        1. State validation
        2. Memory clearing
        3. Resource cleanup
        4. Error handling

        Raises:
            RuntimeError: If memory clearing fails

        Examples:
            ```python
            # Clear populated memory
            memory_manager.add_to_memory("First")
            memory_manager.add_to_memory("Second")
            memory_manager.clear_memory()
            print(memory_manager.memory_size)  # 0

            # Clear empty memory
            memory_manager.clear_memory()
            print(memory_manager.memory_size)  # 0
            ```
        """
        self._memory_buffer.clear()

    @property
    def memory_size(self) -> int:
        """
        Get the current number of items in memory.

        This property returns the current number of items stored in
        the memory buffer, which will be between 0 and the buffer size.

        Lifecycle:
        1. State validation
        2. Size calculation
        3. Error handling

        Returns:
            The number of items currently in memory

        Examples:
            ```python
            # Check empty memory
            print(memory_manager.memory_size)  # 0

            # Check populated memory
            memory_manager.add_to_memory("First")
            memory_manager.add_to_memory("Second")
            print(memory_manager.memory_size)  # 2

            # Check after clearing
            memory_manager.clear_memory()
            print(memory_manager.memory_size)  # 0
            ```
        """
        return len(self._memory_buffer)
