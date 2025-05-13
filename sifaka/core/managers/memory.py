"""
Memory Management Module

A module providing memory management capabilities for Sifaka components.

## Overview
This module provides memory management implementations that can be used
across different components of the Sifaka library. It offers two main
memory manager implementations:

1. KeyValueMemoryManager: For key-value based memory storage and retrieval
2. BufferMemoryManager: For simple circular buffer storage of string items

Both implementations use StateManager for consistent state management
and provide statistics tracking.

## Components
- BaseMemory: Protocol defining the memory interface
- MemoryConfig: Configuration for memory managers
- MemoryResult: Result of memory operations
- KeyValueMemoryManager: Key-value based memory manager
- BufferMemoryManager: Circular buffer memory manager
- Factory functions for creating memory managers

## Usage Examples
```python
from sifaka.core.managers.memory import create_key_value_memory_manager, create_buffer_memory_manager

# Create a key-value memory manager
kv_memory = create_key_value_memory_manager(
    name="my_memory",
    cache_enabled=True,
    max_items=100
)

# Store a value
kv_memory.store("key1", "value1") if kv_memory else ""

# Retrieve a value
results = kv_memory.retrieve("key1") if kv_memory else ""

# Create a buffer memory manager
buffer_memory = create_buffer_memory_manager(buffer_size=10)

# Add items to buffer
buffer_memory.add_to_memory("Item 1") if buffer_memory else ""
buffer_memory.add_to_memory("Item 2") if buffer_memory else ""

# Retrieve items
items = buffer_memory.get_memory() if buffer_memory else ""
```

## Error Handling
The memory managers handle various error conditions:
- Invalid input validation
- Memory storage/retrieval failures
- Configuration errors
- Resource management errors

## Configuration
Memory managers can be configured with:
- Cache settings (enabled/disabled, size)
- Buffer size
- Memory limits
- Custom memory implementations
"""

from typing import Any, Dict, List, Optional, TypeVar, Protocol
import time

from pydantic import Field, ConfigDict, PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
V = TypeVar("V")  # Value type


# Key-Value Memory Manager Implementation


class MemoryConfig(BaseConfig):
    """
    Configuration for key-value memory manager.

    This class defines the configuration parameters for memory managers,
    including cache settings and memory limits.

    ## Architecture
    Extends BaseConfig to provide consistent configuration handling
    across all Sifaka components.

    Attributes:
        cache_enabled (bool): Whether to enable caching
        max_items (int): Maximum number of items to store in memory
    """

    cache_enabled: bool = Field(default=True, description="Whether to enable caching")
    max_items: int = Field(default=100, description="Maximum number of items to store in memory")

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class MemoryResult(BaseResult):
    """
    Result of memory operation.

    This class represents the result of a memory operation, including
    the key, value, and whether the operation was successful.

    ## Architecture
    Extends BaseResult to provide consistent result handling
    across all Sifaka components.

    Attributes:
        key (str): The key used in the memory operation
        value (Any): The value retrieved or stored
        found (bool): Whether the key was found in memory
    """

    key: str = Field(default="")
    value: Any = Field(default=None)
    found: bool = Field(default=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class BaseMemory(Protocol[V]):
    """
    Protocol for memory implementations.

    This protocol defines the interface that all memory implementations
    must follow, providing store and retrieve operations.

    ## Architecture
    Uses Python's Protocol type to define a structural interface
    that memory implementations must satisfy.

    Attributes:
        name (str): The name of the memory implementation
    """

    name: str

    def store(self, key: str, value: V) -> None:
        """
        Store a value in memory.

        Args:
            key (str): The key to store the value under
            value (V): The value to store

        Raises:
            ValueError: If the key or value is invalid
            RuntimeError: If storage fails
        """
        ...

    def retrieve(self, key: str) -> MemoryResult:
        """
        Retrieve a value from memory.

        Args:
            key (str): The key to retrieve the value for

        Returns:
            MemoryResult: The result of the retrieval operation

        Raises:
            ValueError: If the key is invalid
            RuntimeError: If retrieval fails
        """
        ...


class KeyValueMemoryManager(BaseComponent[Dict[str, Any], MemoryResult]):
    """
    Key-value memory manager for Sifaka.

    This class provides stateful memory management and context tracking,
    coordinating between multiple memory stores and tracking memory operations.

    ## Architecture
    The KeyValueMemoryManager follows a component-based architecture:
    - Inherits from BaseComponent for consistent behavior
    - Uses StateManager for state management
    - Implements caching for performance
    - Tracks statistics for monitoring

    ## Lifecycle
    1. Initialization: Set up with memories and configuration
    2. Storage: Store values in memory
    3. Retrieval: Retrieve values from memory
    4. Memory Management: Add/remove memories as needed
    5. Statistics: Track memory operations performance
    """

    # State management
    _state_manager = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        memories: List[BaseMemory[Any]],
        name: str = "memory_manager",
        description: str = "Memory manager for Sifaka",
        cache_enabled: bool = True,
        max_items: int = 100,
        config: Optional[Optional[MemoryConfig]] = None,
        **kwargs: Any,
    ):
        """Initialize the memory manager.

        Args:
            memories: List of memories to use for storage
            name: Name of the manager
            description: Description of the manager
            cache_enabled: Whether to enable caching
            max_items: Maximum number of items to store in memory
            config: Additional configuration
            **kwargs: Additional keyword arguments for configuration
        """
        # Create config if not provided
        if config is None:
            config = MemoryConfig(
                name=name,
                description=description,
                cache_enabled=cache_enabled,
                max_items=max_items,
                **kwargs,
            )

        # Initialize base component
        super().__init__(name, description, config)

        # Store memories in state
        if self._state_manager:
            self._state_manager.update("memories", memories)
            self._state_manager.update("result_cache", {})
            self._state_manager.update("initialized", True)

        # Set metadata
        if self._state_manager:
            self._state_manager.set_metadata("component_type", "memory_manager")
            self._state_manager.set_metadata("creation_time", time.time() if time else 0)
            self._state_manager.set_metadata("memory_count", len(memories))

    def process(self, input: Dict[str, Any]) -> MemoryResult:
        """
        Process the input and return a memory result.

        This is the implementation of the abstract method from BaseComponent.

        Args:
            input: Dictionary containing key and value to store or retrieve
                - operation: "store" or "retrieve"
                - key: The key to store or retrieve
                - value: The value to store (for store operation)

        Returns:
            MemoryResult with operation details

        Raises:
            ValueError: If input is invalid
        """
        # Validate input
        if not isinstance(input, dict):
            return MemoryResult(
                passed=False,
                message="Invalid input: must be a dictionary",
                metadata={"error_type": "invalid_input"},
                score=0.0,
                issues=["Input must be a dictionary"],
                suggestions=["Provide a dictionary with operation, key, and value"],
            )

        operation = input.get("operation") if input else None
        key = input.get("key") if input else None
        value = input.get("value") if input else None

        if not operation or not key:
            return MemoryResult(
                passed=False,
                message="Invalid input: missing required fields",
                metadata={"error_type": "missing_fields"},
                score=0.0,
                issues=["Missing required fields"],
                suggestions=["Provide operation and key fields"],
            )

        # Perform operation
        if operation == "store":
            if value is None:
                return MemoryResult(
                    passed=False,
                    message="Invalid input: missing value for store operation",
                    metadata={"error_type": "missing_value"},
                    score=0.0,
                    issues=["Missing value for store operation"],
                    suggestions=["Provide a value to store"],
                )

            if self:
                self.store(key, value)
            return MemoryResult(
                passed=True,
                message=f"Value stored under key '{key}'",
                metadata={"operation": "store", "key": key},
                score=1.0,
                key=key,
                value=value,
            )

        elif operation == "retrieve":
            results = self.retrieve(key) if self else []
            if not results:
                return MemoryResult(
                    passed=False,
                    message=f"No value found for key '{key}'",
                    metadata={"operation": "retrieve", "key": key},
                    score=0.0,
                    key=key,
                    found=False,
                )

            # Return the first result
            result = results[0] if results else None
            return MemoryResult(
                passed=True,
                message=f"Value retrieved for key '{key}'",
                metadata={"operation": "retrieve", "key": key},
                score=1.0,
                key=key,
                value=result.value if result else None,
                found=True,
            )

        else:
            return MemoryResult(
                passed=False,
                message=f"Invalid operation: {operation}",
                metadata={"error_type": "invalid_operation"},
                score=0.0,
                issues=[f"Invalid operation: {operation}"],
                suggestions=["Use 'store' or 'retrieve' as operation"],
            )

    def store(self, key: str, value: Any) -> None:
        """
        Store a value in memory.

        Args:
            key: The key to store the value under
            value: The value to store

        Raises:
            ValueError: If storage fails
        """
        # Record start time
        start_time = time.time() if time else 0

        try:
            # Get memories from state
            memories = self._state_manager.get("memories", []) if self._state_manager else []
            if not memories:
                if logger:
                    logger.warning("No memories available for storage")
                return

            # Store in each memory
            for memory in memories:
                if memory:
                    memory.store(key, value)

            # Update statistics
            if self:
                self.update_statistics(
                    MemoryResult(
                        passed=True,
                        message=f"Value stored under key '{key}'",
                        metadata={"operation": "store", "key": key},
                        score=1.0,
                        key=key,
                        value=value,
                        processing_time_ms=(time.time() - start_time) * 1000 if time else 0,
                    )
                )

        except Exception as e:
            # Record error
            if self:
                self.record_error(e)
            if logger:
                logger.error(f"Memory storage error: {str(e)}")
            raise ValueError(f"Memory storage error: {str(e)}")

    def retrieve(self, key: str) -> List[MemoryResult]:
        """
        Retrieve a value from memory.

        Args:
            key: The key to retrieve

        Returns:
            List of retrieved values

        Raises:
            ValueError: If retrieval fails
        """
        # Handle empty input
        if not key:
            if logger:
                logger.warning("Empty key provided for retrieval")
            return []

        # Record start time
        start_time = time.time() if time else 0

        try:
            # Check cache if enabled
            cache = self._state_manager.get("result_cache", {}) if self._state_manager else {}

            if key in cache and self.config and self.config.cache_enabled:
                if self._state_manager:
                    self._state_manager.set_metadata("cache_hit", True)
                return cache[key]

            # Mark as cache miss
            if self._state_manager:
                self._state_manager.set_metadata("cache_hit", False)

            # Get memories from state
            memories = self._state_manager.get("memories", []) if self._state_manager else []
            if not memories:
                if logger:
                    logger.warning("No memories available for retrieval")
                return []

            # Retrieve from each memory
            results = []
            for memory in memories:
                try:
                    if memory:
                        result = memory.retrieve(key)
                        if results is not None:
                            results.append(result)
                except Exception as e:
                    if logger:
                        logger.error(f"Error retrieving from memory {memory.name}: {str(e)}")
                    # Continue with other memories

            # Create result for statistics
            retrieval_result = MemoryResult(
                passed=len(results) > 0,
                message=f"Retrieved {len(results)} results for key '{key}'",
                metadata={
                    "operation": "retrieve",
                    "key": key,
                    "result_count": len(results),
                    "cache_hit": False,
                },
                score=1.0 if results else 0.0,
                key=key,
                found=len(results) > 0,
                processing_time_ms=(time.time() - start_time) * 1000 if time else 0,
            )

            # Update statistics
            if self:
                self.update_statistics(retrieval_result)

            # Cache result if caching is enabled
            if self.config and self.config.cache_enabled:
                # Manage cache size
                if len(cache) >= self.config.max_items:
                    # Remove oldest entry (simple approach)
                    if cache:
                        oldest_key = next(iter(cache))
                        del cache[oldest_key]

                cache[key] = results
                if self._state_manager:
                    self._state_manager.update("result_cache", cache)

            return results

        except Exception as e:
            # Record error
            if self:
                self.record_error(e)
            if logger:
                logger.error(f"Memory retrieval error: {str(e)}")
            raise ValueError(f"Memory retrieval error: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about memory operations.

        Returns:
            Dictionary with usage statistics
        """
        # Get base statistics from parent class
        stats = super().get_statistics()

        # Add memory-specific statistics
        if stats:
            stats.update(
                {
                    "cache_size": (
                        len(self._state_manager.get("result_cache", {}))
                        if self._state_manager
                        else 0
                    ),
                    "memory_count": (
                        len(self._state_manager.get("memories", [])) if self._state_manager else 0
                    ),
                    "cache_enabled": self.config.cache_enabled if self.config else False,
                    "max_items": self.config.max_items if self.config else 0,
                }
            )

        return stats

    def clear_cache(self) -> None:
        """Clear the memory result cache."""
        if self._state_manager:
            self._state_manager.update("result_cache", {})
        if logger:
            logger.debug(f"Memory cache cleared for {self.name}")

    def warm_up(self) -> None:
        """Prepare the memory manager for use."""
        super().warm_up()

        # Pre-validate memories
        memories = self._state_manager.get("memories", []) if self._state_manager else []
        for memory in memories:
            if hasattr(memory, "warm_up") and memory:
                memory.warm_up()

        if logger:
            logger.debug(f"Memory manager '{self.name}' warmed up with {len(memories)} memories")

    def add_memory(self, memory: Any) -> None:
        """
        Add a memory store.

        Args:
            memory: The memory store to add

        Raises:
            ValueError: If the memory is invalid
            TypeError: If the input type is incorrect
        """
        # Validate memory type
        if not isinstance(memory, BaseMemory):
            raise ValueError(f"Expected BaseMemory instance, got {type(memory)}")

        # Check for duplicate memory names
        memories = self._state_manager.get("memories", []) if self._state_manager else []
        if any(m.name == memory.name for m in memories):
            if logger:
                logger.warning(
                    f"Memory with name '{memory.name}' already exists, it will be replaced"
                )
            # Remove existing memory with same name
            if self:
                self.remove_memory(memory.name)
            # Get updated memories list
            memories = self._state_manager.get("memories", []) if self._state_manager else []

        # Add memory to the list
        if memories is not None:
            memories.append(memory)
        if self._state_manager:
            self._state_manager.update("memories", memories)

        # Update metadata
        if self._state_manager:
            self._state_manager.set_metadata("memory_count", len(memories))

        # Clear cache since memory operations may change
        if self:
            self.clear_cache()

        if logger:
            logger.debug(f"Added memory '{memory.name}' to memory manager '{self.name}'")

    def remove_memory(self, memory_name: str) -> None:
        """
        Remove a memory store by name.

        Args:
            memory_name: The name of the memory store to remove

        Raises:
            ValueError: If the memory name is invalid or memory not found
        """
        # Validate input
        if not memory_name or not isinstance(memory_name, str):
            raise ValueError(f"Invalid memory name: {memory_name}")

        # Find memory by name
        memory_to_remove = None
        memories = self._state_manager.get("memories", []) if self._state_manager else []
        for memory in memories:
            if memory.name == memory_name:
                memory_to_remove = memory
                break

        if memory_to_remove is None:
            raise ValueError(f"Memory not found: {memory_name}")

        # Remove memory from list
        if memories is not None:
            memories.remove(memory_to_remove)
        if self._state_manager:
            self._state_manager.update("memories", memories)

        # Update metadata
        if self._state_manager:
            self._state_manager.set_metadata("memory_count", len(memories))

        # Clear cache since memory operations may change
        if self:
            self.clear_cache()

        if logger:
            logger.debug(f"Removed memory '{memory_name}' from memory manager '{self.name}'")

    def get_memories(self) -> List[BaseMemory]:
        """
        Get all registered memories.

        Returns:
            The list of registered memories
        """
        return self._state_manager.get("memories", []) if self._state_manager else []


def create_key_value_memory_manager(
    memories: Optional[List[BaseMemory]] = None,
    name: str = "memory_manager",
    description: str = "Memory manager for Sifaka",
    cache_enabled: bool = True,
    max_items: int = 100,
    component_type: Optional[str] = None,
    **kwargs: Any,
) -> KeyValueMemoryManager:
    """
    Create a key-value memory manager.

    This factory function creates and configures a KeyValueMemoryManager
    with the specified parameters, providing a convenient way to create
    memory managers with default or custom configurations.

    Args:
        memories (List[BaseMemory], optional): List of memories to use for storage
        name (str, optional): Name of the manager. Defaults to "memory_manager".
        description (str, optional): Description of the manager. Defaults to "Memory manager for Sifaka".
        cache_enabled (bool, optional): Whether to enable caching. Defaults to True.
        max_items (int, optional): Maximum number of items to store in memory. Defaults to 100.
        component_type (str, optional): Type of component using this memory manager.
            If "chain", uses chain-specific defaults.
        **kwargs (Any): Additional configuration parameters

    Returns:
        KeyValueMemoryManager: Configured KeyValueMemoryManager instance

    Example:
        ```python
        # Create a memory manager with default settings
        memory_manager = create_key_value_memory_manager()

        # Create a memory manager with custom settings
        custom_manager = create_key_value_memory_manager(
            name="custom_memory",
            cache_enabled=False,
            max_items=50
        )

        # Create a chain-specific memory manager
        chain_manager = create_key_value_memory_manager(
            component_type="chain",
            max_items=200
        )
        ```
    """
    # Set component-specific defaults
    if component_type == "chain":
        name = name or "chain_memory_manager"
        description = description or "Memory manager for Sifaka chains"

    config = MemoryConfig(
        name=name,
        description=description,
        cache_enabled=cache_enabled,
        max_items=max_items,
        **kwargs,
    )

    return KeyValueMemoryManager(
        memories=memories or [],
        name=name,
        description=description,
        config=config,
    )


# Buffer Memory Manager Implementation


class BufferMemoryManager:
    """
    Manages memory for components using a circular buffer.

    This class is responsible for storing and retrieving string items
    using a fixed-size circular buffer. It implements a simple interface
    for adding and retrieving items, with automatic overflow handling.

    ## Lifecycle Management

    The BufferMemoryManager manages its lifecycle through three main phases:

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
    """

    def __init__(self, buffer_size: int = 5):
        """
        Initialize a BufferMemoryManager instance.

        This method sets up the memory manager with a fixed-size buffer
        for storing memory items.

        Args:
            buffer_size: The maximum number of items to store in memory

        Raises:
            ValueError: If buffer_size is invalid
            RuntimeError: If initialization fails
        """
        buffer_size = max(1, buffer_size)

        # Initialize state manager
        self._state_manager = StateManager()

        # Initialize state
        if self._state_manager:
            self._state_manager.update("buffer_size", buffer_size)
            self._state_manager.update("memory_buffer", list())
            self._state_manager.update("initialized", True)

            # Initialize metadata
            self._state_manager.set_metadata("component_type", "buffer_memory_manager")
            self._state_manager.set_metadata("creation_time", time.time() if time else 0)
            self._state_manager.set_metadata("add_count", 0)
            self._state_manager.set_metadata("retrieve_count", 0)
            self._state_manager.set_metadata("clear_count", 0)
            self._state_manager.set_metadata("overflow_count", 0)
            self._state_manager.set_metadata("error_count", 0)

    def add_to_memory(self, item: str) -> None:
        """
        Add an item to memory.

        This method adds a new item to the memory buffer, maintaining
        the buffer size limit by removing the oldest item if necessary.

        Args:
            item: The string item to add to memory

        Raises:
            ValueError: If item is empty or invalid
            RuntimeError: If memory operation fails
        """
        if not item or not isinstance(item, str):
            # Track error
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)
            raise ValueError("Invalid memory item: must be non-empty string")

        # Track add count
        if self._state_manager:
            add_count = self._state_manager.get_metadata("add_count", 0)
            self._state_manager.set_metadata("add_count", add_count + 1)

        try:
            # Get current buffer and buffer size
            memory_buffer = (
                self._state_manager.get("memory_buffer", list()) if self._state_manager else []
            )
            buffer_size = self._state_manager.get("buffer_size", 5) if self._state_manager else 5

            # Add the item to the buffer
            if memory_buffer is not None:
                memory_buffer.append(item)

            # Handle overflow - if buffer exceeds size limit, remove oldest items
            if len(memory_buffer) > buffer_size:
                # Track overflow
                if self._state_manager:
                    overflow_count = self._state_manager.get_metadata("overflow_count", 0)
                    self._state_manager.set_metadata("overflow_count", overflow_count + 1)

                # Remove oldest items
                memory_buffer = memory_buffer[-buffer_size:]

            # Update the buffer in state
            if self._state_manager:
                self._state_manager.update("memory_buffer", memory_buffer)

        except Exception as e:
            # Track error
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)

            # Log and raise
            if logger:
                logger.error(f"Failed to add item to memory: {e}")
            raise RuntimeError(f"Failed to add item to memory: {e}")

    def get_memory(self, max_items: Optional[Optional[int]] = None) -> List[str]:
        """
        Retrieve items from memory.

        This method returns items from the memory buffer, optionally
        limiting the number of items returned. Items are returned in
        chronological order (oldest first).

        Args:
            max_items: Optional maximum number of items to return.
                      If None, returns all items.

        Returns:
            A list of memory items, ordered from oldest to newest

        Raises:
            ValueError: If max_items is invalid
            RuntimeError: If memory retrieval fails
        """
        if max_items is not None and max_items < 0:
            # Track error
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)
            raise ValueError("max_items must be non-negative")

        # Track retrieve count
        if self._state_manager:
            retrieve_count = self._state_manager.get_metadata("retrieve_count", 0)
            self._state_manager.set_metadata("retrieve_count", retrieve_count + 1)

        try:
            # Get memory buffer
            memory_buffer = (
                self._state_manager.get("memory_buffer", list()) if self._state_manager else []
            )

            # Return all or recent items based on max_items
            if max_items is not None:
                items = memory_buffer[-max_items:] if memory_buffer else []
            else:
                items = list(memory_buffer)

            return items

        except Exception as e:
            # Track error
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)

            # Log and raise
            if logger:
                logger.error(f"Failed to retrieve items from memory: {e}")
            raise RuntimeError(f"Failed to retrieve items from memory: {e}")

    def clear_memory(self) -> None:
        """
        Clear all items from memory.

        This method removes all items from the memory buffer, effectively
        resetting the memory state. The buffer size remains unchanged.

        Raises:
            RuntimeError: If memory clearing fails
        """
        # Track clear count
        if self._state_manager:
            clear_count = self._state_manager.get_metadata("clear_count", 0)
            self._state_manager.set_metadata("clear_count", clear_count + 1)

        try:
            # Clear the memory buffer
            if self._state_manager:
                self._state_manager.update("memory_buffer", list())

        except Exception as e:
            # Track error
            if self._state_manager:
                error_count = self._state_manager.get_metadata("error_count", 0)
                self._state_manager.set_metadata("error_count", error_count + 1)

            # Log and raise
            if logger:
                logger.error(f"Failed to clear memory: {e}")
            raise RuntimeError(f"Failed to clear memory: {e}")

    @property
    def memory_size(self) -> int:
        """
        Get the current number of items in memory.

        This property returns the current number of items stored in
        the memory buffer, which will be between 0 and the buffer size.

        Returns:
            The number of items currently in memory
        """
        try:
            # Get the memory buffer and return its length
            memory_buffer = (
                self._state_manager.get("memory_buffer", list()) if self._state_manager else []
            )
            return len(memory_buffer)

        except Exception as e:
            # Log error but don't raise (property accessor should be safe)
            if logger:
                logger.error(f"Failed to get memory size: {e}")
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "buffer_size": self._state_manager.get("buffer_size", 0) if self._state_manager else 0,
            "memory_size": self.memory_size,
            "add_count": (
                self._state_manager.get_metadata("add_count", 0) if self._state_manager else 0
            ),
            "retrieve_count": (
                self._state_manager.get_metadata("retrieve_count", 0) if self._state_manager else 0
            ),
            "clear_count": (
                self._state_manager.get_metadata("clear_count", 0) if self._state_manager else 0
            ),
            "overflow_count": (
                self._state_manager.get_metadata("overflow_count", 0) if self._state_manager else 0
            ),
            "error_count": (
                self._state_manager.get_metadata("error_count", 0) if self._state_manager else 0
            ),
            "uptime": (
                (time.time() - self._state_manager.get_metadata("creation_time", time.time()))
                if self._state_manager and time
                else 0
            ),
        }


def create_buffer_memory_manager(
    buffer_size: int = 5,
    name: Optional[str] = None,
    description: Optional[str] = None,
    component_type: Optional[str] = None,
    **kwargs: Any,
) -> BufferMemoryManager:
    """
    Create a buffer memory manager.

    This factory function creates and configures a BufferMemoryManager
    with the specified buffer size, providing a convenient way to create
    buffer memory managers with default or custom configurations.

    Args:
        buffer_size (int, optional): The maximum number of items to store in memory. Defaults to 5.
        name (str, optional): Name of the manager. Used for metadata.
        description (str, optional): Description of the manager. Used for metadata.
        component_type (str, optional): Type of component using this memory manager.
            If "chain", uses chain-specific defaults (buffer_size=10).
        **kwargs (Any): Additional configuration parameters (for future extensibility)

    Returns:
        BufferMemoryManager: Configured BufferMemoryManager instance

    Example:
        ```python
        # Create a buffer memory manager with default settings
        buffer_memory = create_buffer_memory_manager()

        # Create a buffer memory manager with custom buffer size
        large_buffer = create_buffer_memory_manager(buffer_size=20)

        # Create a chain-specific buffer memory manager
        chain_buffer = create_buffer_memory_manager(
            component_type="chain",
            name="chain_buffer",
            description="Buffer for chain memory"
        )
        ```

    Raises:
        ValueError: If buffer_size is less than 1
    """
    # Set component-specific defaults
    if component_type == "chain":
        buffer_size = buffer_size or 10

    # Create the buffer memory manager
    buffer_manager = BufferMemoryManager(buffer_size=buffer_size)

    # Set name and description in state manager if provided
    if name and buffer_manager._state_manager:
        buffer_manager._state_manager.set_metadata("name", name)
    if description and buffer_manager._state_manager:
        buffer_manager._state_manager.set_metadata("description", description)

    return buffer_manager
