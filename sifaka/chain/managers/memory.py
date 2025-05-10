"""
Memory manager for Sifaka chains.

This module provides the memory manager for Sifaka chains,
enabling stateful memory management and context tracking.

## Architecture
The MemoryManager follows a component-based architecture:
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


class MemoryConfig(BaseConfig):
    """Configuration for memory manager."""

    cache_enabled: bool = Field(default=True, description="Whether to enable caching")
    max_items: int = Field(default=100, description="Maximum number of items to store in memory")

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class MemoryResult(BaseResult):
    """Result of memory operation."""

    key: str = Field(default="")
    value: Any = Field(default=None)
    found: bool = Field(default=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class BaseMemory(Protocol[V]):
    """Protocol for memory implementations."""

    name: str

    def store(self, key: str, value: V) -> None:
        """Store a value in memory."""
        ...

    def retrieve(self, key: str) -> MemoryResult:
        """Retrieve a value from memory."""
        ...


class MemoryManager(BaseComponent[Dict[str, Any], MemoryResult]):
    """
    Memory manager for Sifaka chains.

    This class provides stateful memory management and context tracking,
    coordinating between multiple memory stores and tracking memory operations.

    ## Architecture
    The MemoryManager follows a component-based architecture:
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
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        memories: List[BaseMemory[Any]],
        name: str = "memory_manager",
        description: str = "Memory manager for Sifaka chains",
        cache_enabled: bool = True,
        max_items: int = 100,
        config: Optional[MemoryConfig] = None,
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
        self._state.update("memories", memories)
        self._state.update("result_cache", {})
        self._state.update("initialized", True)

        # Set metadata
        self._state.set_metadata("component_type", "memory_manager")
        self._state.set_metadata("creation_time", time.time())
        self._state.set_metadata("memory_count", len(memories))

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

        operation = input.get("operation")
        key = input.get("key")
        value = input.get("value")

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
            results = self.retrieve(key)
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
        start_time = time.time()

        try:
            # Get memories from state
            memories = self._state.get("memories", [])
            if not memories:
                logger.warning("No memories available for storage")
                return

            # Store in each memory
            for memory in memories:
                memory.store(key, value)

            # Update statistics
            self.update_statistics(
                MemoryResult(
                    passed=True,
                    message=f"Value stored under key '{key}'",
                    metadata={"operation": "store", "key": key},
                    score=1.0,
                    key=key,
                    value=value,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            )

        except Exception as e:
            # Record error
            self.record_error(e)
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
            logger.warning("Empty key provided for retrieval")
            return []

        # Record start time
        start_time = time.time()

        try:
            # Check cache if enabled
            cache = self._state.get("result_cache", {})

            if key in cache and self.config.cache_enabled:
                self._state.set_metadata("cache_hit", True)
                return cache[key]

            # Mark as cache miss
            self._state.set_metadata("cache_hit", False)

            # Get memories from state
            memories = self._state.get("memories", [])
            if not memories:
                logger.warning("No memories available for retrieval")
                return []

            # Retrieve from each memory
            results = []
            for memory in memories:
                try:
                    result = memory.retrieve(key)
                    results.append(result)
                except Exception as e:
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
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update statistics
            self.update_statistics(retrieval_result)

            # Cache result if caching is enabled
            if self.config.cache_enabled:
                # Manage cache size
                if len(cache) >= self.config.max_items:
                    # Remove oldest entry (simple approach)
                    if cache:
                        oldest_key = next(iter(cache))
                        del cache[oldest_key]

                cache[key] = results
                self._state.update("result_cache", cache)

            return results

        except Exception as e:
            # Record error
            self.record_error(e)
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
        stats.update(
            {
                "cache_size": len(self._state.get("result_cache", {})),
                "memory_count": len(self._state.get("memories", [])),
                "cache_enabled": self.config.cache_enabled,
                "max_items": self.config.max_items,
            }
        )

        return stats

    def clear_cache(self) -> None:
        """Clear the memory result cache."""
        self._state.update("result_cache", {})
        logger.debug(f"Memory cache cleared for {self.name}")

    def warm_up(self) -> None:
        """Prepare the memory manager for use."""
        super().warm_up()

        # Pre-validate memories
        memories = self._state.get("memories", [])
        for memory in memories:
            if hasattr(memory, "warm_up"):
                memory.warm_up()

        logger.debug(f"Memory manager '{self.name}' warmed up with {len(memories)} memories")

    def add_memory(self, memory: Any) -> None:
        """
        Add a memory store.

        ## Overview
        This method adds a new memory store to the manager's memory list.
        The memory will be used in subsequent operations.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate memory type
           - Check memory validity

        2. **Memory Addition**: Add memory
           - Add to memory list
           - Update state
           - Clear cache

        Args:
            memory: The memory store to add

        Raises:
            ValueError: If the memory is invalid
            TypeError: If the input type is incorrect

        Examples:
            ```python
            manager = MemoryManager(memories=[])
            new_memory = SimpleMemory(name="simple_memory")
            manager.add_memory(new_memory)
            ```
        """
        # Validate memory type
        if not isinstance(memory, BaseMemory):
            raise ValueError(f"Expected BaseMemory instance, got {type(memory)}")

        # Check for duplicate memory names
        memories = self._state.get("memories", [])
        if any(m.name == memory.name for m in memories):
            logger.warning(f"Memory with name '{memory.name}' already exists, it will be replaced")
            # Remove existing memory with same name
            self.remove_memory(memory.name)
            # Get updated memories list
            memories = self._state.get("memories", [])

        # Add memory to the list
        memories.append(memory)
        self._state.update("memories", memories)

        # Update metadata
        self._state.set_metadata("memory_count", len(memories))

        # Clear cache since memory operations may change
        self.clear_cache()

        logger.debug(f"Added memory '{memory.name}' to memory manager '{self.name}'")

    def remove_memory(self, memory_name: str) -> None:
        """
        Remove a memory store by name.

        ## Overview
        This method removes a memory store from the manager's memory list
        based on its name. The memory will no longer be used in subsequent
        operations.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate memory name
           - Find memory to remove

        2. **Memory Removal**: Remove memory
           - Remove from memory list
           - Update state
           - Clear cache

        Args:
            memory_name: The name of the memory store to remove

        Raises:
            ValueError: If the memory name is invalid or memory not found

        Examples:
            ```python
            manager = MemoryManager(memories=[...])
            manager.remove_memory("simple_memory")
            ```
        """
        # Validate input
        if not memory_name or not isinstance(memory_name, str):
            raise ValueError(f"Invalid memory name: {memory_name}")

        # Find memory by name
        memory_to_remove = None
        memories = self._state.get("memories", [])
        for memory in memories:
            if memory.name == memory_name:
                memory_to_remove = memory
                break

        if memory_to_remove is None:
            raise ValueError(f"Memory not found: {memory_name}")

        # Remove memory from list
        memories.remove(memory_to_remove)
        self._state.update("memories", memories)

        # Update metadata
        self._state.set_metadata("memory_count", len(memories))

        # Clear cache since memory operations may change
        self.clear_cache()

        logger.debug(f"Removed memory '{memory_name}' from memory manager '{self.name}'")

    def get_memories(self) -> List[BaseMemory]:
        """
        Get all registered memories.

        ## Overview
        This method returns a list of all memories currently
        registered with the manager.

        ## Lifecycle
        1. **Memory Retrieval**: Get memories
           - Access memory list
           - Return memories

        Returns:
            The list of registered memories

        Examples:
            ```python
            manager = MemoryManager(memories=[...])
            memories = manager.get_memories()
            print(f"Number of memories: {len(memories)}")
            ```
        """
        return self._state.get("memories", [])


def create_memory_manager(
    memories: List[BaseMemory] = None,
    name: str = "memory_manager",
    description: str = "Memory manager for Sifaka chains",
    cache_enabled: bool = True,
    max_items: int = 100,
    **kwargs: Any,
) -> MemoryManager:
    """
    Create a memory manager.

    Args:
        memories: List of memories to use for storage
        name: Name of the manager
        description: Description of the manager
        cache_enabled: Whether to enable caching
        max_items: Maximum number of items to store in memory
        **kwargs: Additional configuration parameters

    Returns:
        Configured MemoryManager instance
    """
    config = MemoryConfig(
        name=name,
        description=description,
        cache_enabled=cache_enabled,
        max_items=max_items,
        **kwargs,
    )

    return MemoryManager(
        memories=memories or [],
        name=name,
        description=description,
        config=config,
    )
