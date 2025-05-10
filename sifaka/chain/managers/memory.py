"""
Memory manager for Sifaka chains.

This module provides the memory manager for Sifaka chains,
enabling stateful memory management and context tracking.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
import time

from pydantic import BaseModel, PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.memory.base import BaseMemory, MemoryResult
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class MemoryManager(BaseComponent):
    """
    Memory manager for Sifaka chains.

    This class provides stateful memory management and context tracking,
    coordinating between multiple memory stores and tracking memory operations.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        memories: List[BaseMemory[OutputType]],
        name: str = "memory_manager",
        description: str = "Memory manager for Sifaka chains",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the memory manager.

        Args:
            memories: List of memories to use for storage
            name: Name of the manager
            description: Description of the manager
            config: Additional configuration
        """
        super().__init__()

        self._state.update("memories", memories)
        self._state.update("name", name)
        self._state.update("description", description)
        self._state.update("config", config or {})
        self._state.update("initialized", True)
        self._state.update("execution_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "memory_manager")
        self._state.set_metadata("creation_time", time.time())

    def store(self, key: str, value: Any) -> None:
        """
        Store a value in memory.

        Args:
            key: The key to store the value under
            value: The value to store

        Raises:
            MemoryError: If storage fails
        """
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Record start time
        start_time = time.time()

        try:
            # Get memories from state
            memories = self._state.get("memories")

            # Store in each memory
            for memory in memories:
                memory.store(key, value)

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Memory storage error: {str(e)}")
            raise

    def retrieve(self, key: str) -> List[MemoryResult]:
        """
        Retrieve a value from memory.

        Args:
            key: The key to retrieve

        Returns:
            List of retrieved values

        Raises:
            MemoryError: If retrieval fails
        """
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Check cache
        cache = self._state.get("result_cache", {})
        if key in cache:
            self._state.set_metadata("cache_hit", True)
            return cache[key]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Record start time
        start_time = time.time()

        try:
            # Get memories from state
            memories = self._state.get("memories")

            # Retrieve from each memory
            results = []
            for memory in memories:
                result = memory.retrieve(key)
                results.append(result)

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

            # Cache result
            cache[key] = results
            self._state.update("result_cache", cache)

            return results

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Memory retrieval error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about memory operations.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "execution_count": self._state.get("execution_count", 0),
            "cache_size": len(self._state.get("result_cache", {})),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "error_count": self._state.get_metadata("error_count", 0),
            "memory_count": len(self._state.get("memories", [])),
        }

    def clear_cache(self) -> None:
        """Clear the memory result cache."""
        self._state.update("result_cache", {})
        logger.debug("Memory cache cleared")

    def add_memory(self, memory: Any) -> None:
        """
        Add a memory store.

        Args:
            memory: The memory store to add

        Raises:
            ValueError: If the memory is invalid
        """
        if not isinstance(memory, BaseMemory):
            raise ValueError(f"Expected BaseMemory instance, got {type(memory)}")

        # Get current config
        current_config = self._state.get("config", {})

        # Add memory to the list
        memories = self._state.get("memories", [])
        memories.append(memory)
        self._state.update("memories", memories)

        # Update config
        self._state.update("config", current_config)

    def remove_memory(self, memory_name: str) -> None:
        """
        Remove a memory store.

        Args:
            memory_name: The name of the memory store to remove

        Raises:
            ValueError: If the memory is not found
        """
        # Find memory by name
        memory_to_remove = None
        memories = self._state.get("memories", [])
        for memory in memories:
            if memory.name == memory_name:
                memory_to_remove = memory
                break

        if memory_to_remove is None:
            raise ValueError(f"Memory '{memory_name}' not found")

        # Remove memory from list
        memories.remove(memory_to_remove)
        self._state.update("memories", memories)

        # Get current config
        current_config = self._state.get("config", {})

        # Update config
        self._state.update("config", current_config)

    def get_memories(self) -> List[Any]:
        """
        Get all memory stores.

        Returns:
            List of memory stores
        """
        return self._state.get("memories", [])
