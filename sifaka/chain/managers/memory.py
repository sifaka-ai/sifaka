"""
Chain Memory Management Module

This module provides memory management capabilities for the Sifaka chain system.
It re-exports the core memory management components with chain-specific
customizations to ensure consistent memory management across all components.

## Overview
This module provides memory management implementations that can be used
by chain components. It offers two main memory manager implementations:

1. ChainMemoryManager: For key-value based memory storage and retrieval
2. ChainBufferMemoryManager: For simple circular buffer storage of string items

Both implementations use StateManager for consistent state management
and provide statistics tracking.

## Components
- ChainMemoryManager: Key-value based memory manager for chains
- ChainBufferMemoryManager: Circular buffer memory manager for chains
- Factory functions for creating memory managers

## Usage Examples
```python
from sifaka.chain.managers.memory import create_chain_memory_manager, create_chain_buffer_memory_manager

# Create a key-value memory manager
kv_memory = create_chain_memory_manager(
    name="chain_memory",
    cache_enabled=True,
    max_items=100
)

# Store a value
kv_memory.store("key1", "value1")

# Retrieve a value
results = kv_memory.retrieve("key1")

# Create a buffer memory manager
buffer_memory = create_chain_buffer_memory_manager(buffer_size=10)

# Add items to buffer
buffer_memory.add_to_memory("Item 1")
buffer_memory.add_to_memory("Item 2")

# Retrieve items
items = buffer_memory.get_memory()
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

from typing import Any, Dict, List, Optional
import time

from sifaka.core.managers.memory import (
    KeyValueMemoryManager as CoreKeyValueMemoryManager,
    BufferMemoryManager as CoreBufferMemoryManager,
    MemoryConfig,
    MemoryResult,
    BaseMemory,
)
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class ChainMemoryManager(CoreKeyValueMemoryManager):
    """
    Key-value memory manager for Sifaka chains.

    This class extends the core KeyValueMemoryManager with chain-specific functionality.
    It provides stateful memory management and context tracking for chain components.

    ## Architecture
    The ChainMemoryManager follows a component-based architecture:
    - Extends the core KeyValueMemoryManager for consistent behavior
    - Uses StateManager for state management
    - Implements caching for performance
    - Tracks statistics for monitoring
    - Provides chain-specific memory operations

    ## Lifecycle
    1. **Initialization**: Manager is created with memories and configuration
    2. **State Setup**: State manager is initialized with memory state
    3. **Operation**: Manager handles memory operations (store, retrieve)
    4. **Cleanup**: Resources are released when no longer needed

    ## Error Handling
    - ValueError: Raised for invalid inputs
    - RuntimeError: Raised for memory operation failures
    - Tracks error statistics in state manager

    Attributes:
        config (MemoryConfig): The memory manager configuration
        _state_manager (StateManager): The state manager for the memory manager
    """

    def __init__(
        self,
        memories: List[BaseMemory[Any]],
        name: str = "chain_memory_manager",
        description: str = "Memory manager for Sifaka chains",
        cache_enabled: bool = True,
        max_items: int = 100,
        config: Optional[MemoryConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize the chain memory manager.

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

        # Initialize the core memory manager
        super().__init__(
            memories=memories,
            name=name,
            description=description,
            cache_enabled=cache_enabled,
            max_items=max_items,
            config=config,
            **kwargs,
        )

        # Log initialization
        logger.debug(f"Chain memory manager '{name}' initialized with {len(memories)} memories")


class ChainBufferMemoryManager(CoreBufferMemoryManager):
    """
    Circular buffer memory manager for Sifaka chains.

    This class extends the core BufferMemoryManager with chain-specific functionality.
    It provides a simple circular buffer for storing string items for chain components.

    ## Architecture
    The ChainBufferMemoryManager follows a component-based architecture:
    - Extends the core BufferMemoryManager for consistent behavior
    - Uses StateManager for state management
    - Implements a circular buffer for memory storage
    - Tracks statistics for monitoring
    - Provides chain-specific memory operations

    ## Lifecycle
    1. **Initialization**: Manager is created with buffer size and configuration
    2. **State Setup**: State manager is initialized with memory state
    3. **Operation**: Manager handles memory operations (add, get, clear)
    4. **Cleanup**: Resources are released when no longer needed

    ## Error Handling
    - ValueError: Raised for invalid inputs
    - RuntimeError: Raised for memory operation failures
    - Tracks error statistics in state manager

    Attributes:
        _state_manager (StateManager): The state manager for the memory manager
    """

    def __init__(
        self,
        buffer_size: int = 10,
        name: str = "chain_buffer_memory_manager",
        description: str = "Buffer memory manager for Sifaka chains",
        **kwargs: Any,
    ):
        """
        Initialize the chain buffer memory manager.

        Args:
            buffer_size: Size of the circular buffer
            name: Name of the manager
            description: Description of the manager
            **kwargs: Additional keyword arguments for configuration
        """
        # Initialize the core buffer memory manager
        super().__init__(
            buffer_size=buffer_size,
            name=name,
            description=description,
            **kwargs,
        )

        # Log initialization
        logger.debug(f"Chain buffer memory manager '{name}' initialized with buffer size {buffer_size}")


def create_chain_memory_manager(
    memories: Optional[List[BaseMemory[Any]]] = None,
    name: str = "chain_memory_manager",
    description: str = "Memory manager for Sifaka chains",
    cache_enabled: bool = True,
    max_items: int = 100,
    **kwargs: Any,
) -> ChainMemoryManager:
    """
    Create a chain memory manager.

    This factory function creates a chain memory manager with the specified
    configuration, following the standardized factory pattern used across
    Sifaka components.

    Args:
        memories: List of memories to use for storage
        name: Name of the manager
        description: Description of the manager
        cache_enabled: Whether to enable caching
        max_items: Maximum number of items to store in memory
        **kwargs: Additional keyword arguments for configuration

    Returns:
        A chain memory manager instance

    Example:
        ```python
        # Create a memory manager with default settings
        memory_manager = create_chain_memory_manager()

        # Create a memory manager with custom settings
        custom_manager = create_chain_memory_manager(
            name="custom_chain_memory",
            cache_enabled=False,
            max_items=50
        )
        ```
    """
    config = MemoryConfig(
        name=name,
        description=description,
        cache_enabled=cache_enabled,
        max_items=max_items,
        **kwargs,
    )

    return ChainMemoryManager(
        memories=memories or [],
        name=name,
        description=description,
        config=config,
    )


def create_chain_buffer_memory_manager(
    buffer_size: int = 10,
    name: str = "chain_buffer_memory_manager",
    description: str = "Buffer memory manager for Sifaka chains",
    **kwargs: Any,
) -> ChainBufferMemoryManager:
    """
    Create a chain buffer memory manager.

    This factory function creates a chain buffer memory manager with the specified
    configuration, following the standardized factory pattern used across
    Sifaka components.

    Args:
        buffer_size: Size of the circular buffer
        name: Name of the manager
        description: Description of the manager
        **kwargs: Additional keyword arguments for configuration

    Returns:
        A chain buffer memory manager instance

    Example:
        ```python
        # Create a buffer memory manager with default settings
        buffer_memory = create_chain_buffer_memory_manager()

        # Create a buffer memory manager with custom settings
        custom_buffer = create_chain_buffer_memory_manager(
            buffer_size=20,
            name="custom_chain_buffer"
        )
        ```
    """
    return ChainBufferMemoryManager(
        buffer_size=buffer_size,
        name=name,
        description=description,
        **kwargs,
    )
