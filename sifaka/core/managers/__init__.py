"""
Core managers for Sifaka.

This package provides core manager implementations that can be used across
different components of the Sifaka library.

## Core Managers

1. **KeyValueMemoryManager**
   - Manages key-value based memory storage and retrieval
   - Supports multiple memory stores
   - Implements caching for performance
   - Tracks statistics for monitoring
   - Provides consistent state management

2. **BufferMemoryManager**
   - Manages a circular buffer of string items
   - Supports fixed-size memory with automatic overflow handling
   - Provides simple add/get operations
   - Tracks usage statistics
   - Implements error handling
"""

from .memory import (
    KeyValueMemoryManager,
    MemoryConfig,
    MemoryResult,
    BaseMemory,
    create_key_value_memory_manager,
    BufferMemoryManager,
    create_buffer_memory_manager,
)

__all__ = [
    # Key-Value Memory Management
    "KeyValueMemoryManager",
    "MemoryConfig",
    "MemoryResult",
    "BaseMemory",
    "create_key_value_memory_manager",
    # Buffer Memory Management
    "BufferMemoryManager",
    "create_buffer_memory_manager",
]
