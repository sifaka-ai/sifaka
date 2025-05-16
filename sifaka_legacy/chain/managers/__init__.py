from typing import Any, List
"""
Chain Managers Module

This module provides managers for the Sifaka chain system.
These managers handle specific aspects of chain execution.

## Overview
The managers in this module are responsible for handling specific aspects of
chain execution, such as caching and retry logic. They provide standardized
interfaces and implementations for these concerns, enabling consistent behavior
across different chain components.

Memory management has been consolidated into the core module and should be
imported directly from sifaka.core.managers.memory.

## Managers
1. **CacheManager**: Manages caching of chain results
2. **RetryManager**: Manages retry logic for chain execution

## Usage Examples
```python
from sifaka.chain.managers import CacheManager, RetryManager
from sifaka.core.managers.memory import KeyValueMemoryManager, BufferMemoryManager

# Create a cache manager
cache_manager = CacheManager(max_size=100)

# Create a retry manager
retry_manager = RetryManager(max_attempts=3, backoff_factor=2.0)

# Create a memory manager
memory_manager = KeyValueMemoryManager(
    memories=[],
    name="memory_manager",
    cache_enabled=True,
    max_items=100
)

# Or use the factory function
from sifaka.core.managers.memory import create_key_value_memory_manager
memory_manager = create_key_value_memory_manager(
    component_type="chain",
    max_items=50
)
```

## Error Handling
- ValueError: Raised for invalid inputs
- RuntimeError: Raised for operation failures
- Tracks error statistics in state managers

## Configuration
Managers can be configured with various options:
- Cache settings (size, TTL)
- Retry settings (attempts, backoff)
"""
from .cache import CacheManager
from .retry import RetryManager
from sifaka.core.managers.memory import KeyValueMemoryManager, BufferMemoryManager, create_key_value_memory_manager, create_buffer_memory_manager
__all__: List[Any] = ['CacheManager', 'RetryManager',
    'KeyValueMemoryManager', 'BufferMemoryManager',
    'create_key_value_memory_manager', 'create_buffer_memory_manager']
