"""
Chain Managers Module

This module provides managers for the Sifaka chain system.
These managers handle specific aspects of chain execution.

## Overview
The managers in this module are responsible for handling specific aspects of
chain execution, such as caching, retry logic, and memory management. They
provide standardized interfaces and implementations for these concerns,
enabling consistent behavior across different chain components.

## Managers
1. **CacheManager**: Manages caching of chain results
2. **RetryManager**: Manages retry logic for chain execution
3. **ChainMemoryManager**: Manages key-value memory for chains
4. **ChainBufferMemoryManager**: Manages buffer memory for chains

## Usage Examples
```python
from sifaka.chain.managers import CacheManager, RetryManager, ChainMemoryManager

# Create a cache manager
cache_manager = CacheManager(max_size=100)

# Create a retry manager
retry_manager = RetryManager(max_attempts=3, backoff_factor=2.0)

# Create a memory manager
memory_manager = ChainMemoryManager(max_items=50)
```

## Error Handling
- ValueError: Raised for invalid inputs
- RuntimeError: Raised for operation failures
- Tracks error statistics in state managers

## Configuration
Managers can be configured with various options:
- Cache settings (size, TTL)
- Retry settings (attempts, backoff)
- Memory settings (size, caching)
"""

from .cache import CacheManager
from .retry import RetryManager
from .memory import (
    ChainMemoryManager,
    ChainBufferMemoryManager,
    create_chain_memory_manager,
    create_chain_buffer_memory_manager,
)

__all__ = [
    "CacheManager",
    "RetryManager",
    "ChainMemoryManager",
    "ChainBufferMemoryManager",
    "create_chain_memory_manager",
    "create_chain_buffer_memory_manager",
]
