"""
Persistence layer for Sifaka.

This package provides persistence mechanisms for storing and retrieving Thoughts,
context, and other data structures in the Sifaka framework.

Available persistence backends:
- JSON: File-based JSON storage for thoughts and metadata
- Milvus: Vector database storage for embeddings and semantic search (future)
- Redis: In-memory caching and session storage (future)

Example:
    ```python
    from sifaka.persistence.json import JSONThoughtStorage
    from sifaka.core.thought import Thought

    # Create a storage instance
    storage = JSONThoughtStorage(storage_dir="./thoughts")

    # Create and save a thought
    thought = Thought(prompt="Write a story about AI")
    storage.save_thought(thought)

    # Retrieve the thought
    loaded_thought = storage.get_thought(thought.id)

    # Query thoughts
    recent_thoughts = storage.query_thoughts(limit=10)
    ```
"""

from .base import (
    ThoughtStorage,
    PersistenceError,
    ThoughtQuery,
    ThoughtQueryResult,
)

from .json import JSONThoughtStorage
from .config import (
    JSONPersistenceConfig,
    MilvusPersistenceConfig,
    RedisPersistenceConfig,
    create_json_config,
    create_milvus_config,
    create_redis_config,
)

__all__ = [
    "ThoughtStorage",
    "PersistenceError",
    "ThoughtQuery",
    "ThoughtQueryResult",
    "JSONThoughtStorage",
    "JSONPersistenceConfig",
    "MilvusPersistenceConfig",
    "RedisPersistenceConfig",
    "create_json_config",
    "create_milvus_config",
    "create_redis_config",
]
