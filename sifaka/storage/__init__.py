"""Storage backends for persisting text improvement results.

This module provides various storage backends for saving and retrieving
SifakaResult objects. Storage enables:

- **Persistence**: Save improvement sessions for later analysis
- **Auditing**: Track how text evolved through iterations
- **Analytics**: Build datasets of improvements for research
- **Sharing**: Export results across systems or teams

## Available Backends:

### MemoryStorage
In-memory storage for temporary results. Data is lost when the process ends.
Useful for testing or when persistence isn't needed.

### FileStorage
Saves results as JSON files in a local directory. Simple and portable,
works everywhere without dependencies. Default storage backend.

### RedisStorage (Optional)
High-performance storage using Redis. Supports distributed access and
fast retrieval. Requires 'redis' package to be installed.

### PostgresStorage (Optional)
Database storage with full query capabilities. Best for production
deployments with complex search needs. Requires 'asyncpg' package.

## Usage:

    >>> from sifaka import improve, FileStorage
    >>> storage = FileStorage(directory=\"./results\")
    >>> result = await improve(\"text\", storage=storage)
    >>> 
    >>> # Later, load the result
    >>> loaded = await storage.load(result.id)

## Custom Storage:

    >>> from sifaka.storage import StorageBackend
    >>> class MyStorage(StorageBackend):
    ...     async def save(self, result):
    ...         # Custom implementation
    ...         return result.id
"""

from .base import StorageBackend
from .memory import MemoryStorage
from .file import FileStorage

# Optional Redis storage (requires redis package)
try:
    from .redis import RedisStorage

    __all__ = ["StorageBackend", "MemoryStorage", "FileStorage", "RedisStorage"]
except ImportError:
    # Redis not installed
    __all__ = ["StorageBackend", "MemoryStorage", "FileStorage"]
