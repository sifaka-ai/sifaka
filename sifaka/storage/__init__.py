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

### MultiStorage
Writes to multiple storage backends simultaneously. Perfect for:
- Redundancy (save to both Redis and files)
- Different access patterns (fast Redis reads, permanent file archive)
- Graceful fallback (continues working if one backend fails)
- Best of all worlds approach

### RedisStorage (Optional)
High-performance storage using Redis with optional RediSearch support.
Features:
- Distributed access and fast retrieval
- Full-text search with RediSearch module
- Filter by critic type, confidence, dates
- Automatic fallback for basic Redis
Requires 'redis' package. Enhanced features with Redis Stack.

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
    >>>
    >>> # Redis with search (requires Redis Stack)
    >>> from sifaka.storage import RedisStorage
    >>> storage = RedisStorage()
    >>> results = await storage.search(\"@critic:{style} improve*\")
    >>> results = await storage.search_advanced(
    ...     critics=[\"style\"],
    ...     min_confidence=0.8
    ... )
    >>>
    >>> # Multi-storage for redundancy and best of both worlds
    >>> from sifaka.storage import MultiStorage, FileStorage, RedisStorage
    >>> storage = MultiStorage([
    ...     FileStorage("./archive"),     # Permanent archive
    ...     RedisStorage(ttl=86400)       # Fast access for 24h
    ... ])
    >>> result = await improve("text", storage=storage)

## Custom Storage:

    >>> from sifaka.storage import StorageBackend
    >>> class MyStorage(StorageBackend):
    ...     async def save(self, result):
    ...         # Custom implementation
    ...         return result.id
"""

from .base import StorageBackend
from .file import FileStorage
from .memory import MemoryStorage
from .multi import MultiStorage

# Optional Redis storage (requires redis package)
try:
    from .redis import RedisStorage

    __all__ = [
        "StorageBackend",
        "MemoryStorage",
        "FileStorage",
        "MultiStorage",
        "RedisStorage",
    ]
except ImportError:
    # Redis not installed
    __all__ = ["StorageBackend", "MemoryStorage", "FileStorage", "MultiStorage"]
