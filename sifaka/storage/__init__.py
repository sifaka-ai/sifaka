"""Storage backends for persisting SifakaResult thoughts."""

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
