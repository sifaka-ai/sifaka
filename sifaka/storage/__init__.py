"""Simple, flexible storage system for Sifaka.

By default, everything is stored in memory with no persistence. Users can optionally
choose to persist data to files, Redis, or other backends as needed.

The storage system follows a simple protocol:
- get(key) -> value or None
- set(key, value) -> None
- search(query, limit) -> List[results] (for text search)
- clear() -> None

Example:
    ```python
    from sifaka.storage import MemoryStorage, FileStorage, RedisStorage
    from sifaka.core import Chain
    from sifaka.models import create_model

    # Default: Memory only (no persistence)
    model = create_model("openai:gpt-4")
    chain = Chain(model=model, prompt="Write about AI")
    result = chain.run()  # Thoughts stored in memory only

    # Simple file persistence
    chain = Chain(
        model=model,
        prompt="Write about AI",
        storage=FileStorage("./my_thoughts.json")
    )

    # Redis persistence
    chain = Chain(
        model=model,
        storage=RedisStorage(redis_config)
    )

    # Compose storage layers if needed
    storage = CachedStorage(
        memory=MemoryStorage(),
        persistence=FileStorage("./backup.json")
    )
    chain = Chain(model=model, storage=storage)
    ```
"""

from sifaka.storage.cached import CachedStorage
from sifaka.storage.file import FileStorage
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.protocol import Storage
from sifaka.storage.redis import RedisStorage

__all__ = [
    "Storage",
    "MemoryStorage",
    "FileStorage",
    "RedisStorage",
    "CachedStorage",
]
