"""Unified storage system for Sifaka.

This package provides a unified 3-tier storage architecture across all Sifaka components:

Tier 1 (L1): In-Memory Storage
- Fastest access (microseconds)
- LRU eviction for memory management
- Process-local hot data cache

Tier 2 (L2): Redis Cache
- Fast network access (1-5ms)
- Cross-process shared cache
- TTL-based expiration

Tier 3 (L3): Milvus Persistence
- Persistent storage with semantic search (10-1000ms)
- Vector similarity search capabilities
- Long-term data retention

All storage components follow the same pattern:
- Get: Check L1 → L2 → L3, cache results in faster tiers
- Set: Save to L1, async save to L2 + L3
- Search: Use L3 for semantic/vector search, cache results

Example:
    ```python
    from sifaka.storage import SifakaStorage
    from sifaka.mcp import MCPServerConfig, MCPTransportType

    # Configure storage backends
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379"
    )

    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus"
    )

    # Create unified storage manager
    storage = SifakaStorage(redis_config, milvus_config)

    # Get component-specific storage
    thought_storage = storage.get_thought_storage()
    checkpoint_storage = storage.get_checkpoint_storage()
    cached_retriever = storage.get_retriever_cache(base_retriever)
    ```
"""

from .base import (
    StorageError,
    InMemoryStorage,
    RedisCache,
    MilvusStorage,
    CachedStorage,
)

from .manager import SifakaStorage
from .protocols import Retriever

from .thoughts import CachedThoughtStorage, ThoughtStorage, ThoughtQuery, ThoughtQueryResult
from .checkpoints import CachedCheckpointStorage, ChainCheckpoint
from .retrievers import CachedRetriever
from .metrics import CachedMetricsStorage

__all__ = [
    # Core storage classes
    "StorageError",
    "InMemoryStorage",
    "RedisCache",
    "MilvusStorage",
    "CachedStorage",
    # Storage manager
    "SifakaStorage",
    # Protocols
    "Retriever",
    # Component-specific storage
    "CachedThoughtStorage",
    "ThoughtStorage",
    "ThoughtQuery",
    "ThoughtQueryResult",
    "CachedCheckpointStorage",
    "ChainCheckpoint",
    "CachedRetriever",
    "CachedMetricsStorage",
]
