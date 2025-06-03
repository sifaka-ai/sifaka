"""Storage system for Sifaka using PydanticAI native persistence.

This module provides storage implementations that integrate with PydanticAI's
BaseStatePersistence system for seamless graph state management.

Key Features:
- PydanticAI native persistence integration
- Multiple storage backends (Memory, File, Redis)
- Hybrid multi-backend storage with failover
- MCP tool integration for retrieval
- Complete SifakaThought serialization

Example:
    ```python
    from sifaka.storage import RedisPersistence, HybridPersistence
    from sifaka import SifakaEngine

    # Redis persistence
    redis_persistence = RedisPersistence(mcp_client, key_prefix="sifaka:")
    engine = SifakaEngine(persistence=redis_persistence)

    # Hybrid persistence with failover
    hybrid_persistence = HybridPersistence(
        primary=redis_persistence,
        backup=file_persistence
    )
    engine = SifakaEngine(persistence=hybrid_persistence)
    ```
"""

from .base import SifakaBasePersistence
from .memory import MemoryPersistence
from .file import SifakaFilePersistence
from .redis import RedisPersistence
from .flexible_hybrid import FlexibleHybridPersistence, BackendConfig, BackendRole
from .migration import StorageMigrator, migrate_legacy_file_to_persistence

# from .tools import ThoughtRetrievalTools, create_retrieval_tools  # Temporarily disabled due to pydantic_ai tool import issue

# Backward compatibility alias
HybridPersistence = FlexibleHybridPersistence

# Optional backends (may require additional dependencies)
try:
    from .postgresql import PostgreSQLPersistence

    _POSTGRESQL_AVAILABLE = True
except ImportError:
    _POSTGRESQL_AVAILABLE = False

try:
    from .milvus import MilvusPersistence

    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False

__all__ = [
    "SifakaBasePersistence",
    "MemoryPersistence",
    "SifakaFilePersistence",
    "RedisPersistence",
    "FlexibleHybridPersistence",
    "HybridPersistence",  # Backward compatibility alias
    "BackendConfig",
    "BackendRole",
    "StorageMigrator",
    "migrate_legacy_file_to_persistence",
    "ThoughtRetrievalTools",
    "create_retrieval_tools",
]

# Add optional backends to __all__ if available
if _POSTGRESQL_AVAILABLE:
    from .postgresql import PostgreSQLPersistence

    __all__.append("PostgreSQLPersistence")

if _MILVUS_AVAILABLE:
    from .milvus import MilvusPersistence

    __all__.append("MilvusPersistence")
