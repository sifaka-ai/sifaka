"""Central storage manager for unified 3-tier architecture.

This module provides the SifakaStorage class which coordinates all storage
operations across the unified memory → cache → persistence architecture.
"""

from typing import Any, Dict, Optional

from sifaka.mcp import MCPServerConfig
from sifaka.utils.logging import get_logger

from .base import InMemoryStorage, RedisCache, MilvusStorage, CachedStorage
from .protocols import Retriever
from .thoughts import CachedThoughtStorage
from .checkpoints import CachedCheckpointStorage
from .retrievers import CachedRetriever
from .metrics import CachedMetricsStorage


logger = get_logger(__name__)


class SifakaStorage:
    """Central storage manager for all Sifaka components.

    This class provides a unified interface for creating and managing storage
    instances across all Sifaka components. It ensures consistent configuration
    and behavior across thoughts, checkpoints, retrievers, and metrics.

    All storage follows the same 3-tier pattern:
    - L1: In-memory (fastest, LRU eviction)
    - L2: Redis cache (fast, cross-process)
    - L3: Milvus persistence (semantic search, long-term)

    Attributes:
        redis_config: Configuration for Redis MCP server.
        milvus_config: Configuration for Milvus MCP server.
        memory_size: Size limit for in-memory storage.
        cache_ttl: Default TTL for Redis cache.
    """

    def __init__(
        self,
        redis_config: MCPServerConfig,
        milvus_config: MCPServerConfig,
        memory_size: int = 1000,
        cache_ttl: int = 3600,
    ):
        """Initialize storage manager.

        Args:
            redis_config: MCP server configuration for Redis.
            milvus_config: MCP server configuration for Milvus.
            memory_size: Maximum items in memory storage.
            cache_ttl: Default TTL for Redis cache in seconds.
        """
        self.redis_config = redis_config
        self.milvus_config = milvus_config
        self.memory_size = memory_size
        self.cache_ttl = cache_ttl

        # Storage instances (created lazily)
        self._thought_storage: Optional[CachedThoughtStorage] = None
        self._checkpoint_storage: Optional[CachedCheckpointStorage] = None
        self._retriever_cache: Optional[CachedRetriever] = None
        self._metrics_storage: Optional[CachedMetricsStorage] = None

        logger.info(
            f"Initialized SifakaStorage with memory_size={memory_size}, cache_ttl={cache_ttl}"
        )

    def _create_cached_storage(self, collection_name: str, key_prefix: str) -> CachedStorage:
        """Create a new CachedStorage instance.

        Args:
            collection_name: Name for Milvus collection.
            key_prefix: Prefix for Redis keys.

        Returns:
            Configured CachedStorage instance.
        """
        memory = InMemoryStorage(max_size=self.memory_size)
        cache = RedisCache(mcp_config=self.redis_config, ttl=self.cache_ttl, key_prefix=key_prefix)
        persistence = MilvusStorage(mcp_config=self.milvus_config, collection_name=collection_name)

        return CachedStorage(memory, cache, persistence)

    def get_thought_storage(self) -> CachedThoughtStorage:
        """Get thought storage instance.

        Returns:
            CachedThoughtStorage instance for storing thoughts.
        """
        if self._thought_storage is None:
            storage = self._create_cached_storage(
                collection_name="sifaka_thoughts", key_prefix="sifaka:thoughts"
            )
            self._thought_storage = CachedThoughtStorage(storage)
            logger.debug("Created thought storage instance")

        return self._thought_storage

    def get_checkpoint_storage(self) -> CachedCheckpointStorage:
        """Get checkpoint storage instance.

        Returns:
            CachedCheckpointStorage instance for storing chain checkpoints.
        """
        if self._checkpoint_storage is None:
            storage = self._create_cached_storage(
                collection_name="sifaka_checkpoints", key_prefix="sifaka:checkpoints"
            )
            self._checkpoint_storage = CachedCheckpointStorage(storage)
            logger.debug("Created checkpoint storage instance")

        return self._checkpoint_storage

    def get_retriever_cache(self, base_retriever: Retriever) -> CachedRetriever:
        """Get cached retriever wrapper.

        Args:
            base_retriever: Base retriever to wrap with caching.

        Returns:
            CachedRetriever instance that wraps the base retriever.
        """
        storage = self._create_cached_storage(
            collection_name="sifaka_retrievals", key_prefix="sifaka:retrievals"
        )

        cached_retriever = CachedRetriever(storage, base_retriever)
        logger.debug(f"Created cached retriever for {base_retriever.__class__.__name__}")
        return cached_retriever

    def get_metrics_storage(self) -> CachedMetricsStorage:
        """Get metrics storage instance.

        Returns:
            CachedMetricsStorage instance for storing performance metrics.
        """
        if self._metrics_storage is None:
            storage = self._create_cached_storage(
                collection_name="sifaka_metrics", key_prefix="sifaka:metrics"
            )
            self._metrics_storage = CachedMetricsStorage(storage)
            logger.debug("Created metrics storage instance")

        return self._metrics_storage

    def clear_all_caches(self) -> None:
        """Clear all storage caches across all components."""
        logger.info("Clearing all storage caches...")

        if self._thought_storage:
            self._thought_storage.clear()
        if self._checkpoint_storage:
            self._checkpoint_storage.clear()
        if self._metrics_storage:
            self._metrics_storage.clear()

        logger.info("All storage caches cleared")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics.

        Returns:
            Dictionary containing statistics from all storage components.
        """
        stats = {
            "config": {
                "memory_size": self.memory_size,
                "cache_ttl": self.cache_ttl,
                "redis_server": self.redis_config.name,
                "milvus_server": self.milvus_config.name,
            },
            "components": {},
        }

        if self._thought_storage:
            stats["components"]["thoughts"] = self._thought_storage.get_stats()

        if self._checkpoint_storage:
            stats["components"]["checkpoints"] = self._checkpoint_storage.get_stats()

        if self._metrics_storage:
            stats["components"]["metrics"] = self._metrics_storage.get_stats()

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all storage backends.

        Returns:
            Dictionary containing health status of all backends.
        """
        health = {"overall": "healthy", "backends": {}}

        try:
            # Test Redis connection
            test_cache = RedisCache(self.redis_config, ttl=60, key_prefix="health_check")
            test_cache.set("test_key", "test_value")
            test_result = test_cache.get("test_key")
            health["backends"]["redis"] = {  # type: ignore
                "status": "healthy" if test_result == "test_value" else "degraded",
                "server": self.redis_config.name,
            }
        except Exception as e:
            health["backends"]["redis"] = {  # type: ignore
                "status": "unhealthy",
                "error": str(e),
                "server": self.redis_config.name,
            }
            health["overall"] = "degraded"

        try:
            # Test Milvus connection
            test_storage = MilvusStorage(self.milvus_config, "health_check")
            test_storage.set("test_key", "test_value")
            test_result = test_storage.get("test_key")
            health["backends"]["milvus"] = {  # type: ignore
                "status": "healthy" if test_result == "test_value" else "degraded",
                "server": self.milvus_config.name,
            }
        except Exception as e:
            health["backends"]["milvus"] = {  # type: ignore
                "status": "unhealthy",
                "error": str(e),
                "server": self.milvus_config.name,
            }
            health["overall"] = "degraded"

        # Memory is always healthy (local)
        health["backends"]["memory"] = {"status": "healthy", "max_size": self.memory_size}  # type: ignore

        return health


def create_storage_manager(
    redis_config: MCPServerConfig, milvus_config: MCPServerConfig, **kwargs: Any
) -> SifakaStorage:
    """Factory function to create a storage manager.

    Args:
        redis_config: MCP server configuration for Redis.
        milvus_config: MCP server configuration for Milvus.
        **kwargs: Additional arguments for SifakaStorage.

    Returns:
        Configured SifakaStorage instance.
    """
    return SifakaStorage(redis_config, milvus_config, **kwargs)
