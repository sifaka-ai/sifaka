"""Flexible multi-backend persistence implementation for Sifaka.

This module provides a highly flexible hybrid storage implementation that can combine
any number of storage backends with configurable routing, failover, and caching.
Perfect for complex architectures like: Memory → PostgreSQL → File → Milvus → S3
"""

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import SifakaBasePersistence
from .memory import MemoryPersistence
from sifaka.core.thought import SifakaThought
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class BackendRole(Enum):
    """Roles that storage backends can play in the hybrid system."""

    CACHE = "cache"  # Fast cache layer (e.g., Memory, Redis)
    PRIMARY = "primary"  # Primary storage (e.g., PostgreSQL, Redis)
    BACKUP = "backup"  # Backup storage (e.g., File, S3)
    SEARCH = "search"  # Search/indexing (e.g., Milvus, Elasticsearch)
    ARCHIVE = "archive"  # Long-term archival (e.g., S3, Glacier)


class BackendConfig:
    """Configuration for a storage backend in the hybrid system."""

    def __init__(
        self,
        backend: SifakaBasePersistence,
        role: BackendRole,
        priority: int = 0,
        read_enabled: bool = True,
        write_enabled: bool = True,
        failover_enabled: bool = True,
        read_repair_target: bool = False,
        name: Optional[str] = None,
    ):
        """Initialize backend configuration.

        Args:
            backend: The storage backend instance
            role: The role this backend plays
            priority: Priority for read operations (lower = higher priority)
            read_enabled: Whether to read from this backend
            write_enabled: Whether to write to this backend
            failover_enabled: Whether this backend can be used for failover
            read_repair_target: Whether to repair this backend when data is found elsewhere
            name: Optional name for the backend (defaults to class name)
        """
        self.backend = backend
        self.role = role
        self.priority = priority
        self.read_enabled = read_enabled
        self.write_enabled = write_enabled
        self.failover_enabled = failover_enabled
        self.read_repair_target = read_repair_target
        self.name = name or type(backend).__name__

        # Runtime statistics
        self.read_count = 0
        self.write_count = 0
        self.error_count = 0
        self.last_error = None


class FlexibleHybridPersistence(SifakaBasePersistence):
    """Flexible multi-backend persistence with configurable routing and failover.

    This implementation supports any number of storage backends with configurable
    roles, priorities, and routing policies. Perfect for complex storage architectures.

    Features:
    - Unlimited number of backends with configurable roles
    - Priority-based read routing (cache → primary → backup → search → archive)
    - Selective write routing based on backend roles and configuration
    - Automatic failover and read repair
    - Per-backend statistics and health monitoring
    - Flexible backend configuration (enable/disable read/write per backend)

    Example Configurations:

    Simple Cache + Primary:
    ```python
    hybrid = FlexibleHybridPersistence([
        BackendConfig(MemoryPersistence(), BackendRole.CACHE, priority=0),
        BackendConfig(PostgreSQLPersistence(), BackendRole.PRIMARY, priority=1),
    ])
    ```

    Complex Multi-Backend:
    ```python
    hybrid = FlexibleHybridPersistence([
        BackendConfig(MemoryPersistence(), BackendRole.CACHE, priority=0),
        BackendConfig(RedisPersistence(), BackendRole.CACHE, priority=1),
        BackendConfig(PostgreSQLPersistence(), BackendRole.PRIMARY, priority=2),
        BackendConfig(SifakaFilePersistence(), BackendRole.BACKUP, priority=3),
        BackendConfig(MilvusPersistence(), BackendRole.SEARCH, priority=4,
                     read_enabled=False, write_enabled=True),  # Write-only search index
        BackendConfig(S3Persistence(), BackendRole.ARCHIVE, priority=5,
                     write_enabled=False),  # Read-only archive
    ])
    ```
    """

    def __init__(
        self,
        backends: List[BackendConfig],
        key_prefix: str = "sifaka",
        write_through: bool = True,
        read_repair: bool = True,
        max_concurrent_writes: int = 5,
    ):
        """Initialize flexible hybrid persistence.

        Args:
            backends: List of backend configurations
            key_prefix: Prefix for storage keys
            write_through: Whether to write to all enabled backends immediately
            read_repair: Whether to repair missing data in faster layers
            max_concurrent_writes: Maximum number of concurrent write operations
        """
        super().__init__(key_prefix)

        # Validate and sort backends by priority
        if not backends:
            raise ValueError("At least one backend must be provided")

        self.backends = sorted(backends, key=lambda b: b.priority)
        self.write_through = write_through
        self.read_repair = read_repair
        self.max_concurrent_writes = max_concurrent_writes

        # Create backend lookup maps
        self.backends_by_role = {}
        for backend_config in self.backends:
            role = backend_config.role
            if role not in self.backends_by_role:
                self.backends_by_role[role] = []
            self.backends_by_role[role].append(backend_config)

        # Global statistics
        self.stats = {
            "total_reads": 0,
            "total_writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "read_repairs": 0,
            "write_failures": 0,
        }

        logger.debug(
            f"Initialized FlexibleHybridPersistence with {len(self.backends)} backends: "
            f"{[b.name for b in self.backends]}"
        )

    def get_read_backends(self) -> List[BackendConfig]:
        """Get backends enabled for reading, sorted by priority."""
        return [b for b in self.backends if b.read_enabled]

    def get_write_backends(self) -> List[BackendConfig]:
        """Get backends enabled for writing."""
        return [b for b in self.backends if b.write_enabled]

    def get_backends_by_role(self, role: BackendRole) -> List[BackendConfig]:
        """Get all backends with a specific role."""
        return self.backends_by_role.get(role, [])

    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data across all enabled write backends."""
        write_backends = self.get_write_backends()

        if not write_backends:
            raise RuntimeError("No write-enabled backends available")

        self.stats["total_writes"] += 1
        errors = []
        successful_writes = 0

        # Prepare write tasks
        async def write_to_backend(backend_config: BackendConfig) -> bool:
            try:
                await backend_config.backend._store_raw(key, data)
                backend_config.write_count += 1
                return True
            except Exception as e:
                backend_config.error_count += 1
                backend_config.last_error = str(e)
                errors.append(f"{backend_config.name}: {e}")
                logger.warning(f"Failed to write to {backend_config.name}: {e}")
                return False

        # Execute writes (with concurrency limit)
        if self.write_through:
            # Write to all backends concurrently
            semaphore = asyncio.Semaphore(self.max_concurrent_writes)

            async def limited_write(backend_config: BackendConfig) -> bool:
                async with semaphore:
                    return await write_to_backend(backend_config)

            results = await asyncio.gather(
                *[limited_write(bc) for bc in write_backends], return_exceptions=True
            )

            successful_writes = sum(1 for r in results if r is True)
        else:
            # Write to backends in priority order until one succeeds
            for backend_config in write_backends:
                if await write_to_backend(backend_config):
                    successful_writes = 1
                    break

        # Check if we had any successful writes
        if successful_writes == 0:
            self.stats["write_failures"] += 1
            raise Exception(f"All write backends failed: {errors}")

        logger.debug(f"Stored key {key} to {successful_writes}/{len(write_backends)} backends")

    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data with multi-backend fallback and read repair."""
        read_backends = self.get_read_backends()

        if not read_backends:
            logger.warning("No read-enabled backends available")
            return None

        self.stats["total_reads"] += 1
        data = None
        successful_backend = None

        # Try backends in priority order
        for backend_config in read_backends:
            try:
                data = await backend_config.backend._retrieve_raw(key)
                if data is not None:
                    backend_config.read_count += 1
                    successful_backend = backend_config

                    # Cache hit for cache backends
                    if backend_config.role == BackendRole.CACHE:
                        self.stats["cache_hits"] += 1

                    logger.debug(f"Retrieved key {key} from {backend_config.name}")
                    break

            except Exception as e:
                backend_config.error_count += 1
                backend_config.last_error = str(e)
                logger.warning(f"Failed to read from {backend_config.name}: {e}")
                continue

        if data is None:
            self.stats["cache_misses"] += 1
            logger.debug(f"Key {key} not found in any backend")
            return None

        # Perform read repair if enabled
        if self.read_repair and successful_backend:
            await self._perform_read_repair(key, data, successful_backend)

        return data

    async def _perform_read_repair(
        self, key: str, data: str, source_backend: BackendConfig
    ) -> None:
        """Repair missing data in higher-priority backends."""
        try:
            repair_tasks = []

            # Find backends that should be repaired (higher priority than source)
            for backend_config in self.backends:
                if (
                    backend_config.priority < source_backend.priority
                    and backend_config.read_repair_target
                    and backend_config.write_enabled
                ):

                    repair_tasks.append(self._repair_backend(backend_config, key, data))

            if repair_tasks:
                # Execute repairs concurrently
                results = await asyncio.gather(*repair_tasks, return_exceptions=True)
                successful_repairs = sum(1 for r in results if r is True)

                if successful_repairs > 0:
                    self.stats["read_repairs"] += 1
                    logger.debug(f"Read repair completed for {key}: {successful_repairs} backends")

        except Exception as e:
            logger.warning(f"Read repair failed for key {key}: {e}")

    async def _repair_backend(self, backend_config: BackendConfig, key: str, data: str) -> bool:
        """Repair a single backend with missing data."""
        try:
            await backend_config.backend._store_raw(key, data)
            logger.debug(f"Repaired {backend_config.name} with key {key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to repair {backend_config.name}: {e}")
            return False

    async def _delete_raw(self, key: str) -> bool:
        """Delete data from all backends."""
        write_backends = self.get_write_backends()
        deleted_count = 0

        # Delete from all write-enabled backends
        for backend_config in write_backends:
            try:
                if await backend_config.backend._delete_raw(key):
                    deleted_count += 1
            except Exception as e:
                backend_config.error_count += 1
                backend_config.last_error = str(e)
                logger.warning(f"Failed to delete from {backend_config.name}: {e}")

        success = deleted_count > 0
        logger.debug(f"Deleted key {key} from {deleted_count}/{len(write_backends)} backends")
        return success

    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys matching the pattern from all backends."""
        all_keys = set()

        # Collect keys from all read-enabled backends
        for backend_config in self.get_read_backends():
            try:
                keys = await backend_config.backend._list_keys(pattern)
                all_keys.update(keys)
            except Exception as e:
                logger.warning(f"Failed to list keys from {backend_config.name}: {e}")

        result = list(all_keys)
        logger.debug(
            f"Listed {len(result)} unique keys from {len(self.get_read_backends())} backends"
        )
        return result

    async def get_backend_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for all backends."""
        backend_stats = {}

        for backend_config in self.backends:
            stats = {
                "role": backend_config.role.value,
                "priority": backend_config.priority,
                "read_enabled": backend_config.read_enabled,
                "write_enabled": backend_config.write_enabled,
                "read_count": backend_config.read_count,
                "write_count": backend_config.write_count,
                "error_count": backend_config.error_count,
                "last_error": backend_config.last_error,
            }

            # Add backend-specific stats if available
            if hasattr(backend_config.backend, "get_stats"):
                try:
                    backend_specific = await backend_config.backend.get_stats()
                    stats["backend_specific"] = backend_specific
                except Exception as e:
                    stats["backend_specific_error"] = str(e)

            backend_stats[backend_config.name] = stats

        return {
            "global_stats": self.stats,
            "backend_stats": backend_stats,
            "backend_count": len(self.backends),
            "roles": list(self.backends_by_role.keys()),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all backends."""
        health_results = {}

        for backend_config in self.backends:
            try:
                # Try a simple operation to test backend health
                test_key = f"{self.key_prefix}:health_check:test"
                test_data = '{"test": true}'

                # Test write if enabled
                write_ok = True
                if backend_config.write_enabled:
                    try:
                        await backend_config.backend._store_raw(test_key, test_data)
                    except Exception as e:
                        write_ok = False
                        logger.debug(f"Health check write failed for {backend_config.name}: {e}")

                # Test read if enabled
                read_ok = True
                if backend_config.read_enabled:
                    try:
                        await backend_config.backend._retrieve_raw(test_key)
                    except Exception as e:
                        read_ok = False
                        logger.debug(f"Health check read failed for {backend_config.name}: {e}")

                # Clean up test data
                if backend_config.write_enabled and write_ok:
                    try:
                        await backend_config.backend._delete_raw(test_key)
                    except Exception:
                        pass  # Ignore cleanup errors

                health_results[backend_config.name] = {
                    "healthy": write_ok and read_ok,
                    "write_ok": write_ok,
                    "read_ok": read_ok,
                    "role": backend_config.role.value,
                    "priority": backend_config.priority,
                }

            except Exception as e:
                health_results[backend_config.name] = {
                    "healthy": False,
                    "error": str(e),
                    "role": backend_config.role.value,
                    "priority": backend_config.priority,
                }

        # Overall health assessment
        healthy_backends = sum(1 for h in health_results.values() if h.get("healthy", False))
        total_backends = len(self.backends)

        return {
            "overall_healthy": healthy_backends > 0,
            "healthy_backends": healthy_backends,
            "total_backends": total_backends,
            "health_percentage": (
                (healthy_backends / total_backends * 100) if total_backends > 0 else 0
            ),
            "backend_health": health_results,
        }

    # PydanticAI BaseStatePersistence interface implementation
    async def snapshot_node(self, state: "SifakaThought", next_node: str) -> None:
        """Snapshot the current state before executing a node."""
        try:
            # Store the thought with a snapshot key
            snapshot_key = f"{self.key_prefix}:snapshot:{state.id}:{next_node}"
            data = await self.serialize_state(state)
            await self._store_raw(snapshot_key, data)

            # Also store as regular thought
            await self.store_thought(state)

            logger.debug(
                f"Flexible hybrid snapshotted state for thought {state.id} before node {next_node}"
            )

        except Exception as e:
            logger.error(f"Failed to snapshot state for thought {state.id}: {e}")
            raise

    async def load_state(self, state_id: str) -> Optional["SifakaThought"]:
        """Load a previously saved state."""
        return await self.retrieve_thought(state_id)
