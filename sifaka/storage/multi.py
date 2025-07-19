"""Multi-storage backend for writing to multiple storage backends simultaneously.

This module provides a storage backend that can write to multiple storage
backends at once, enabling redundancy, different access patterns, and
failover capabilities.

Example:
    >>> from sifaka.storage import MultiStorage, FileStorage, RedisStorage
    >>>
    >>> storage = MultiStorage([
    ...     FileStorage(directory="./results"),
    ...     RedisStorage(prefix="sifaka:"),
    ... ])
    >>>
    >>> result = await improve(text, storage=storage)
    >>> # Automatically saved to both filesystem and Redis
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.models import SifakaResult
from .base import StorageBackend

logger = logging.getLogger(__name__)


class MultiStorage(StorageBackend):
    """Storage backend that writes to multiple backends simultaneously.

    This backend provides redundancy and flexibility by allowing writes
    to multiple storage systems at once. Useful for:

    - Redundancy: Keep copies in multiple locations
    - Performance: Use fast storage for reads, permanent for backup
    - Migration: Gradually move from one backend to another
    - Debugging: Save to both Redis (for search) and files (for inspection)

    Behavior:
    - **Save**: Writes to all backends, returns first successful ID
    - **Load**: Tries each backend until one returns a result
    - **List**: Uses the first backend that succeeds
    - **Search**: Uses the first backend with search capability
    - **Delete**: Attempts deletion from all backends

    Example:
        >>> # Redundant storage with fallback
        >>> storage = MultiStorage([
        ...     RedisStorage(ttl=3600),      # Fast access, temporary
        ...     FileStorage("./archive"),     # Permanent archive
        ... ])
        >>>
        >>> # Save to both, load from fastest available
        >>> result_id = await storage.save(result)
        >>> loaded = await storage.load(result_id)  # Tries Redis first

    Error handling:
        If a backend fails during save, a warning is logged but the
        operation continues with other backends. At least one backend
        must succeed for save() to return an ID.
    """

    def __init__(
        self,
        backends: List[StorageBackend],
        require_all: bool = False,
        primary_backend: Optional[int] = None,
    ):
        """Initialize multi-storage with multiple backends.

        Args:
            backends: List of storage backends to use. Order matters for
                read operations (first backend is tried first).
            require_all: If True, save operations must succeed on all
                backends or raise an exception. Default False allows
                partial success.
            primary_backend: Index of the primary backend for read operations.
                If set, this backend is always tried first for loads and
                searches. Default None uses list order.

        Raises:
            ValueError: If backends list is empty

        Example:
            >>> # Basic multi-storage
            >>> storage = MultiStorage([
            ...     FileStorage("./results"),
            ...     RedisStorage()
            ... ])
            >>>
            >>> # Strict mode - all backends must succeed
            >>> storage = MultiStorage(
            ...     backends=[...],
            ...     require_all=True
            ... )
            >>>
            >>> # Redis primary for reads, file for backup
            >>> storage = MultiStorage(
            ...     backends=[FileStorage(), RedisStorage()],
            ...     primary_backend=1  # Use Redis (index 1) first
            ... )
        """
        if not backends:
            raise ValueError("At least one backend must be provided")

        self.backends = backends
        self.require_all = require_all
        self.primary_backend = primary_backend

        # Reorder backends for read operations if primary is specified
        if primary_backend is not None and 0 <= primary_backend < len(backends):
            self._read_order = [backends[primary_backend]] + [
                b for i, b in enumerate(backends) if i != primary_backend
            ]
        else:
            self._read_order = backends

    async def save(self, result: SifakaResult) -> str:
        """Save result to all configured backends.

        Attempts to save to all backends. Returns the ID from the first
        successful save. If require_all is True, raises exception if any
        backend fails.

        Args:
            result: The SifakaResult to save

        Returns:
            Result ID from the first successful backend

        Raises:
            Exception: If require_all=True and any backend fails
            RuntimeError: If no backends successfully save
        """
        ids = []
        errors = []

        for backend in self.backends:
            try:
                result_id = await backend.save(result)
                ids.append((backend.__class__.__name__, result_id))
                logger.debug(f"Saved to {backend.__class__.__name__}: {result_id}")
            except Exception as e:
                error_msg = f"Failed to save to {backend.__class__.__name__}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)

                if self.require_all:
                    raise Exception(f"Required backend failed: {error_msg}")

        if not ids:
            raise RuntimeError(f"All storage backends failed: {'; '.join(errors)}")

        # Return the first successful ID
        return ids[0][1]

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load result from the first backend that has it.

        Tries backends in read order (primary first if configured).
        Returns the first successful load.

        Args:
            result_id: ID of the result to load

        Returns:
            SifakaResult if found, None otherwise
        """
        for backend in self._read_order:
            try:
                result = await backend.load(result_id)
                if result:
                    logger.debug(f"Loaded from {backend.__class__.__name__}")
                    return result
            except Exception as e:
                logger.debug(f"Load failed from {backend.__class__.__name__}: {e}")
                continue

        return None

    async def list(self, limit: int = 10, offset: int = 0) -> List[str]:
        """List results from the first available backend.

        Uses read order (primary first if configured).

        Args:
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of result IDs
        """
        for backend in self._read_order:
            try:
                results = await backend.list(limit, offset)
                logger.debug(f"Listed from {backend.__class__.__name__}")
                return results
            except Exception as e:
                logger.debug(f"List failed from {backend.__class__.__name__}: {e}")
                continue

        return []

    async def delete(self, result_id: str) -> bool:
        """Delete result from all backends.

        Attempts deletion from all backends. Logs warnings for failures
        but doesn't raise exceptions unless require_all is True.

        Args:
            result_id: ID of the result to delete

        Raises:
            Exception: If require_all=True and any deletion fails
        """
        errors = []

        for backend in self.backends:
            try:
                await backend.delete(result_id)
                logger.debug(f"Deleted from {backend.__class__.__name__}")
            except Exception as e:
                error_msg = f"Failed to delete from {backend.__class__.__name__}: {e}"  # nosec B608
                errors.append(error_msg)
                logger.warning(error_msg)

                if self.require_all:
                    raise Exception(f"Required deletion failed: {error_msg}")

        return len(errors) == 0

    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search using the first backend that supports it.

        Tries backends in read order until one returns results.
        Backends without search capability are skipped.

        Args:
            query: Search query (backend-specific format)
            limit: Maximum results to return

        Returns:
            List of matching result IDs
        """
        for backend in self._read_order:
            try:
                results = await backend.search(query, limit)
                logger.debug(f"Searched using {backend.__class__.__name__}")
                return results
            except NotImplementedError:
                logger.debug(f"{backend.__class__.__name__} doesn't support search")
                continue
            except Exception as e:
                logger.debug(f"Search failed on {backend.__class__.__name__}: {e}")
                continue

        logger.warning("No backend available for search")
        return []

    async def cleanup(self) -> None:
        """Clean up all backend resources.

        Calls cleanup on all backends. Continues even if some fail.
        """
        for backend in self.backends:
            try:
                if hasattr(backend, "cleanup"):
                    await backend.cleanup()
                    logger.debug(f"Cleaned up {backend.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {backend.__class__.__name__}: {e}")

    def get_backend_status(self) -> Dict[str, Any]:
        """Get status information about all backends.

        Returns:
            Dictionary with backend names and their configuration
        """
        return {
            "backends": [b.__class__.__name__ for b in self.backends],
            "require_all": self.require_all,
            "primary_backend": self.primary_backend,
            "primary_name": (
                self.backends[self.primary_backend].__class__.__name__
                if self.primary_backend is not None
                else None
            ),
        }
