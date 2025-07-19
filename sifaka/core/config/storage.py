"""Storage-specific configuration."""

from typing import Any, Optional

from pydantic import Field, field_validator

from ..type_defs import StorageBackendSettings
from ..types import StorageType
from .base import BaseConfig


class StorageConfig(BaseConfig):
    """Configuration for storage backends and persistence.

    Controls where and how Sifaka stores results, thoughts, and cached data.
    Supports file-based, Redis, and custom storage backends.

    Example:
        >>> from sifaka.core.types import StorageType
        >>> storage_config = StorageConfig(
        ...     backend=StorageType.REDIS,
        ...     backend_settings={
        ...         "url": "redis://localhost:6379",
        ...         "ttl": 86400
        ...     }
        ... )
    """

    # Storage backend selection
    backend: StorageType = Field(
        default=StorageType.MEMORY,
        description="Storage backend to use (memory, file, redis)",
    )

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v: Any) -> StorageType:
        """Validate backend type - ONLY StorageType enums allowed."""
        if isinstance(v, StorageType):
            return v
        else:
            available = ", ".join(f"StorageType.{s.name}" for s in StorageType)
            raise ValueError(
                f"Invalid storage backend: {v} (type: {type(v).__name__}). "
                f"Must use StorageType enum values: {available}"
            )

    # File storage settings
    storage_path: Optional[str] = Field(
        default="./sifaka_storage", description="Path for file-based storage"
    )

    file_format: str = Field(
        default="json",
        pattern="^(json|yaml|pickle)$",
        description="File format for storage (json, yaml, pickle)",
    )

    # Redis settings
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL (uses REDIS_URL env var if not set)",
    )

    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password (uses REDIS_PASSWORD env var if not set)",
    )

    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number")

    redis_key_prefix: str = Field(
        default="sifaka:", description="Prefix for all Redis keys"
    )

    # General storage settings
    enable_compression: bool = Field(
        default=False, description="Enable compression for stored data"
    )

    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Time-to-live for stored items in seconds (None = no expiry)",
    )

    max_storage_size_mb: Optional[int] = Field(
        default=None, gt=0, description="Maximum storage size in MB (None = unlimited)"
    )

    # Connection pool settings
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size for network-based backends",
    )

    connection_timeout_seconds: float = Field(
        default=5.0, gt=0, le=30, description="Connection timeout in seconds"
    )

    # Custom backend settings
    backend_settings: StorageBackendSettings = Field(
        default_factory=lambda: StorageBackendSettings(),
        description="Additional settings for storage backend",
    )

    # Thoughts storage
    store_thoughts: bool = Field(
        default=True, description="Store detailed thought process and iterations"
    )

    thoughts_format: str = Field(
        default="json",
        pattern="^(json|yaml)$",
        description="Format for thoughts storage",
    )

    def get_redis_config(self) -> StorageBackendSettings:
        """Get Redis configuration as a dictionary."""
        import os

        config: StorageBackendSettings = {
            "url": self.redis_url or os.getenv("REDIS_URL", "redis://localhost:6379"),
            "password": self.redis_password or os.getenv("REDIS_PASSWORD"),
            "db": self.redis_db,
        }

        # Add connection settings (not defined in TypedDict but allowed by total=False)
        return config

    def get_file_storage_path(self) -> str:
        """Get the full path for file storage."""
        import os

        path = self.storage_path or "./sifaka_storage"
        return os.path.abspath(os.path.expanduser(path))

    def should_store_result(self, result_size_bytes: int) -> bool:
        """Check if a result should be stored based on size limits."""
        if self.max_storage_size_mb is None:
            return True

        max_bytes = self.max_storage_size_mb * 1024 * 1024
        return result_size_bytes <= max_bytes
