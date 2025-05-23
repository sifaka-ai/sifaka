"""
Configuration classes for persistence in Sifaka.

This module provides configuration classes for different persistence backends,
allowing users to customize storage behavior and settings.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PersistenceConfig(BaseModel):
    """Base configuration for persistence backends.

    This class provides common configuration options that apply
    to all persistence backends.

    Attributes:
        enabled: Whether persistence is enabled
        auto_save: Whether to automatically save thoughts
        compression: Whether to enable compression
        encryption: Whether to enable encryption
        backup_enabled: Whether to enable automatic backups
        metadata: Additional configuration metadata
    """

    enabled: bool = True
    auto_save: bool = True
    compression: bool = False
    encryption: bool = False
    backup_enabled: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JSONPersistenceConfig(PersistenceConfig):
    """Configuration for JSON-based persistence.

    This class provides configuration options specific to the
    JSON file-based storage backend.

    Attributes:
        storage_dir: Directory for storing JSON files
        auto_create_dirs: Whether to automatically create directories
        enable_indexing: Whether to maintain search indexes
        max_file_size_mb: Maximum size for individual files
        pretty_print: Whether to format JSON with indentation
        file_permissions: File permissions for created files (octal)
        backup_dir: Directory for backup files
        backup_frequency: How often to create backups (in hours)
        cleanup_old_backups: Whether to automatically clean up old backups
        max_backup_age_days: Maximum age for backup files
    """

    storage_dir: str = "./sifaka_storage"
    auto_create_dirs: bool = True
    enable_indexing: bool = True
    max_file_size_mb: int = 10
    pretty_print: bool = True
    file_permissions: Optional[int] = None
    backup_dir: Optional[str] = None
    backup_frequency: int = 24  # hours
    cleanup_old_backups: bool = True
    max_backup_age_days: int = 30

    def get_storage_path(self) -> Path:
        """Get the storage directory as a Path object."""
        return Path(self.storage_dir).expanduser().resolve()

    def get_backup_path(self) -> Optional[Path]:
        """Get the backup directory as a Path object."""
        if self.backup_dir:
            return Path(self.backup_dir).expanduser().resolve()
        return None


class MilvusPersistenceConfig(PersistenceConfig):
    """Configuration for Milvus-based persistence.

    This class provides configuration options for the Milvus
    vector database storage backend.

    Attributes:
        host: Milvus server host
        port: Milvus server port
        collection_name: Name of the collection to use
        dimension: Dimension of the embedding vectors
        metric_type: Metric type for similarity search
        index_type: Index type for the collection
        index_params: Parameters for index creation
        search_params: Parameters for search operations
        connection_timeout: Connection timeout in seconds
        max_connections: Maximum number of connections
    """

    host: str = "localhost"
    port: int = 19530
    collection_name: str = "sifaka_thoughts"
    dimension: int = 768  # Default for many embedding models
    metric_type: str = "COSINE"
    index_type: str = "IVF_FLAT"
    index_params: Dict[str, Any] = Field(default_factory=lambda: {"nlist": 1024})
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"nprobe": 10})
    connection_timeout: int = 30
    max_connections: int = 10


class RedisPersistenceConfig(PersistenceConfig):
    """Configuration for Redis-based persistence.

    This class provides configuration options for the Redis
    in-memory storage backend.

    Attributes:
        host: Redis server host
        port: Redis server port
        db: Redis database number
        password: Redis password
        connection_timeout: Connection timeout in seconds
        max_connections: Maximum number of connections
        key_prefix: Prefix for all Redis keys
        ttl_seconds: Time-to-live for cached items
        enable_clustering: Whether to enable Redis clustering
        cluster_nodes: List of cluster node addresses
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    connection_timeout: int = 30
    max_connections: int = 10
    key_prefix: str = "sifaka:"
    ttl_seconds: Optional[int] = None  # No expiration by default
    enable_clustering: bool = False
    cluster_nodes: Optional[list[str]] = None


def create_json_config(
    storage_dir: str = "./sifaka_storage", **kwargs: Any
) -> JSONPersistenceConfig:
    """Create a JSON persistence configuration.

    Args:
        storage_dir: Directory for storing JSON files
        **kwargs: Additional configuration options

    Returns:
        Configured JSONPersistenceConfig instance
    """
    return JSONPersistenceConfig(storage_dir=storage_dir, **kwargs)


def create_milvus_config(
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "sifaka_thoughts",
    **kwargs: Any,
) -> MilvusPersistenceConfig:
    """Create a Milvus persistence configuration.

    Args:
        host: Milvus server host
        port: Milvus server port
        collection_name: Name of the collection to use
        **kwargs: Additional configuration options

    Returns:
        Configured MilvusPersistenceConfig instance
    """
    return MilvusPersistenceConfig(host=host, port=port, collection_name=collection_name, **kwargs)


def create_redis_config(
    host: str = "localhost", port: int = 6379, db: int = 0, **kwargs: Any
) -> RedisPersistenceConfig:
    """Create a Redis persistence configuration.

    Args:
        host: Redis server host
        port: Redis server port
        db: Redis database number
        **kwargs: Additional configuration options

    Returns:
        Configured RedisPersistenceConfig instance
    """
    return RedisPersistenceConfig(host=host, port=port, db=db, **kwargs)
