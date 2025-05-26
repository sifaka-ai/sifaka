"""File-based storage implementation.

Simple JSON file persistence for thoughts and other data. Perfect for single-user
applications and development where you want to persist data between runs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class FileStorage:
    """Simple file-based storage using JSON.

    Stores all data in a single JSON file. Data is loaded into memory on startup
    and written back to file on every set operation.

    Perfect for:
    - Single-user applications
    - Development and testing with persistence
    - Small to medium datasets
    - Cases where you want human-readable storage

    Attributes:
        file_path: Path to the JSON storage file.
        data: In-memory cache of the file contents.
    """

    def __init__(self, file_path: Optional[str] = None, directory: Optional[str] = None):
        """Initialize file storage.

        Args:
            file_path: Path to the JSON file for storage.
            directory: Directory to store the default file (alternative to file_path).
        """
        if file_path is not None:
            self.file_path = Path(file_path)
        elif directory is not None:
            self.file_path = Path(directory) / "sifaka_storage.json"
        else:
            self.file_path = Path("sifaka_storage.json")

        self.data: Dict[str, Any] = {}

        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if file exists
        self._load()

        logger.debug(f"Initialized FileStorage at {self.file_path}")

    def _load(self) -> None:
        """Load data from file into memory."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.debug(f"Loaded {len(self.data)} items from {self.file_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load from {self.file_path}: {e}")
                self.data = {}
        else:
            logger.debug(f"File {self.file_path} does not exist, starting with empty storage")

    def _save(self) -> None:
        """Save current data to file."""
        try:
            # Write to temporary file first, then rename for atomic operation
            temp_path = self.file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)

            # Atomic rename
            temp_path.replace(self.file_path)
            logger.debug(f"Saved {len(self.data)} items to {self.file_path}")

        except IOError as e:
            logger.error(f"Failed to save to {self.file_path}: {e}")
            # Clean up temp file if it exists
            temp_path = self.file_path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        value = self.data.get(key)
        logger.debug(f"File get: {key} -> {'found' if value is not None else 'not found'}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key and save to file.

        Args:
            key: The storage key.
            value: The value to store.
        """
        self.data[key] = value
        self._save()
        logger.debug(f"File set: {key} -> stored and saved")

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        For file storage, this just returns all values (no semantic search).

        Args:
            query: The search query (ignored for file storage).
            limit: Maximum number of results to return.

        Returns:
            List of all stored values, limited by the limit parameter.
        """
        values = list(self.data.values())[:limit]
        logger.debug(f"File search: '{query}' -> {len(values)} results")
        return values

    def clear(self) -> None:
        """Clear all stored data and remove file."""
        count = len(self.data)
        self.data.clear()

        # Remove the file
        if self.file_path.exists():
            self.file_path.unlink()

        logger.debug(f"File clear: removed {count} items and deleted {self.file_path}")

    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        return key in self.data

    def save(self, key: str, value: Any) -> None:
        """Save a value for a key (same as set).

        Args:
            key: The storage key.
            value: The value to store.
        """
        self.set(key, value)

    def exists(self, key: str) -> bool:
        """Check if key exists in storage (same as __contains__).

        Args:
            key: The storage key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self.data

    def load(self, key: str) -> Optional[Any]:
        """Load a value by key (same as get).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return self.get(key)
