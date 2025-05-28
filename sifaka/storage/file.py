"""File-based storage implementation.

Simple JSON file persistence for thoughts and other data. Perfect for single-user
applications and development where you want to persist data between runs.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

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

    def __init__(
        self,
        file_path: Optional[str] = None,
        directory: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Initialize file storage.

        Args:
            file_path: Path to the JSON file for storage.
            directory: Directory to store the default file (alternative to file_path).
            overwrite: If True, start with empty storage (overwrite existing file).
                      If False, load existing data and append to it (default behavior).
        """
        if file_path is not None:
            self.file_path = Path(file_path)
        elif directory is not None:
            self.file_path = Path(directory) / "sifaka_storage.json"
        else:
            self.file_path = Path("sifaka_storage.json")

        self.data: Dict[str, Any] = {}
        self.overwrite = overwrite

        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if file exists and overwrite is False
        if not overwrite:
            self._load()
        else:
            # If overwrite is True, start with empty storage and remove existing file
            if self.file_path.exists():
                self.file_path.unlink()
                logger.debug(f"Removed existing file {self.file_path} due to overwrite=True")

        logger.debug(f"Initialized FileStorage at {self.file_path} (overwrite={overwrite})")

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

    async def _load_async(self) -> None:
        """Load data from file into memory asynchronously."""
        if self.file_path.exists():
            try:
                async with aiofiles.open(self.file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    self.data = json.loads(content)
                logger.debug(f"Loaded {len(self.data)} items from {self.file_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load from {self.file_path}: {e}")
                self.data = {}
        else:
            logger.debug(f"File {self.file_path} does not exist, starting with empty storage")

    async def _save_async(self) -> None:
        """Save current data to file asynchronously."""
        try:
            # Write to temporary file first, then rename for atomic operation
            temp_path = self.file_path.with_suffix(".tmp")
            content = json.dumps(self.data, indent=2, ensure_ascii=False, default=str)

            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(content)

            # Atomic rename
            temp_path.replace(self.file_path)
            logger.debug(f"Saved {len(self.data)} items to {self.file_path}")

        except IOError as e:
            logger.error(f"Failed to save to {self.file_path}: {e}")
            # Clean up temp file if it exists
            temp_path = self.file_path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()

    # Internal async methods (required by Storage protocol)
    async def _get_async(self, key: str) -> Optional[Any]:
        """Get a value by key asynchronously (internal method).

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        value = self.data.get(key)
        logger.debug(f"File get: {key} -> {'found' if value is not None else 'not found'}")

        # If the value is a dictionary and looks like a Thought, try to reconstruct it
        if value is not None and isinstance(value, dict) and "id" in value and "prompt" in value:
            try:
                from sifaka.core.thought import Thought

                return Thought.from_dict(value)
            except Exception as e:
                logger.debug(f"Failed to reconstruct Thought from dict: {e}")
                # Return the raw dict if reconstruction fails
                return value

        return value

    async def _set_async(self, key: str, value: Any) -> None:
        """Set a value for a key and save to file asynchronously (internal method).

        Args:
            key: The storage key.
            value: The value to store.
        """
        # If the value is a Thought object, convert it to a dict for JSON serialization
        if hasattr(value, "model_dump"):
            # This is likely a Pydantic model (like Thought)
            stored_value = value.model_dump()
        else:
            stored_value = value

        self.data[key] = stored_value
        await self._save_async()
        logger.debug(f"File set: {key} -> stored and saved")

    async def _search_async(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query asynchronously (internal method).

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

    async def _clear_async(self) -> None:
        """Clear all stored data and remove file asynchronously (internal method)."""
        count = len(self.data)
        self.data.clear()

        # Remove the file
        if self.file_path.exists():
            self.file_path.unlink()

        logger.debug(f"File clear: removed {count} items and deleted {self.file_path}")

    async def _delete_async(self, key: str) -> bool:
        """Delete a value by key asynchronously (internal method).

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        if key in self.data:
            del self.data[key]
            await self._save_async()
            logger.debug(f"File delete: {key} -> deleted")
            return True
        else:
            logger.debug(f"File delete: {key} -> not found")
            return False

    async def _keys_async(self) -> List[str]:
        """Get all keys asynchronously (internal method).

        Returns:
            List of all storage keys.
        """
        return list(self.data.keys())

    # Public sync methods (backward compatible API)
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key.

        Args:
            key: The storage key.

        Returns:
            The stored value, or None if not found.
        """
        return asyncio.run(self._get_async(key))

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key and save to file.

        Args:
            key: The storage key.
            value: The value to store.
        """
        return asyncio.run(self._set_async(key, value))

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for items matching a query.

        For file storage, this just returns all values (no semantic search).

        Args:
            query: The search query (ignored for file storage).
            limit: Maximum number of results to return.

        Returns:
            List of all stored values, limited by the limit parameter.
        """
        return asyncio.run(self._search_async(query, limit))

    def clear(self) -> None:
        """Clear all stored data and remove file."""
        return asyncio.run(self._clear_async())

    def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The storage key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        return asyncio.run(self._delete_async(key))

    def keys(self) -> List[str]:
        """Get all keys in storage.

        Returns:
            List of all storage keys.
        """
        return asyncio.run(self._keys_async())

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
