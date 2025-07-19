"""File-based storage backend for persisting SifakaResults to disk.

This module provides a simple, reliable file-based storage implementation that:
- Stores results as JSON files in a designated directory
- Supports save, load, list, delete, and search operations
- Uses async I/O for non-blocking file operations
- Includes automatic cleanup for old files

## Design Principles:

1. **Simplicity**: Direct file system storage without external dependencies
2. **Portability**: JSON format ensures cross-platform compatibility
3. **Debuggability**: Human-readable files for easy inspection
4. **Performance**: Async I/O prevents blocking during file operations

## File Naming Convention:

Files are named using one of two patterns:
- Simple: `{result_id}.json` (for direct ID lookups)
- Descriptive: `{critics}_{timestamp}_{id_prefix}.json` (for organization)

## Usage:

    >>> storage = FileStorage("./thoughts")
    >>>
    >>> # Save a result
    >>> result_id = await storage.save(sifaka_result)
    >>>
    >>> # Load it back
    >>> loaded = await storage.load(result_id)
    >>>
    >>> # Search by content
    >>> matching_ids = await storage.search("machine learning")
    >>>
    >>> # Clean up old files
    >>> deleted = storage.cleanup_old_files(days_old=30)

## Error Handling:

All file operations are wrapped with appropriate error handling:
- Permission errors → StorageError with helpful context
- Disk space issues → StorageError with operation details
- Corrupt files → Graceful handling during search/list

## Limitations:

- No built-in backup/versioning (use external tools)
- Simple search (full text scan, not indexed)
- No concurrent write protection (single process assumed)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles  # type: ignore[import-untyped]

from ..core.exceptions import StorageError
from ..core.models import SifakaResult
from .base import StorageBackend


class FileStorage(StorageBackend):
    """File-based storage backend for persistent storage.

    Provides a simple, reliable way to persist SifakaResults to the local
    file system. Each result is stored as a JSON file, making it easy to
    inspect, backup, and migrate data.

    Key features:
    - Automatic directory creation
    - Descriptive file naming with critic names and timestamps
    - Partial ID matching for convenient lookups
    - Async I/O throughout for non-blocking operations
    - Built-in cleanup for old files

    Example:
        >>> # Initialize with custom directory
        >>> storage = FileStorage("./my_thoughts")
        >>>
        >>> # Save a result (returns ID)
        >>> result_id = await storage.save(my_result)
        >>> print(f"Saved as: {result_id}")
        >>>
        >>> # Load by full or partial ID
        >>> loaded = await storage.load(result_id[:8])
        >>>
        >>> # List recent results
        >>> recent_ids = await storage.list(limit=10)
        >>>
        >>> # Clean up old files
        >>> deleted = storage.cleanup_old_files(days_old=7)
        >>> print(f"Deleted {deleted} old files")

    Attributes:
        storage_dir: Path to the directory where files are stored.
            Created automatically if it doesn't exist.
    """

    def __init__(self, storage_dir: str = "./thoughts"):
        """Initialize file storage with the specified directory.

        Args:
            storage_dir: Directory path for storing JSON files.
                Defaults to "./thoughts" in the current directory.
                Directory is created if it doesn't exist.

        Example:
            >>> # Use default directory
            >>> storage = FileStorage()
            >>>
            >>> # Use custom directory
            >>> storage = FileStorage("/data/sifaka_results")
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def _get_file_path(
        self, result_id: str, critics: Optional[List[str]] = None
    ) -> Path:
        """Generate file path for a result ID.

        Creates descriptive filenames that include critic names and timestamps
        when available, making it easier to browse and organize results.

        Args:
            result_id: Unique identifier for the result (UUID)
            critics: Optional list of critic names used. If provided,
                creates a more descriptive filename.

        Returns:
            Path object pointing to where the file should be stored

        File naming patterns:
        - With critics: `{critics}_{timestamp}_{id_prefix}.json`
          Example: "style_reflexion_20240115_143022_a1b2c3d4.json"
        - Without critics: `{result_id}.json`
          Example: "a1b2c3d4-e5f6-7890-abcd-ef1234567890.json"
        """
        if critics:
            critics_str = "_".join(critics)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self.storage_dir / f"{critics_str}_{timestamp}_{result_id[:8]}.json"
        return self.storage_dir / f"{result_id}.json"

    async def save(self, result: SifakaResult) -> str:
        """Save a SifakaResult to disk as JSON.

        Extracts critic names from the result to create a descriptive filename,
        then serializes the result to JSON with proper formatting.

        Args:
            result: The SifakaResult to save. Must have a valid ID.
                All nested objects (generations, critiques, etc.) are
                automatically serialized.

        Returns:
            The result ID for future retrieval

        Raises:
            StorageError: If save fails due to permissions, disk space,
                or other I/O errors. Includes specific error details.

        Example:
            >>> result = await sifaka.improve("My text")
            >>> storage = FileStorage()
            >>> saved_id = await storage.save(result)
            >>> print(f"Saved with ID: {saved_id}")

        Note:
            Files are saved with indentation for readability. Large results
            may create sizeable files - consider cleanup_old_files() for
            maintenance.
        """
        # Extract critics from the result
        critics = []
        if result.critiques:
            critics = list(
                set(c.critic for c in result.critiques if c.critic != "system")
            )

        file_path = self._get_file_path(result.id, critics)

        try:
            # Convert to JSON-serializable dict
            data = result.model_dump(mode="json")

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, default=str))

            return result.id
        except (OSError, PermissionError) as e:
            raise StorageError(
                f"Failed to save result {result.id}: {e}",
                storage_type="file",
                operation="save",
            )

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a SifakaResult from disk by ID.

        Supports both full and partial ID matching. First tries exact filename
        match, then searches for files containing the ID prefix.

        Args:
            result_id: Full or partial result ID. Can be:
                - Full UUID: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
                - Prefix (min 8 chars): "a1b2c3d4"

        Returns:
            The loaded SifakaResult if found, None if not found

        Raises:
            StorageError: If file exists but cannot be read or parsed.
                Does NOT raise for missing files (returns None).

        Example:
            >>> # Load by full ID
            >>> result = await storage.load("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
            >>>
            >>> # Load by prefix (convenient for CLI)
            >>> result = await storage.load("a1b2c3d4")
            >>>
            >>> if result:
            ...     print(f"Loaded: {result.final_text[:50]}...")
            ... else:
            ...     print("Result not found")

        Performance note:
            Partial ID matching requires scanning directory entries, so
            full IDs are faster for large directories.
        """
        # Try exact match first
        file_path = self._get_file_path(result_id)
        if file_path.exists():
            return await self._load_from_path(file_path)

        # Try to find by ID prefix
        for f in self.storage_dir.glob(f"*{result_id[:8]}*.json"):
            try:
                result = await self._load_from_path(f)
                if result and result.id == result_id:
                    return result
            except Exception:
                continue

        return None

    async def _load_from_path(self, file_path: Path) -> Optional[SifakaResult]:
        """Load a SifakaResult from a specific file path.

        Internal method that handles the actual file reading and deserialization.
        Includes proper error handling for various failure modes.

        Args:
            file_path: Path object pointing to the JSON file

        Returns:
            Deserialized SifakaResult or None if file doesn't exist

        Raises:
            StorageError: For permission errors, I/O errors, or corrupt JSON.
                Missing files return None rather than raising.

        Note:
            Uses Pydantic's model_validate for robust deserialization with
            validation. Handles version compatibility automatically.
        """

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                return SifakaResult.model_validate(data)
        except FileNotFoundError:
            return None  # Result doesn't exist
        except (OSError, PermissionError) as e:
            raise StorageError(
                f"Failed to load from {file_path}: {e}",
                storage_type="file",
                operation="load",
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise StorageError(
                f"Failed to parse from {file_path}: {e}",
                storage_type="file",
                operation="load",
            )

    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List stored result IDs, newest first.

        Returns a paginated list of result IDs sorted by modification time.
        Useful for displaying recent results or implementing a UI.

        Args:
            limit: Maximum number of IDs to return. Defaults to 100.
                Set to smaller values for better performance.
            offset: Number of results to skip. Use for pagination.
                Example: offset=100 with limit=100 gets results 101-200.

        Returns:
            List of result IDs (filenames without .json extension).
            May return fewer than `limit` if not enough results exist.

        Example:
            >>> # Get first page of results
            >>> page1 = await storage.list(limit=20, offset=0)
            >>>
            >>> # Get second page
            >>> page2 = await storage.list(limit=20, offset=20)
            >>>
            >>> # Get all results (careful with large directories)
            >>> all_ids = await storage.list(limit=10000)

        Performance:
            Lists directory contents and sorts by mtime. For very large
            directories (>10k files), consider using smaller limits.
        """
        json_files = list(self.storage_dir.glob("*.json"))

        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Extract IDs from filenames
        ids = [f.stem for f in json_files[offset : offset + limit]]
        return ids

    async def delete(self, result_id: str) -> bool:
        """Delete a stored result from disk.

        Permanently removes the JSON file for the given result ID.
        Only exact ID matches are deleted (no partial matching).

        Args:
            result_id: The exact result ID to delete

        Returns:
            True if file was deleted, False if file didn't exist

        Example:
            >>> # Delete a specific result
            >>> deleted = await storage.delete("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
            >>> if deleted:
            ...     print("Result deleted")
            ... else:
            ...     print("Result not found")

        Note:
            Deletion is permanent. Consider implementing a soft delete
            or backup mechanism if you need recovery options.
        """
        # Try exact match first
        file_path = self._get_file_path(result_id)
        if file_path.exists():
            file_path.unlink()
            return True

        # Try to find by ID prefix (for descriptive filenames)
        for f in self.storage_dir.glob(f"*{result_id[:8]}*.json"):
            try:
                # Load to verify it's the right file
                result = await self._load_from_path(f)
                if result and result.id == result_id:
                    f.unlink()
                    return True
            except Exception:
                continue

        return False

    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search results by text content.

        Performs a simple case-insensitive substring search across both
        original and final text of all stored results. Results are returned
        in the order they're found (typically newest first).

        Args:
            query: Search term to look for in result texts.
                Case-insensitive substring match.
            limit: Maximum number of matching results to return.
                Defaults to 10 to prevent excessive I/O.

        Returns:
            List of result IDs that contain the search query

        Example:
            >>> # Search for results about machine learning
            >>> ml_results = await storage.search("machine learning")
            >>>
            >>> # Display matches
            >>> for result in ml_results:
            ...     print(f"{result.id[:8]}: {result.final_text[:50]}...")

        Performance warning:
            This performs a full scan of all files, loading and searching
            each one. For large collections, this can be slow. Consider:
            - Using smaller search limits
            - Implementing an external search index
            - Caching frequently accessed results

        Error handling:
            Corrupt or inaccessible files are logged and skipped rather
            than failing the entire search.
        """
        matches = []
        query_lower = query.lower()

        for file_path in self.storage_dir.glob("*.json"):
            try:
                result = await self.load(file_path.stem)
                if result:
                    # Simple text search
                    if (
                        query_lower in result.original_text.lower()
                        or query_lower in result.final_text.lower()
                    ):
                        matches.append(result.id)

                    if len(matches) >= limit:
                        break

            except Exception as e:
                # Log error but continue searching other files
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Error reading result file {file_path}: {e}")
                continue

        return matches

    def cleanup_old_files(self, days_old: int = 30) -> int:
        """Clean up files older than specified days.

        Maintenance method to prevent unbounded growth of stored results.
        Deletes JSON files based on modification time (not creation time).

        Args:
            days_old: Delete files older than this many days.
                Defaults to 30 days. Use smaller values for more
                aggressive cleanup.

        Returns:
            Number of files deleted

        Example:
            >>> # Clean up files older than a week
            >>> deleted = storage.cleanup_old_files(days_old=7)
            >>> print(f"Deleted {deleted} old result files")
            >>>
            >>> # Very aggressive cleanup (older than 1 day)
            >>> deleted = storage.cleanup_old_files(days_old=1)

        Safety notes:
            - Only deletes *.json files in the storage directory
            - Uses modification time, so recently accessed files are preserved
            - Deletion is permanent - ensure you have backups if needed

        Suggested usage:
            Run periodically (e.g., daily cron job) to maintain disk space.
            Adjust days_old based on your retention requirements.
        """
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        for file_path in self.storage_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1

        return deleted_count

    async def cleanup(self) -> None:
        """Clean up resources (no-op for FileStorage).

        FileStorage doesn't maintain persistent connections or resources
        that need cleanup. This method exists for API compatibility with
        other storage backends.
        """
        # No cleanup needed for file storage
