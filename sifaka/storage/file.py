"""File-based storage backend for SifakaResults."""

import json
import aiofiles
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .base import StorageBackend
from ..core.models import SifakaResult
from ..core.exceptions import StorageError


class FileStorage(StorageBackend):
    """File-based storage backend for persistent storage."""

    def __init__(self, storage_dir: str = "./thoughts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def _get_file_path(
        self, result_id: str, critics: Optional[List[str]] = None
    ) -> Path:
        """Get file path for a result ID."""
        if critics:
            critics_str = "_".join(critics)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self.storage_dir / f"{critics_str}_{timestamp}_{result_id[:8]}.json"
        return self.storage_dir / f"{result_id}.json"

    async def save(self, result: SifakaResult) -> str:
        """Save a SifakaResult to disk."""
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
        except (OSError, IOError, PermissionError) as e:
            raise StorageError(
                f"Failed to save result {result.id}: {e}",
                storage_type="file",
                operation="save",
            )

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a SifakaResult from disk."""
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
        """Load from a specific file path."""

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                return SifakaResult.model_validate(data)
        except FileNotFoundError:
            return None  # Result doesn't exist
        except (OSError, IOError, PermissionError) as e:
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
        """List stored result IDs."""
        json_files = list(self.storage_dir.glob("*.json"))

        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Extract IDs from filenames
        ids = [f.stem for f in json_files[offset : offset + limit]]
        return ids

    async def delete(self, result_id: str) -> bool:
        """Delete a stored result from disk."""
        file_path = self._get_file_path(result_id)

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search results by text content."""
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
                        matches.append(file_path.stem)

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

        Args:
            days_old: Delete files older than this many days

        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        for file_path in self.storage_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1

        return deleted_count
