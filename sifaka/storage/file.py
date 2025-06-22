"""File-based storage backend for SifakaResults."""

import json
import aiofiles
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .base import StorageBackend
from .mixins import SearchMixin
from ..core.models import SifakaResult
from ..core.exceptions import StorageError


class FileStorage(StorageBackend, SearchMixin):
    """File-based storage backend for persistent storage."""

    def __init__(self, storage_dir: str = "./sifaka_thoughts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def _get_file_path(self, result_id: str) -> Path:
        """Get file path for a result ID."""
        return self.storage_dir / f"{result_id}.json"

    async def save(self, result: SifakaResult) -> str:
        """Save a SifakaResult to disk."""
        file_path = self._get_file_path(result.id)

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
        file_path = self._get_file_path(result_id)

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                return SifakaResult.model_validate(data)
        except FileNotFoundError:
            return None  # Result doesn't exist
        except (OSError, IOError, PermissionError) as e:
            raise StorageError(
                f"Failed to load result {result_id}: {e}",
                storage_type="file",
                operation="load",
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise StorageError(
                f"Failed to parse result {result_id}: {e}",
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
        scored_results = []

        for file_path in self.storage_dir.glob("*.json"):
            try:
                # Load the result to search properly
                result = await self.load(file_path.stem)
                if result:
                    # Build searchable text using mixin
                    searchable_text = self._build_searchable_text(result)
                    
                    # Check if matches query
                    if self._text_matches_query(searchable_text, query):
                        # Calculate relevance score
                        score = self._calculate_relevance_score(result, query, searchable_text)
                        scored_results.append((file_path.stem, score))

            except Exception:
                continue

        # Rank and limit results
        return self._rank_search_results(scored_results, limit)

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
