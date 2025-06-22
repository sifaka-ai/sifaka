"""In-memory storage backend for SifakaResults."""

from typing import Optional, List, Dict

from .base import StorageBackend
from .mixins import SearchMixin
from ..core.models import SifakaResult
from ..core.exceptions import StorageError


class MemoryStorage(StorageBackend, SearchMixin):
    """In-memory storage backend (default, non-persistent)."""

    def __init__(self) -> None:
        self._storage: Dict[str, SifakaResult] = {}

    async def save(self, result: SifakaResult) -> str:
        """Save a SifakaResult in memory."""
        try:
            self._storage[result.id] = result
            return result.id
        except Exception as e:
            raise StorageError(
                f"Failed to save result {result.id}",
                storage_type="memory",
                operation="save",
            ) from e

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a SifakaResult from memory."""
        try:
            return self._storage.get(result_id)
        except Exception as e:
            raise StorageError(
                f"Failed to load result {result_id}",
                storage_type="memory",
                operation="load",
            ) from e

    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List stored result IDs."""
        ids = list(self._storage.keys())
        return ids[offset : offset + limit]

    async def delete(self, result_id: str) -> bool:
        """Delete a stored result from memory."""
        if result_id in self._storage:
            del self._storage[result_id]
            return True
        return False

    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search results by text content."""
        try:
            scored_results = []

            for result_id, result in self._storage.items():
                # Build searchable text using mixin
                searchable_text = self._build_searchable_text(result)
                
                # Check if matches query
                if self._text_matches_query(searchable_text, query):
                    # Calculate relevance score
                    score = self._calculate_relevance_score(result, query, searchable_text)
                    scored_results.append((result_id, score))

            # Rank and limit results
            return self._rank_search_results(scored_results, limit)
        except Exception as e:
            raise StorageError(
                f"Search failed for query: {query}",
                storage_type="memory",
                operation="search",
            ) from e

    def clear(self) -> None:
        """Clear all stored results."""
        self._storage.clear()

    def size(self) -> int:
        """Get number of stored results."""
        return len(self._storage)
