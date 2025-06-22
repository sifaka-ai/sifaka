"""Base storage interface for SifakaResult persistence."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from ..core.models import SifakaResult


class StorageBackend(ABC):
    """Base interface for thought storage backends."""

    @abstractmethod
    async def save(self, result: SifakaResult) -> str:
        """Save a SifakaResult and return its ID.

        Args:
            result: The SifakaResult to save

        Returns:
            Storage ID for the saved result
        """
        pass

    @abstractmethod
    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a SifakaResult by ID.

        Args:
            result_id: The ID of the result to load

        Returns:
            The loaded SifakaResult or None if not found
        """
        pass

    @abstractmethod
    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List stored result IDs.

        Args:
            limit: Maximum number of IDs to return
            offset: Number of IDs to skip

        Returns:
            List of result IDs
        """
        pass

    @abstractmethod
    async def delete(self, result_id: str) -> bool:
        """Delete a stored result.

        Args:
            result_id: The ID of the result to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search stored results by text content.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching result IDs
        """
        pass

    async def get_metadata(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a stored result.

        Args:
            result_id: The ID of the result

        Returns:
            Metadata dict or None if not found
        """
        result = await self.load(result_id)
        if not result:
            return None

        return {
            "id": result.id,
            "original_text": (
                result.original_text[:100] + "..."
                if len(result.original_text) > 100
                else result.original_text
            ),
            "final_text": (
                result.final_text[:100] + "..."
                if len(result.final_text) > 100
                else result.final_text
            ),
            "iteration": result.iteration,
            "processing_time": result.processing_time,
            "created_at": result.created_at,
            "updated_at": result.updated_at,
            "num_generations": len(result.generations),
            "num_validations": len(result.validations),
            "num_critiques": len(result.critiques),
        }
