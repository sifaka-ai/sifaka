"""Thought storage management for Sifaka.

This module contains the ThoughtStorage class which handles persistence
and retrieval of thoughts across different storage backends.
"""

from typing import Dict, List, Optional

from sifaka.core.thought.constants import DEFAULT_MAX_ITERATIONS
from sifaka.core.thought.thought import Thought
from sifaka.core.thought.utils import extract_chain_ids_from_keys, generate_thought_key
from sifaka.storage.protocol import Storage
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ThoughtStorage:
    """Manages persistence and retrieval of thoughts.

    This class provides a unified interface for storing and retrieving
    thoughts across different storage backends, with support for
    caching and persistence layers.
    """

    def __init__(self, storage: Optional[Storage] = None):
        """Initialize the thought storage.

        Args:
            storage: Optional storage backend. Defaults to MemoryStorage.
        """
        if storage is None:
            from sifaka.storage.memory import MemoryStorage

            self.storage: Storage = MemoryStorage()
        else:
            self.storage = storage

    def save_thought(self, thought: Thought) -> None:
        """Save a thought to storage.

        Args:
            thought: The thought to save.
        """
        try:
            thought_key = generate_thought_key(thought.chain_id, thought.iteration)
            self.storage.set(thought_key, thought.model_dump())
            logger.debug(f"Saved thought {thought.id} (iteration {thought.iteration}) to storage")
        except Exception as e:
            logger.warning(f"Failed to save thought {thought.id}: {e}")

    def load_thought(self, chain_id: str, iteration: int) -> Optional[Thought]:
        """Load a thought from storage.

        Args:
            chain_id: The chain ID.
            iteration: The iteration number.

        Returns:
            The loaded thought, or None if not found.
        """
        try:
            thought_key = generate_thought_key(chain_id, iteration)
            data = self.storage.get(thought_key)
            if data:
                # Handle case where storage returns a Thought object directly
                if isinstance(data, Thought):
                    return data
                # Handle case where storage returns a dictionary
                elif isinstance(data, dict):
                    return Thought.from_dict(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to load thought {chain_id}_{iteration}: {e}")
            return None

    def load_latest_thought(self, chain_id: str) -> Optional[Thought]:
        """Load the latest thought for a chain.

        Args:
            chain_id: The chain ID.

        Returns:
            The latest thought, or None if not found.
        """
        # Try to find the latest iteration (including iteration 0)
        for iteration in range(DEFAULT_MAX_ITERATIONS, -1, -1):
            thought = self.load_thought(chain_id, iteration)
            if thought:
                return thought
        return None

    def load_thought_history(self, chain_id: str) -> List[Thought]:
        """Load the complete thought history for a chain.

        Args:
            chain_id: The chain ID.

        Returns:
            List of thoughts in chronological order.
        """
        thoughts = []
        # Load all iterations we can find (including iteration 0)
        for iteration in range(0, DEFAULT_MAX_ITERATIONS + 1):
            thought = self.load_thought(chain_id, iteration)
            if thought:
                thoughts.append(thought)
            else:
                # For iteration 0, continue checking higher iterations
                # For higher iterations, stop when we don't find the next one
                if iteration > 0:
                    break

        logger.debug(f"Loaded {len(thoughts)} thoughts for chain {chain_id}")
        return thoughts

    def delete_thought(self, chain_id: str, iteration: int) -> bool:
        """Delete a thought from storage.

        Args:
            chain_id: The chain ID.
            iteration: The iteration number.

        Returns:
            True if deleted successfully, False otherwise.
        """
        try:
            thought_key = generate_thought_key(chain_id, iteration)
            self.storage.delete(thought_key)
            logger.debug(f"Deleted thought {thought_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete thought {chain_id}_{iteration}: {e}")
            return False

    def delete_chain_thoughts(self, chain_id: str) -> int:
        """Delete all thoughts for a chain.

        Args:
            chain_id: The chain ID.

        Returns:
            Number of thoughts deleted.
        """
        deleted_count = 0
        # Delete all iterations we can find (including iteration 0)
        for iteration in range(0, DEFAULT_MAX_ITERATIONS + 1):
            if self.delete_thought(chain_id, iteration):
                deleted_count += 1

        logger.debug(f"Deleted {deleted_count} thoughts for chain {chain_id}")
        return deleted_count

    def list_chains(self) -> List[str]:
        """List all chain IDs that have stored thoughts.

        Returns:
            List of chain IDs.
        """
        try:
            # This is a simplified implementation
            # In practice, you'd want to maintain an index of chains
            keys = self.storage.keys() if hasattr(self.storage, "keys") else []
            chain_ids = extract_chain_ids_from_keys(keys)
            return list(chain_ids)
        except Exception as e:
            logger.warning(f"Failed to list chains: {e}")
            return []

    def get_storage_stats(self) -> Dict[str, int]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        try:
            chains = self.list_chains()
            total_thoughts = 0

            for chain_id in chains:
                thoughts = self.load_thought_history(chain_id)
                total_thoughts += len(thoughts)

            return {
                "total_chains": len(chains),
                "total_thoughts": total_thoughts,
                "average_thoughts_per_chain": int(total_thoughts / len(chains)) if chains else 0,
            }
        except Exception as e:
            logger.warning(f"Failed to get storage stats: {e}")
            return {"total_chains": 0, "total_thoughts": 0, "average_thoughts_per_chain": 0}

    def cleanup_old_thoughts(self, max_age_days: int = 30) -> int:
        """Clean up old thoughts based on age.

        Args:
            max_age_days: Maximum age in days for thoughts to keep.

        Returns:
            Number of thoughts cleaned up.
        """
        # This would require timestamp-based cleanup
        # For now, just return 0 as a placeholder
        logger.info(f"Cleanup of thoughts older than {max_age_days} days not implemented yet")
        return 0
