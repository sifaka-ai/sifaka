"""Thought storage management for PydanticAI chains.

This module handles the persistence of thoughts during chain execution,
including intermediate iterations and final results.
"""

from sifaka.core.thought import Thought
from sifaka.storage.protocol import Storage
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ThoughtStorage:
    """Manages thought persistence during chain execution."""

    def __init__(self, storage: Storage):
        """Initialize the thought storage manager.

        Args:
            storage: The storage backend to use.
        """
        self.storage = storage

    async def save_intermediate_thought(self, thought: Thought) -> None:
        """Save an intermediate thought iteration.

        Args:
            thought: The thought to save.
        """
        try:
            # Use thought.id as key to match original behavior
            logger.debug(f"Saving intermediate thought with key: {thought.id}")

            # Check if storage has async method, otherwise fall back to sync
            if hasattr(self.storage, "_set_async"):
                await self.storage._set_async(thought.id, thought)
            else:
                self.storage.set(thought.id, thought)

            logger.debug(f"Successfully saved intermediate thought: iteration {thought.iteration}")
        except Exception as e:
            logger.error(f"Failed to save intermediate thought: {e}")
            # Don't raise the exception to avoid breaking the chain execution

    async def save_final_thought(self, thought: Thought) -> None:
        """Save the final thought result.

        Args:
            thought: The final thought to save.
        """
        try:
            # Use thought.id as key to match original behavior
            logger.debug(f"Saving final thought with key: {thought.id}")

            # Check if storage has async method, otherwise fall back to sync
            if hasattr(self.storage, "_set_async"):
                await self.storage._set_async(thought.id, thought)
            else:
                self.storage.set(thought.id, thought)

            logger.info(
                f"Successfully saved thought to: {getattr(self.storage, 'file_path', 'storage')}"
            )
        except Exception as e:
            logger.error(f"Failed to save final thought: {e}")
            # Don't raise the exception to avoid breaking the chain execution

    async def load_thought(self, thought_id: str) -> Thought:
        """Load a thought by ID.

        Args:
            thought_id: The thought ID.

        Returns:
            The loaded thought.

        Raises:
            KeyError: If the thought is not found.
        """
        logger.debug(f"Loading thought with key: {thought_id}")

        # Check if storage has async method, otherwise fall back to sync
        if hasattr(self.storage, "_get_async"):
            result = await self.storage._get_async(thought_id)
        else:
            result = self.storage.get(thought_id)

        if result is None:
            raise KeyError(f"Thought not found: {thought_id}")

        return result
