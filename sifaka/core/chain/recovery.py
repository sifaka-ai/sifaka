"""Simplified chain recovery and checkpointing for Sifaka.

This module contains the RecoveryManager class which handles basic checkpointing
functionality for chain execution.
"""

from typing import Optional

from sifaka.core.chain.config import ChainConfig
from sifaka.core.thought import Thought
from sifaka.storage.checkpoints import ChainCheckpoint
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class RecoveryManager:
    """Simplified recovery manager for chain execution checkpointing.

    This class handles saving execution state at key points for basic
    recovery functionality.
    """

    def __init__(self, config: ChainConfig):
        """Initialize the recovery manager with chain configuration.

        Args:
            config: The chain configuration.
        """
        self.config = config
        self.current_checkpoint: Optional[ChainCheckpoint] = None

    def save_checkpoint(self, step: str, thought: Thought, iteration: int) -> None:
        """Save a checkpoint for the current execution state.

        Args:
            step: The current execution step.
            thought: The current thought state.
            iteration: The current iteration number.
        """
        if not self.config.checkpoint_storage:
            return

        try:
            # Create simplified checkpoint
            checkpoint = ChainCheckpoint(
                chain_id=self.config.chain_id,
                current_step=step,
                iteration=iteration,
                thought=thought,
                performance_data={},  # Simplified - no performance monitoring
                recovery_point=step,
                completed_validators=[v.__class__.__name__ for v in self.config.validators],
                completed_critics=[c.__class__.__name__ for c in self.config.critics],
                metadata={
                    "step": step,
                    "iteration": iteration,
                },
            )

            self.config.checkpoint_storage.save_checkpoint(checkpoint)
            self.current_checkpoint = checkpoint

            logger.debug(f"Saved checkpoint for step '{step}' at iteration {iteration}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint for step '{step}': {e}")

    def get_latest_checkpoint(self) -> Optional[ChainCheckpoint]:
        """Get the latest checkpoint for the current chain.

        Returns:
            The latest checkpoint, or None if no checkpoints exist.
        """
        if not self.config.checkpoint_storage:
            return None

        try:
            return self.config.checkpoint_storage.get_latest_checkpoint(self.config.chain_id)
        except Exception as e:
            logger.warning(f"Failed to get latest checkpoint: {e}")
            return None

    def can_resume(self) -> bool:
        """Check if the chain can be resumed from a checkpoint.

        Returns:
            True if resumption is possible, False otherwise.
        """
        if not self.config.checkpoint_storage:
            return False

        latest = self.get_latest_checkpoint()
        return latest is not None and latest.current_step != "complete"
