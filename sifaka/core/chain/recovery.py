"""Chain recovery and checkpointing for Sifaka.

This module contains the RecoveryManager class which handles checkpointing
and recovery functionality for chain execution.
"""

from typing import List, Optional

from sifaka.core.chain.config import ChainConfig
from sifaka.core.thought import Thought
from sifaka.storage.checkpoints import ChainCheckpoint
from sifaka.recovery.manager import RecoveryAction, RecoveryStrategy
from sifaka.utils.performance import PerformanceMonitor
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class RecoveryManager:
    """Manages checkpointing and recovery for chain execution.

    This class handles saving execution state at key points and
    recovering from failures by resuming from the last successful checkpoint.
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
            # Get performance data
            monitor = PerformanceMonitor.get_instance()
            performance_data = monitor.get_summary()

            # Create checkpoint
            checkpoint = ChainCheckpoint(
                chain_id=self.config.chain_id,
                current_step=step,
                iteration=iteration,
                thought=thought,
                performance_data=performance_data,
                recovery_point=step,
                completed_validators=[v.__class__.__name__ for v in self.config.validators],
                completed_critics=[c.__class__.__name__ for c in self.config.critics],
                metadata={
                    "model_name": getattr(self.config.model, "model_name", "unknown"),
                    "prompt_length": len(self.config.prompt or ""),
                    "model_retriever_count": len(self.config.model_retrievers),
                    "critic_retriever_count": len(self.config.critic_retrievers),
                },
            )

            self.config.checkpoint_storage.save_checkpoint(checkpoint)
            self.current_checkpoint = checkpoint

            logger.debug(f"Saved checkpoint for step '{step}' at iteration {iteration}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint for step '{step}': {e}")

    def get_existing_checkpoints(self) -> List[ChainCheckpoint]:
        """Get existing checkpoints for the current chain.

        Returns:
            List of existing checkpoints, or empty list if none found.
        """
        if not self.config.checkpoint_storage:
            return []

        try:
            return self.config.checkpoint_storage.get_chain_checkpoints(self.config.chain_id)
        except Exception as e:
            logger.warning(f"Failed to retrieve existing checkpoints: {e}")
            return []

    def get_latest_checkpoint(self) -> Optional[ChainCheckpoint]:
        """Get the latest checkpoint for the current chain.

        Returns:
            The latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.get_existing_checkpoints()
        if not checkpoints:
            return None

        # Find the latest incomplete checkpoint
        for checkpoint in reversed(checkpoints):
            if checkpoint.current_step != "complete":
                return checkpoint

        return None

    def can_resume(self) -> bool:
        """Check if the chain can be resumed from a checkpoint.

        Returns:
            True if resumption is possible, False otherwise.
        """
        if not self.config.checkpoint_storage:
            return False

        latest = self.get_latest_checkpoint()
        return latest is not None

    def analyze_failure(
        self, checkpoint: ChainCheckpoint, error: Exception
    ) -> List[RecoveryAction]:
        """Analyze a failure and suggest recovery actions.

        Args:
            checkpoint: The checkpoint where failure occurred.
            error: The exception that caused the failure.

        Returns:
            List of suggested recovery actions.
        """
        recovery_actions = []

        # Analyze the type of error and suggest appropriate actions
        error_type = type(error).__name__
        error_message = str(error)

        if "timeout" in error_message.lower() or "connection" in error_message.lower():
            recovery_actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY_CURRENT_STEP,
                    description="Retry with exponential backoff due to connection/timeout error",
                    confidence=0.7,
                    parameters={"max_retries": 3, "base_delay": 2},
                )
            )

        if "rate limit" in error_message.lower() or "quota" in error_message.lower():
            recovery_actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY_CURRENT_STEP,
                    description="Wait longer before retry due to rate limiting",
                    confidence=0.6,
                    parameters={"delay": 60, "max_retries": 2},
                )
            )

        if "validation" in error_message.lower():
            recovery_actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.SKIP_TO_NEXT_STEP,
                    description="Skip problematic validation and continue",
                    confidence=0.5,
                    parameters={"skip_validator": True},
                )
            )

        # Default recovery action
        if not recovery_actions:
            recovery_actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY_CURRENT_STEP,
                    description="Simple retry of the failed operation",
                    confidence=0.4,
                    parameters={"max_retries": 1},
                )
            )

        return recovery_actions

    def apply_recovery_action(self, action: RecoveryAction) -> bool:
        """Apply a recovery action to the chain configuration.

        Args:
            action: The recovery action to apply.

        Returns:
            True if the action was applied successfully, False otherwise.
        """
        try:
            if action.strategy == RecoveryStrategy.RETRY_CURRENT_STEP:
                # This would be handled by the calling code
                logger.info(f"Applying recovery action: {action.description}")
                return True

            elif action.strategy == RecoveryStrategy.SKIP_TO_NEXT_STEP:
                # Temporarily disable validators
                logger.info("Temporarily disabling validators for recovery")
                self.config.validators = []
                return True

            elif action.strategy == RecoveryStrategy.FULL_RESTART:
                logger.info("Performing full restart")
                return True

            else:
                logger.warning(f"Unknown recovery strategy: {action.strategy}")
                return False

        except Exception as e:
            logger.error(f"Failed to apply recovery action {action.strategy}: {e}")
            return False

    def cleanup_checkpoints(self, keep_latest: int = 5) -> None:
        """Clean up old checkpoints, keeping only the most recent ones.

        Args:
            keep_latest: Number of latest checkpoints to keep.
        """
        if not self.config.checkpoint_storage:
            return

        try:
            checkpoints = self.get_existing_checkpoints()
            if len(checkpoints) <= keep_latest:
                return

            # Remove older checkpoints
            to_remove = checkpoints[:-keep_latest]
            for checkpoint in to_remove:
                self.config.checkpoint_storage.delete_checkpoint(checkpoint.checkpoint_id)

            logger.debug(f"Cleaned up {len(to_remove)} old checkpoints")

        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoints: {e}")
