"""Checkpoint storage for chain recovery using unified 3-tier architecture.

This module provides checkpoint storage capabilities for chain execution
recovery, including vector similarity search for finding similar execution
patterns and recovery strategies.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from sifaka.core.thought import Thought
from sifaka.utils.logging import get_logger

from .base import CachedStorage, StorageError

logger = get_logger(__name__)


class ChainCheckpoint(BaseModel):
    """Chain execution checkpoint for recovery.

    Represents a snapshot of chain execution state that can be used
    to resume execution after interruption or failure.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        chain_id: ID of the chain this checkpoint belongs to.
        timestamp: When this checkpoint was created.
        current_step: Current execution step.
        iteration: Current iteration number.
        completed_validators: List of validators that have completed.
        completed_critics: List of critics that have completed.
        thought: Current thought state.
        performance_data: Performance metrics at checkpoint time.
        recovery_point: Where to resume execution from.
        metadata: Additional checkpoint metadata.
    """

    # Identity
    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    chain_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Execution state
    current_step: str  # "pre_retrieval", "generation", "validation", "criticism", "complete"
    iteration: int
    completed_validators: List[str] = Field(default_factory=list)
    completed_critics: List[str] = Field(default_factory=list)

    # Chain state
    thought: Thought

    # Performance and recovery data
    performance_data: Dict[str, Any] = Field(default_factory=dict)
    recovery_point: str  # Where to resume from
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_search_text(self) -> str:
        """Get text representation for vector similarity search."""
        return f"""
        Step: {self.current_step}
        Iteration: {self.iteration}
        Prompt: {self.thought.prompt}
        Text: {self.thought.text or ''}
        Validators: {', '.join(self.completed_validators)}
        Critics: {', '.join(self.completed_critics)}
        Recovery Point: {self.recovery_point}
        """.strip()


class CachedCheckpointStorage:
    """Storage for chain execution checkpoints.

    Provides checkpoint storage with vector similarity search for finding
    similar execution patterns and recovery strategies.

    Attributes:
        storage: Underlying CachedStorage instance.
    """

    def __init__(self, storage: CachedStorage):
        """Initialize cached checkpoint storage.

        Args:
            storage: CachedStorage instance for 3-tier storage.
        """
        self.storage = storage
        logger.debug("Initialized CachedCheckpointStorage")

    def save_checkpoint(self, checkpoint: ChainCheckpoint) -> None:
        """Save a checkpoint to storage.

        Args:
            checkpoint: The checkpoint to save.

        Raises:
            StorageError: If the save operation fails.
        """
        try:
            key = f"checkpoint:{checkpoint.checkpoint_id}"

            # Prepare metadata for vector storage
            metadata = {
                "chain_id": [checkpoint.chain_id],
                "current_step": [checkpoint.current_step],
                "iteration": [checkpoint.iteration],
                "timestamp": [checkpoint.timestamp.isoformat()],
                "recovery_point": [checkpoint.recovery_point],
                "validator_count": [len(checkpoint.completed_validators)],
                "critic_count": [len(checkpoint.completed_critics)],
            }

            # Save to all tiers
            self.storage.set(key, checkpoint, metadata)

            logger.debug(
                f"Saved checkpoint {checkpoint.checkpoint_id} for chain {checkpoint.chain_id} "
                f"at step {checkpoint.current_step}"
            )

        except Exception as e:
            raise StorageError(
                f"Failed to save checkpoint {checkpoint.checkpoint_id}",
                operation="save_checkpoint",
                storage_type="CachedCheckpointStorage",
                metadata={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "chain_id": checkpoint.chain_id,
                    "step": checkpoint.current_step,
                },
            ) from e

    def get_checkpoint(self, checkpoint_id: str) -> Optional[ChainCheckpoint]:
        """Retrieve a checkpoint by ID.

        Args:
            checkpoint_id: The ID of the checkpoint to retrieve.

        Returns:
            The checkpoint if found, None otherwise.

        Raises:
            StorageError: If the retrieval operation fails.
        """
        try:
            key = f"checkpoint:{checkpoint_id}"
            checkpoint = self.storage.get(key)

            if checkpoint:
                logger.debug(f"Retrieved checkpoint {checkpoint_id}")
            else:
                logger.debug(f"Checkpoint {checkpoint_id} not found")

            return checkpoint

        except Exception as e:
            raise StorageError(
                f"Failed to retrieve checkpoint {checkpoint_id}",
                operation="get_checkpoint",
                storage_type="CachedCheckpointStorage",
                metadata={"checkpoint_id": checkpoint_id},
            ) from e

    def get_latest_checkpoint(self, chain_id: str) -> Optional[ChainCheckpoint]:
        """Get the most recent checkpoint for a chain.

        Args:
            chain_id: The chain ID to get the latest checkpoint for.

        Returns:
            The most recent checkpoint for the chain, or None if not found.
        """
        try:
            # Get all checkpoints for this chain from memory first
            chain_checkpoints = []
            for key, value in self.storage.memory.data.items():
                if (
                    key.startswith("checkpoint:")
                    and isinstance(value, ChainCheckpoint)
                    and value.chain_id == chain_id
                ):
                    chain_checkpoints.append(value)

            if not chain_checkpoints:
                # If not in memory, search in persistence layer
                query_text = f"chain_id:{chain_id}"
                similar_checkpoints = self.storage.search_similar(query_text, limit=20)
                chain_checkpoints = [
                    cp
                    for cp in similar_checkpoints
                    if isinstance(cp, ChainCheckpoint) and cp.chain_id == chain_id
                ]

            if not chain_checkpoints:
                return None

            # Return the most recent checkpoint
            latest = max(chain_checkpoints, key=lambda cp: cp.timestamp)
            logger.debug(f"Found latest checkpoint {latest.checkpoint_id} for chain {chain_id}")
            return latest

        except Exception as e:
            logger.warning(f"Failed to get latest checkpoint for chain {chain_id}: {e}")
            return None

    def find_similar_checkpoints(
        self, checkpoint: ChainCheckpoint, limit: int = 5
    ) -> List[ChainCheckpoint]:
        """Find checkpoints similar to the given checkpoint.

        Uses vector similarity search to find checkpoints with similar
        execution patterns, which can be useful for recovery strategies.

        Args:
            checkpoint: The checkpoint to find similar checkpoints for.
            limit: Maximum number of similar checkpoints to return.

        Returns:
            List of similar checkpoints.
        """
        try:
            query_text = checkpoint.get_search_text()
            similar_items = self.storage.search_similar(query_text, limit)

            # Filter to only return ChainCheckpoint objects
            checkpoints = [
                item
                for item in similar_items
                if isinstance(item, ChainCheckpoint)
                and item.checkpoint_id != checkpoint.checkpoint_id
            ]

            logger.debug(
                f"Found {len(checkpoints)} similar checkpoints for {checkpoint.checkpoint_id} "
                f"at step {checkpoint.current_step}"
            )
            return checkpoints

        except Exception as e:
            logger.warning(f"Similar checkpoints search failed: {e}")
            return []

    def get_chain_checkpoints(self, chain_id: str) -> List[ChainCheckpoint]:
        """Get all checkpoints for a specific chain.

        Args:
            chain_id: The chain ID to get checkpoints for.

        Returns:
            List of checkpoints for the chain, sorted by timestamp.
        """
        try:
            # Search for checkpoints with this chain_id
            query_text = f"chain_id:{chain_id}"
            similar_items = self.storage.search_similar(query_text, limit=100)

            # Filter and sort by chain_id and timestamp
            chain_checkpoints = [
                item
                for item in similar_items
                if isinstance(item, ChainCheckpoint) and item.chain_id == chain_id
            ]
            chain_checkpoints.sort(key=lambda cp: cp.timestamp)

            logger.debug(f"Found {len(chain_checkpoints)} checkpoints for chain {chain_id}")
            return chain_checkpoints

        except Exception as e:
            logger.warning(f"Failed to get checkpoints for chain {chain_id}: {e}")
            return []

    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """Clean up old checkpoints to save storage space.

        Args:
            max_age_days: Maximum age of checkpoints to keep.

        Returns:
            Number of checkpoints cleaned up.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0

            # Clean up from memory storage
            keys_to_remove = []
            for key, value in self.storage.memory.data.items():
                if (
                    key.startswith("checkpoint:")
                    and isinstance(value, ChainCheckpoint)
                    and value.timestamp < cutoff_date
                ):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.storage.memory.data[key]
                cleaned_count += 1

            logger.debug(
                f"Cleaned up {cleaned_count} old checkpoints (older than {max_age_days} days)"
            )
            return cleaned_count

        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed: {e}")
            return 0

    def get_recovery_suggestions(self, failed_checkpoint: ChainCheckpoint) -> List[Dict[str, Any]]:
        """Get recovery suggestions based on similar checkpoints.

        Args:
            failed_checkpoint: The checkpoint that failed.

        Returns:
            List of recovery suggestions with strategies and confidence scores.
        """
        try:
            similar_checkpoints = self.find_similar_checkpoints(failed_checkpoint, limit=10)

            suggestions = []
            for similar_cp in similar_checkpoints:
                # Analyze what made this checkpoint successful
                suggestion = {
                    "checkpoint_id": similar_cp.checkpoint_id,
                    "similarity_score": 0.8,  # Would be calculated based on vector similarity
                    "recovery_strategy": self._analyze_recovery_strategy(
                        similar_cp, failed_checkpoint
                    ),
                    "success_factors": self._identify_success_factors(similar_cp),
                    "recommended_actions": self._generate_recovery_actions(
                        similar_cp, failed_checkpoint
                    ),
                }
                suggestions.append(suggestion)

            # Sort by similarity score
            suggestions.sort(key=lambda s: s["similarity_score"], reverse=True)  # type: ignore

            logger.debug(f"Generated {len(suggestions)} recovery suggestions")
            return suggestions[:5]  # Return top 5 suggestions

        except Exception as e:
            logger.warning(f"Failed to generate recovery suggestions: {e}")
            return []

    def _analyze_recovery_strategy(
        self, successful_cp: ChainCheckpoint, failed_cp: ChainCheckpoint
    ) -> str:
        """Analyze what recovery strategy might work based on successful checkpoint."""
        if successful_cp.current_step != failed_cp.current_step:
            return (
                f"Resume from {successful_cp.recovery_point} instead of {failed_cp.recovery_point}"
            )
        elif len(successful_cp.completed_validators) != len(failed_cp.completed_validators):
            return "Adjust validator configuration based on successful pattern"
        elif len(successful_cp.completed_critics) != len(failed_cp.completed_critics):
            return "Modify critic application strategy"
        else:
            return "Retry with similar configuration"

    def _identify_success_factors(self, checkpoint: ChainCheckpoint) -> List[str]:
        """Identify factors that contributed to checkpoint success."""
        factors = []

        if checkpoint.current_step == "complete":
            factors.append("Successfully completed all steps")

        if len(checkpoint.completed_validators) > 0:
            factors.append(f"Passed {len(checkpoint.completed_validators)} validators")

        if len(checkpoint.completed_critics) > 0:
            factors.append(f"Applied {len(checkpoint.completed_critics)} critics")

        if checkpoint.iteration > 1:
            factors.append(f"Successful iteration {checkpoint.iteration}")

        return factors

    def _generate_recovery_actions(
        self, successful_cp: ChainCheckpoint, failed_cp: ChainCheckpoint
    ) -> List[str]:
        """Generate specific recovery actions based on comparison."""
        actions = []

        if successful_cp.recovery_point != failed_cp.recovery_point:
            actions.append(f"Change recovery point to '{successful_cp.recovery_point}'")

        missing_validators = set(successful_cp.completed_validators) - set(
            failed_cp.completed_validators
        )
        if missing_validators:
            actions.append(f"Enable validators: {', '.join(missing_validators)}")

        missing_critics = set(successful_cp.completed_critics) - set(failed_cp.completed_critics)
        if missing_critics:
            actions.append(f"Apply critics: {', '.join(missing_critics)}")

        if not actions:
            actions.append("Retry with current configuration")

        return actions

    def clear(self) -> None:
        """Clear all checkpoint storage."""
        self.storage.clear()
        logger.debug("Cleared all checkpoint storage")

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint storage statistics."""
        base_stats = self.storage.get_stats()

        # Add checkpoint-specific stats
        checkpoint_count = sum(
            1 for key in self.storage.memory.data.keys() if key.startswith("checkpoint:")
        )

        return {
            **base_stats,
            "checkpoint_count_in_memory": checkpoint_count,
            "storage_type": "CachedCheckpointStorage",
        }
