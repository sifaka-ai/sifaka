"""Recovery management for Sifaka chains.

This module provides intelligent recovery strategies for chain execution failures,
including pattern analysis from similar past executions and automatic recovery
suggestions.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from sifaka.core.thought import Thought
from sifaka.storage.checkpoints import ChainCheckpoint, CachedCheckpointStorage
from sifaka.utils.error_handling import SifakaError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RETRY_CURRENT_STEP = "retry_current_step"
    SKIP_TO_NEXT_STEP = "skip_to_next_step"
    RESTART_ITERATION = "restart_iteration"
    RESTART_FROM_VALIDATION = "restart_from_validation"
    RESTART_FROM_GENERATION = "restart_from_generation"
    FULL_RESTART = "full_restart"
    MODIFY_PARAMETERS = "modify_parameters"


class RecoveryAction:
    """Represents a specific recovery action to take."""

    def __init__(
        self,
        strategy: RecoveryStrategy,
        description: str,
        confidence: float,
        parameters: Optional[Dict[str, Any]] = None,
        estimated_success_rate: Optional[float] = None,
    ):
        """Initialize recovery action.

        Args:
            strategy: The recovery strategy to use
            description: Human-readable description of the action
            confidence: Confidence in this action (0.0 to 1.0)
            parameters: Optional parameters for the action
            estimated_success_rate: Estimated success rate based on historical data
        """
        self.strategy = strategy
        self.description = description
        self.confidence = confidence
        self.parameters = parameters or {}
        self.estimated_success_rate = estimated_success_rate


class RecoveryManager:
    """Manages chain recovery and error handling.

    This class analyzes failed chain executions and suggests recovery strategies
    based on historical patterns and similar execution contexts.
    """

    def __init__(self, checkpoint_storage: CachedCheckpointStorage):
        """Initialize the recovery manager.

        Args:
            checkpoint_storage: Storage for chain checkpoints
        """
        self.checkpoint_storage = checkpoint_storage
        logger.debug("Initialized RecoveryManager")

    def analyze_failure(
        self, failed_checkpoint: ChainCheckpoint, error: Exception
    ) -> List[RecoveryAction]:
        """Analyze a failure and suggest recovery actions.

        Args:
            failed_checkpoint: The checkpoint where failure occurred
            error: The exception that caused the failure

        Returns:
            List of recovery actions, ordered by confidence
        """
        logger.info(
            f"Analyzing failure at step '{failed_checkpoint.current_step}' "
            f"for chain {failed_checkpoint.chain_id}"
        )

        # Get similar checkpoints for pattern analysis
        similar_checkpoints = self._find_similar_checkpoints(failed_checkpoint)

        # Analyze the specific error type
        error_patterns = self._analyze_error_patterns(error, similar_checkpoints)

        # Generate recovery actions
        actions = []

        # Strategy 1: Retry current step (always available)
        actions.append(
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY_CURRENT_STEP,
                description=f"Retry the current step: {failed_checkpoint.current_step}",
                confidence=0.6,
                estimated_success_rate=self._estimate_retry_success_rate(similar_checkpoints),
            )
        )

        # Strategy 2: Skip to next step (if applicable)
        if self._can_skip_step(failed_checkpoint):
            actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.SKIP_TO_NEXT_STEP,
                    description=f"Skip the failed step and continue to next",
                    confidence=0.4,
                    estimated_success_rate=0.3,
                )
            )

        # Strategy 3: Restart iteration (if in critic loop)
        if failed_checkpoint.iteration > 1:
            actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.RESTART_ITERATION,
                    description=f"Restart from iteration {failed_checkpoint.iteration - 1}",
                    confidence=0.7,
                    estimated_success_rate=self._estimate_restart_success_rate(similar_checkpoints),
                )
            )

        # Strategy 4: Restart from specific steps
        if failed_checkpoint.current_step in ["criticism", "validation"]:
            actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.RESTART_FROM_GENERATION,
                    description="Restart from text generation step",
                    confidence=0.5,
                    estimated_success_rate=0.6,
                )
            )

        # Strategy 5: Parameter modification (based on error patterns)
        param_modifications = self._suggest_parameter_modifications(error, error_patterns)
        if param_modifications:
            actions.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.MODIFY_PARAMETERS,
                    description=f"Modify parameters: {', '.join(param_modifications.keys())}",
                    confidence=0.8,
                    parameters=param_modifications,
                    estimated_success_rate=0.7,
                )
            )

        # Strategy 6: Full restart (last resort)
        actions.append(
            RecoveryAction(
                strategy=RecoveryStrategy.FULL_RESTART,
                description="Restart the entire chain execution",
                confidence=0.3,
                estimated_success_rate=0.9,
            )
        )

        # Sort by confidence (highest first)
        actions.sort(key=lambda a: a.confidence, reverse=True)

        logger.info(f"Generated {len(actions)} recovery actions for failure analysis")
        return actions

    def _find_similar_checkpoints(
        self, checkpoint: ChainCheckpoint, limit: int = 20
    ) -> List[ChainCheckpoint]:
        """Find checkpoints similar to the given one.

        Args:
            checkpoint: The checkpoint to find similar ones for
            limit: Maximum number of similar checkpoints to return

        Returns:
            List of similar checkpoints
        """
        # Create a search query based on checkpoint characteristics
        query_parts = [
            f"step:{checkpoint.current_step}",
            f"iteration:{checkpoint.iteration}",
        ]

        # Add thought content for semantic similarity
        if checkpoint.thought.text:
            query_parts.append(checkpoint.thought.text[:200])

        query = " ".join(query_parts)

        try:
            similar_items = self.checkpoint_storage.storage.search(query, limit=limit)
            similar_checkpoints = [
                item
                for item in similar_items
                if isinstance(item, ChainCheckpoint)
                and item.checkpoint_id != checkpoint.checkpoint_id
            ]

            logger.debug(f"Found {len(similar_checkpoints)} similar checkpoints")
            return similar_checkpoints

        except Exception as e:
            logger.warning(f"Failed to find similar checkpoints: {e}")
            return []

    def _analyze_error_patterns(
        self, error: Exception, similar_checkpoints: List[ChainCheckpoint]
    ) -> Dict[str, Any]:
        """Analyze error patterns from similar checkpoints.

        Args:
            error: The current error
            similar_checkpoints: Similar checkpoints to analyze

        Returns:
            Dictionary of error patterns and insights
        """
        patterns: Dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "common_failures": [],
            "success_patterns": [],
        }

        # Analyze metadata from similar checkpoints for patterns
        for cp in similar_checkpoints:
            if "error" in cp.metadata:
                patterns["common_failures"].append(cp.metadata["error"])
            elif cp.current_step == "complete":
                patterns["success_patterns"].append(
                    {"step": cp.current_step, "iteration": cp.iteration, "metadata": cp.metadata}
                )

        return patterns

    def _can_skip_step(self, checkpoint: ChainCheckpoint) -> bool:
        """Determine if the current step can be skipped.

        Args:
            checkpoint: The checkpoint to check

        Returns:
            True if the step can be skipped
        """
        # Generally, validation and criticism steps can be skipped
        # but generation steps cannot
        skippable_steps = ["validation", "criticism", "post_retrieval"]
        return checkpoint.current_step in skippable_steps

    def _estimate_retry_success_rate(self, similar_checkpoints: List[ChainCheckpoint]) -> float:
        """Estimate success rate for retrying the current step.

        Args:
            similar_checkpoints: Similar checkpoints to analyze

        Returns:
            Estimated success rate (0.0 to 1.0)
        """
        if not similar_checkpoints:
            return 0.5  # Default estimate

        # Count successful retries vs failures
        successful_retries = sum(
            1
            for cp in similar_checkpoints
            if cp.metadata.get("retry_count", 0) > 0 and cp.current_step == "complete"
        )

        total_retries = sum(
            1 for cp in similar_checkpoints if cp.metadata.get("retry_count", 0) > 0
        )

        if total_retries == 0:
            return 0.5

        return successful_retries / total_retries

    def _estimate_restart_success_rate(self, similar_checkpoints: List[ChainCheckpoint]) -> float:
        """Estimate success rate for restarting from previous iteration.

        Args:
            similar_checkpoints: Similar checkpoints to analyze

        Returns:
            Estimated success rate (0.0 to 1.0)
        """
        if not similar_checkpoints:
            return 0.6  # Default estimate

        # Count successful restarts
        successful_restarts = sum(
            1
            for cp in similar_checkpoints
            if cp.metadata.get("restart_count", 0) > 0 and cp.current_step == "complete"
        )

        total_restarts = sum(
            1 for cp in similar_checkpoints if cp.metadata.get("restart_count", 0) > 0
        )

        if total_restarts == 0:
            return 0.6

        return successful_restarts / total_restarts

    def _suggest_parameter_modifications(
        self, error: Exception, error_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest parameter modifications based on error analysis.

        Args:
            error: The current error
            error_patterns: Analyzed error patterns

        Returns:
            Dictionary of suggested parameter modifications
        """
        modifications = {}

        error_type = error_patterns["error_type"]
        error_message = error_patterns["error_message"].lower()

        # Timeout errors - reduce complexity
        if "timeout" in error_message or "time" in error_message:
            modifications.update({"max_improvement_iterations": 2, "model_timeout": 60})

        # Rate limit errors - add delays
        if "rate" in error_message or "limit" in error_message:
            modifications.update({"retry_delay": 5, "max_retries": 3})

        # Memory errors - reduce batch sizes
        if "memory" in error_message or "oom" in error_message:
            modifications.update({"batch_size": 1, "max_context_length": 2000})

        # Validation errors - relax constraints
        if error_type in ["ValidationError", "ValueError"]:
            modifications.update({"validation_strict": False, "allow_partial_results": True})

        return modifications

    def get_recovery_history(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get recovery history for a specific chain.

        Args:
            chain_id: The chain ID to get recovery history for

        Returns:
            List of recovery events
        """
        try:
            checkpoints = self.checkpoint_storage.get_chain_checkpoints(chain_id)

            recovery_events = []
            for cp in checkpoints:
                if "recovery_action" in cp.metadata:
                    recovery_events.append(
                        {
                            "timestamp": cp.timestamp,
                            "step": cp.current_step,
                            "iteration": cp.iteration,
                            "action": cp.metadata["recovery_action"],
                            "success": cp.metadata.get("recovery_success", False),
                        }
                    )

            return recovery_events

        except Exception as e:
            logger.warning(f"Failed to get recovery history for chain {chain_id}: {e}")
            return []

    def cleanup_old_checkpoints(self, max_age_days: int = 30) -> int:
        """Clean up old checkpoints to manage storage.

        Args:
            max_age_days: Maximum age of checkpoints to keep

        Returns:
            Number of checkpoints cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        try:
            # This would need to be implemented in the storage layer
            # For now, just log the intent
            logger.info(f"Would clean up checkpoints older than {cutoff_date}")
            return 0

        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
            return 0
