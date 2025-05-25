"""Tests for the Chain Checkpoint Recovery System.

This module tests the checkpoint and recovery functionality to ensure
robust chain execution with automatic recovery capabilities.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from sifaka.core.chain import Chain
from sifaka.models.base import MockModel
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.storage.checkpoints import ChainCheckpoint, CachedCheckpointStorage
from sifaka.recovery.manager import RecoveryManager, RecoveryStrategy, RecoveryAction
from sifaka.core.thought import Thought
from sifaka.utils.error_handling import ChainError


class TestChainCheckpointRecovery:
    """Test the Chain checkpoint and recovery functionality."""

    @pytest.fixture
    def checkpoint_storage(self):
        """Create a checkpoint storage for testing."""
        # Create a mock storage that actually stores checkpoints
        mock_storage = Mock()

        # Storage for checkpoints
        stored_checkpoints = {}

        def mock_set(key, value, metadata=None):
            stored_checkpoints[key] = value

        def mock_get(key):
            return stored_checkpoints.get(key)

        def mock_search_similar(query, limit=10):
            # Return all stored checkpoints for testing
            return list(stored_checkpoints.values())

        mock_storage.get = mock_get
        mock_storage.set = mock_set
        mock_storage.search_similar = mock_search_similar
        mock_storage.get_stats = Mock(return_value={})

        # Add memory attribute for in-memory storage simulation
        mock_storage.memory = Mock()
        mock_storage.memory.data = stored_checkpoints

        return CachedCheckpointStorage(mock_storage)

    @pytest.fixture
    def chain_with_checkpoints(self, checkpoint_storage):
        """Create a chain with checkpoint storage configured."""
        model = MockModel(model_name="test-model")
        chain = Chain(
            model=model, checkpoint_storage=checkpoint_storage, max_improvement_iterations=2
        )
        chain.with_prompt("Test prompt for checkpoint recovery")
        return chain

    def test_chain_initialization_with_checkpoints(self, checkpoint_storage):
        """Test that chain initializes correctly with checkpoint storage."""
        model = MockModel(model_name="test-model")
        chain = Chain(model=model, checkpoint_storage=checkpoint_storage)

        assert chain._checkpoint_storage is checkpoint_storage
        assert chain._recovery_manager is not None
        assert chain._current_checkpoint is None

    def test_run_with_recovery_no_checkpoints(self, chain_with_checkpoints):
        """Test run_with_recovery when no existing checkpoints exist."""
        result = chain_with_checkpoints.run_with_recovery()

        assert result is not None
        assert result.text is not None
        assert result.prompt == "Test prompt for checkpoint recovery"

    def test_checkpoint_creation_during_execution(self, chain_with_checkpoints):
        """Test that checkpoints are created during execution."""
        # Add a validator to ensure we go through validation step
        validator = LengthValidator(min_length=1, max_length=1000)
        chain_with_checkpoints.validate_with(validator)

        result = chain_with_checkpoints.run_with_recovery()

        # Check that checkpoints were created
        checkpoints = chain_with_checkpoints.get_checkpoint_history()
        assert len(checkpoints) > 0

        # Should have at least initialization and completion checkpoints
        checkpoint_steps = [cp.current_step for cp in checkpoints]
        assert "initialization" in checkpoint_steps
        assert "complete" in checkpoint_steps

    def test_checkpoint_contains_correct_data(self, chain_with_checkpoints):
        """Test that checkpoints contain the correct execution data."""
        validator = LengthValidator(min_length=1, max_length=1000)
        chain_with_checkpoints.validate_with(validator)

        chain_with_checkpoints.run_with_recovery()

        checkpoints = chain_with_checkpoints.get_checkpoint_history()
        assert len(checkpoints) > 0

        checkpoint = checkpoints[0]
        assert checkpoint.chain_id == chain_with_checkpoints._chain_id
        assert checkpoint.thought is not None
        assert checkpoint.thought.prompt == "Test prompt for checkpoint recovery"
        assert isinstance(checkpoint.timestamp, datetime)
        assert checkpoint.metadata is not None

    def test_recovery_manager_initialization(self, checkpoint_storage):
        """Test that recovery manager is properly initialized."""
        recovery_manager = RecoveryManager(checkpoint_storage)

        assert recovery_manager.checkpoint_storage is checkpoint_storage

    def test_recovery_action_creation(self):
        """Test creation of recovery actions."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY_CURRENT_STEP,
            description="Retry the current step",
            confidence=0.8,
            estimated_success_rate=0.7,
        )

        assert action.strategy == RecoveryStrategy.RETRY_CURRENT_STEP
        assert action.description == "Retry the current step"
        assert action.confidence == 0.8
        assert action.estimated_success_rate == 0.7

    def test_failure_analysis(self, checkpoint_storage):
        """Test failure analysis and recovery suggestion generation."""
        recovery_manager = RecoveryManager(checkpoint_storage)

        # Create a mock checkpoint
        thought = Thought(prompt="Test prompt")
        checkpoint = ChainCheckpoint(
            chain_id="test-chain",
            current_step="generation",
            iteration=1,
            thought=thought,
            recovery_point="generation",
        )

        # Analyze a mock failure
        error = RuntimeError("Test error")
        actions = recovery_manager.analyze_failure(checkpoint, error)

        assert len(actions) > 0
        assert all(isinstance(action, RecoveryAction) for action in actions)

        # Should be sorted by confidence (highest first)
        confidences = [action.confidence for action in actions]
        assert confidences == sorted(confidences, reverse=True)

    def test_recovery_strategy_types(self, checkpoint_storage):
        """Test that different recovery strategies are generated."""
        recovery_manager = RecoveryManager(checkpoint_storage)

        thought = Thought(prompt="Test prompt")
        checkpoint = ChainCheckpoint(
            chain_id="test-chain",
            current_step="validation",
            iteration=2,  # Multiple iterations to enable restart strategies
            thought=thought,
            recovery_point="validation",
        )

        error = ValueError("Validation failed")
        actions = recovery_manager.analyze_failure(checkpoint, error)

        strategies = [action.strategy for action in actions]

        # Should include retry and restart strategies
        assert RecoveryStrategy.RETRY_CURRENT_STEP in strategies
        assert RecoveryStrategy.RESTART_ITERATION in strategies

    def test_parameter_modification_suggestions(self, checkpoint_storage):
        """Test that parameter modifications are suggested for specific errors."""
        recovery_manager = RecoveryManager(checkpoint_storage)

        thought = Thought(prompt="Test prompt")
        checkpoint = ChainCheckpoint(
            chain_id="test-chain",
            current_step="generation",
            iteration=1,
            thought=thought,
            recovery_point="generation",
        )

        # Test timeout error
        timeout_error = RuntimeError("Connection timeout")
        actions = recovery_manager.analyze_failure(checkpoint, timeout_error)

        # Should include parameter modification action
        param_actions = [a for a in actions if a.strategy == RecoveryStrategy.MODIFY_PARAMETERS]
        assert len(param_actions) > 0

        param_action = param_actions[0]
        assert param_action.parameters is not None
        assert len(param_action.parameters) > 0

    def test_recovery_action_application(self, chain_with_checkpoints):
        """Test applying recovery actions to a chain."""
        # Create a recovery action
        action = RecoveryAction(
            strategy=RecoveryStrategy.MODIFY_PARAMETERS,
            description="Modify timeout parameters",
            confidence=0.8,
            parameters={"model_timeout": 60, "max_retries": 3},
        )

        # Apply the action
        success = chain_with_checkpoints._apply_recovery_action(action)

        assert success is True
        assert chain_with_checkpoints._options["model_timeout"] == 60
        assert chain_with_checkpoints._options["max_retries"] == 3

    def test_get_recovery_suggestions(self, chain_with_checkpoints):
        """Test getting recovery suggestions from a chain."""
        # Create a mock checkpoint
        thought = Thought(prompt="Test prompt")
        checkpoint = ChainCheckpoint(
            chain_id=chain_with_checkpoints._chain_id,
            current_step="generation",
            iteration=1,
            thought=thought,
            recovery_point="generation",
        )
        chain_with_checkpoints._current_checkpoint = checkpoint

        error = RuntimeError("Test error")
        suggestions = chain_with_checkpoints.get_recovery_suggestions(error)

        assert len(suggestions) > 0
        assert all(isinstance(s, RecoveryAction) for s in suggestions)

    def test_checkpoint_history_retrieval(self, chain_with_checkpoints):
        """Test retrieving checkpoint history for a chain."""
        # Run the chain to create checkpoints
        chain_with_checkpoints.run_with_recovery()

        # Get checkpoint history
        history = chain_with_checkpoints.get_checkpoint_history()

        assert len(history) > 0
        assert all(isinstance(cp, ChainCheckpoint) for cp in history)
        assert all(cp.chain_id == chain_with_checkpoints._chain_id for cp in history)

        # Should be sorted by timestamp
        timestamps = [cp.timestamp for cp in history]
        assert timestamps == sorted(timestamps)

    def test_run_with_recovery_fallback(self):
        """Test that run_with_recovery falls back to regular run() without checkpoint storage."""
        model = MockModel(model_name="test-model")
        chain = Chain(model=model)  # No checkpoint storage
        chain.with_prompt("Test prompt")

        # Should fall back to regular run() and still work
        result = chain.run_with_recovery()

        assert result is not None
        assert result.text is not None


class TestRecoveryManager:
    """Test the RecoveryManager class independently."""

    @pytest.fixture
    def recovery_manager(self):
        """Create a recovery manager for testing."""
        # Create a mock storage that actually stores checkpoints
        mock_storage = Mock()

        # Storage for checkpoints
        stored_checkpoints = {}

        def mock_set(key, value):
            stored_checkpoints[key] = value

        def mock_get(key):
            return stored_checkpoints.get(key)

        def mock_search_similar(query, limit=10):
            # Return all stored checkpoints for testing
            return list(stored_checkpoints.values())

        mock_storage.get = mock_get
        mock_storage.set = mock_set
        mock_storage.search_similar = mock_search_similar
        mock_storage.get_stats = Mock(return_value={})

        # Add memory attribute for in-memory storage simulation
        mock_storage.memory = Mock()
        mock_storage.memory.data = stored_checkpoints

        checkpoint_storage = CachedCheckpointStorage(mock_storage)
        return RecoveryManager(checkpoint_storage)

    def test_error_pattern_analysis(self, recovery_manager):
        """Test error pattern analysis functionality."""
        thought = Thought(prompt="Test prompt")
        checkpoint = ChainCheckpoint(
            chain_id="test-chain",
            current_step="generation",
            iteration=1,
            thought=thought,
            recovery_point="generation",
        )

        error = RuntimeError("Connection timeout")

        # This should not raise an exception
        actions = recovery_manager.analyze_failure(checkpoint, error)
        assert isinstance(actions, list)

    def test_recovery_history_tracking(self, recovery_manager):
        """Test recovery history tracking."""
        chain_id = "test-chain-history"

        # Should return empty list for non-existent chain
        history = recovery_manager.get_recovery_history(chain_id)
        assert history == []

    def test_checkpoint_cleanup(self, recovery_manager):
        """Test checkpoint cleanup functionality."""
        # Should not raise an exception
        cleaned_count = recovery_manager.cleanup_old_checkpoints(max_age_days=30)
        assert isinstance(cleaned_count, int)
        assert cleaned_count >= 0


if __name__ == "__main__":
    pytest.main([__file__])
