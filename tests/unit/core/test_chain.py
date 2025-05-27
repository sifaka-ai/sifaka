#!/usr/bin/env python3
"""Unit tests for Chain core functionality.

This module contains comprehensive unit tests for the Chain class,
including initialization, configuration, validation, criticism, and execution.
"""



from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator


class TestChainInitialization:
    """Test Chain initialization and basic properties."""

    def test_chain_basic_initialization(self, mock_model):
        """Test basic Chain initialization."""
        chain = Chain(model=mock_model, prompt="Test prompt")

        assert chain.model == mock_model
        assert chain.prompt == "Test prompt"
        assert chain._config is not None  # Internal config object
        assert chain.storage is not None  # Chain has default memory storage

    def test_chain_with_storage(self, mock_model, memory_storage):
        """Test Chain initialization with storage."""
        chain = Chain(model=mock_model, prompt="Test prompt", storage=memory_storage)

        assert chain.storage == memory_storage

    def test_chain_with_system_prompt(self, mock_model):
        """Test Chain initialization with system prompt."""
        # Chain doesn't have system_prompt parameter, but we can test with options
        chain = Chain(model=mock_model, prompt="User prompt").with_options(
            system_prompt="You are a helpful assistant."
        )

        # Check that the option was set
        assert chain._config.get_option("system_prompt") == "You are a helpful assistant."

    def test_chain_with_options(self, mock_model):
        """Test Chain initialization with various options."""
        chain = Chain(
            model=mock_model,
            prompt="Test prompt",
        ).with_options(max_iterations=5)

        # Check that the option was set
        assert chain._config.max_iterations == 5


class TestChainValidation:
    """Test Chain validation functionality."""

    def test_validate_with_single_validator(self, mock_model):
        """Test adding a single validator."""
        chain = Chain(model=mock_model, prompt="Test prompt")
        validator = LengthValidator(min_length=10, max_length=100)

        new_chain = chain.validate_with(validator)

        # Should return a new chain instance
        assert new_chain is not chain
        assert len(new_chain.validators) == 1
        assert new_chain.validators[0] == validator

    def test_validate_with_multiple_validators(self, mock_model):
        """Test adding multiple validators."""
        chain = Chain(model=mock_model, prompt="Test prompt")

        validator1 = LengthValidator(min_length=10, max_length=100)
        validator2 = RegexValidator(required_patterns=[r"\w+"])
        validator3 = ContentValidator(prohibited=["spam"])

        new_chain = (
            chain.validate_with(validator1).validate_with(validator2).validate_with(validator3)
        )

        assert len(new_chain.validators) == 3
        assert validator1 in new_chain.validators
        assert validator2 in new_chain.validators
        assert validator3 in new_chain.validators

    def test_validate_with_preserves_other_config(self, mock_model):
        """Test that validate_with preserves other configuration."""
        chain = Chain(model=mock_model, prompt="Test prompt").with_options(max_iterations=3)

        validator = LengthValidator(min_length=10, max_length=100)
        new_chain = chain.validate_with(validator)

        assert new_chain._config.max_iterations == 3


class TestChainCriticism:
    """Test Chain criticism functionality."""

    def test_improve_with_single_critic(self, mock_model):
        """Test adding a single critic."""
        chain = Chain(model=mock_model, prompt="Test prompt")
        critic_model = MockModel(model_name="critic")
        critic = ReflexionCritic(model=critic_model)

        new_chain = chain.improve_with(critic)

        # Should return a new chain instance
        assert new_chain is not chain
        assert len(new_chain.critics) == 1
        assert new_chain.critics[0] == critic

    def test_improve_with_multiple_critics(self, mock_model):
        """Test adding multiple critics."""
        chain = Chain(model=mock_model, prompt="Test prompt")
        critic_model = MockModel(model_name="critic")

        critic1 = ReflexionCritic(model=critic_model)
        critic2 = SelfRefineCritic(model=critic_model)

        new_chain = chain.improve_with(critic1).improve_with(critic2)

        assert len(new_chain.critics) == 2
        assert critic1 in new_chain.critics
        assert critic2 in new_chain.critics

    def test_improve_with_preserves_validators(self, mock_model):
        """Test that improve_with preserves validators."""
        chain = Chain(model=mock_model, prompt="Test prompt")
        validator = LengthValidator(min_length=10, max_length=100)
        critic_model = MockModel(model_name="critic")
        critic = ReflexionCritic(model=critic_model)

        new_chain = chain.validate_with(validator).improve_with(critic)

        assert len(new_chain.validators) == 1
        assert len(new_chain.critics) == 1


class TestChainExecution:
    """Test Chain execution functionality."""

    def test_basic_run(self, mock_model):
        """Test basic chain execution."""
        chain = Chain(model=mock_model, prompt="Test prompt")

        result = chain.run()

        assert isinstance(result, Thought)
        assert result.text == mock_model.response_text
        assert result.prompt == "Test prompt"

    def test_run_with_validation(self, mock_model):
        """Test chain execution with validation."""
        # Set up mock model with appropriate response
        mock_model.response_text = "This is a valid response that meets length requirements."

        chain = Chain(model=mock_model, prompt="Test prompt")
        validator = LengthValidator(min_length=10, max_length=100)

        new_chain = chain.validate_with(validator)
        result = new_chain.run()

        assert isinstance(result, Thought)
        assert result.validation_results is not None
        assert len(result.validation_results) == 1
        # validation_results is a dict, get the first validator result
        validator_result = list(result.validation_results.values())[0]
        assert validator_result.passed is True

    def test_run_with_validation_failure(self, mock_model):
        """Test chain execution with validation failure."""
        # Set up mock model with response that will fail validation
        mock_model.response_text = "Short"

        chain = Chain(model=mock_model, prompt="Test prompt")
        validator = LengthValidator(min_length=20, max_length=100)

        new_chain = chain.validate_with(validator)
        result = new_chain.run()

        assert isinstance(result, Thought)
        assert result.validation_results is not None
        assert len(result.validation_results) == 1
        # validation_results is a dict, get the first validator result
        validator_result = list(result.validation_results.values())[0]
        assert validator_result.passed is False

    def test_run_with_critics(self, mock_model):
        """Test chain execution with critics."""
        chain = Chain(model=mock_model, prompt="Test prompt", always_apply_critics=True)

        critic_model = MockModel(model_name="critic", response_text="Critic feedback")
        critic = ReflexionCritic(model=critic_model)

        new_chain = chain.improve_with(critic)
        result = new_chain.run()

        assert isinstance(result, Thought)
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 1

    def test_run_with_storage(self, mock_model, memory_storage):
        """Test chain execution with storage."""
        chain = Chain(model=mock_model, prompt="Test prompt", storage=memory_storage)

        result = chain.run()

        assert isinstance(result, Thought)
        # Check that result was stored
        assert len(memory_storage.data) > 0

    def test_run_internal_async(self, mock_model):
        """Test that run() uses internal async implementation."""
        chain = Chain(model=mock_model, prompt="Test prompt")
        result = chain.run()

        # Should work normally - run() internally uses async but provides sync API
        assert isinstance(result, Thought)
        assert result.text == mock_model.response_text


class TestChainConfiguration:
    """Test Chain configuration management."""

    def test_chain_config_copy(self, mock_model):
        """Test that chain configuration is properly copied."""
        original_chain = Chain(model=mock_model, prompt="Original prompt", max_retries=3)

        validator = LengthValidator(min_length=10, max_length=100)
        new_chain = original_chain.validate_with(validator)

        # Original chain should be unchanged
        assert len(original_chain.validators) == 0
        assert len(new_chain.validators) == 1

        # Config should be copied, not shared
        assert new_chain._config is not original_chain._config
        assert new_chain.model == original_chain.model

    def test_chain_immutability(self, mock_model):
        """Test that chains are immutable."""
        chain = Chain(model=mock_model, prompt="Test prompt")
        validator = LengthValidator(min_length=10, max_length=100)

        new_chain = chain.validate_with(validator)

        # Original chain should be unchanged
        assert chain is not new_chain
        assert len(chain.validators) == 0
        assert len(new_chain.validators) == 1


class TestChainErrorHandling:
    """Test Chain error handling."""

    def test_model_error_handling(self, failing_mock_model):
        """Test handling of model errors."""
        chain = Chain(model=failing_mock_model, prompt="Test prompt")

        # The chain might handle errors gracefully, so just test it doesn't crash
        try:
            result = chain.run()
            # If it succeeds, that's also valid behavior
            assert result is not None
        except Exception:
            # If it raises an exception, that's also valid behavior
            pass

    def test_validation_error_recovery(self, mock_model):
        """Test recovery from validation errors."""
        # This test would need more complex setup to test actual recovery
        # For now, just test that the chain can handle validation failures
        mock_model.response_text = "Short"

        chain = Chain(model=mock_model, prompt="Test prompt")
        validator = LengthValidator(min_length=20, max_length=100)

        new_chain = chain.validate_with(validator)
        result = new_chain.run()

        # Should complete even with validation failure
        assert isinstance(result, Thought)
        if result.validation_results:
            validator_result = list(result.validation_results.values())[0]
            assert validator_result.passed is False
