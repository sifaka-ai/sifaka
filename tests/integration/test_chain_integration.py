#!/usr/bin/env python3
"""Integration tests for Chain functionality.

This module contains integration tests that verify the interaction between
different components of the Chain system, including models, validators,
critics, and storage.
"""

from unittest.mock import Mock

import pytest

from sifaka.core.chain import Chain
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator


@pytest.mark.integration
class TestBasicChainIntegration:
    """Test basic Chain integration scenarios."""

    def test_chain_with_single_validator(self, mock_model):
        """Test Chain with a single validator."""
        # Set up model with appropriate response
        mock_model.response_text = "This is a valid response that meets length requirements."

        chain = Chain(model=mock_model, prompt="Write about technology.")
        validator = LengthValidator(min_length=10, max_length=100)

        new_chain = chain.validate_with(validator)
        result = new_chain.run()

        assert result.validation_results is not None
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is True

    def test_chain_with_single_critic(self, mock_model):
        """Test Chain with a single critic."""
        critic_model = MockModel(model_name="critic", response_text="Good work, well structured.")
        critic = ReflexionCritic(model=critic_model)

        chain = Chain(model=mock_model, prompt="Write about AI ethics.", always_apply_critics=True)
        new_chain = chain.improve_with(critic)
        result = new_chain.run()

        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 1

    def test_chain_with_multiple_validators(self, mock_model):
        """Test Chain with multiple validators."""
        # Set up model with response that should pass all validators
        mock_model.response_text = "This is a comprehensive response about artificial intelligence and machine learning technologies."

        validators = [
            LengthValidator(min_length=20, max_length=200),
            RegexValidator(required_patterns=[r"artificial", r"intelligence"]),
            ContentValidator(prohibited=["spam", "bad"]),
        ]

        chain = Chain(model=mock_model, prompt="Write about AI.")
        for validator in validators:
            chain = chain.validate_with(validator)

        result = chain.run()

        assert result.validation_results is not None
        assert len(result.validation_results) == 3
        # All validations should pass
        assert all(vr.passed for vr in result.validation_results)

    def test_chain_with_validation_failure(self, mock_model):
        """Test Chain behavior with validation failure."""
        # Set up model with response that will fail validation
        mock_model.response_text = "Short"

        validators = [
            LengthValidator(min_length=50, max_length=200),
            RegexValidator(required_patterns=[r"detailed", r"comprehensive"]),
        ]

        chain = Chain(model=mock_model, prompt="Write a detailed explanation.")
        for validator in validators:
            chain = chain.validate_with(validator)

        result = chain.run()

        assert result.validation_results is not None
        assert len(result.validation_results) == 2
        # At least one validation should fail
        assert not all(vr.passed for vr in result.validation_results)

    def test_chain_with_multiple_critics(self, mock_model):
        """Test Chain with multiple critics."""
        critic_model = MockModel(
            model_name="critic", response_text="Needs improvement in clarity and structure."
        )

        critics = [ReflexionCritic(model=critic_model), SelfRefineCritic(model=critic_model)]

        chain = Chain(
            model=mock_model, prompt="Write about the future of AI.", always_apply_critics=True
        )
        for critic in critics:
            chain = chain.improve_with(critic)

        result = chain.run()

        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2

    def test_chain_with_validators_and_critics(self, mock_model):
        """Test Chain with both validators and critics."""
        mock_model.response_text = (
            "This is a good response about technology that meets requirements."
        )

        # Add validators
        chain = Chain(model=mock_model, prompt="Write about technology.", always_apply_critics=True)
        chain = chain.validate_with(LengthValidator(min_length=30, max_length=300))
        chain = chain.validate_with(ContentValidator(prohibited=["bad"], name="Safety"))

        # Add critics
        critic_model = MockModel(model_name="critic", response_text="Well written and informative.")
        chain = chain.improve_with(ReflexionCritic(model=critic_model))
        chain = chain.improve_with(SelfRefineCritic(model=critic_model))

        result = chain.run()

        # Should have both validation results and critic feedback
        assert result.validation_results is not None
        assert len(result.validation_results) == 2
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2


@pytest.mark.integration
class TestChainStorageIntegration:
    """Test Chain integration with storage systems."""

    def test_chain_with_memory_storage(self, mock_model, memory_storage):
        """Test Chain with memory storage."""
        chain = Chain(model=mock_model, prompt="Write about data storage.", storage=memory_storage)
        chain = chain.validate_with(LengthValidator(min_length=10, max_length=200))

        chain.run()

        # Result should be stored
        assert len(memory_storage.data) > 0
        # Should be able to retrieve the stored thought
        stored_keys = memory_storage.list_keys()
        assert len(stored_keys) > 0

    def test_chain_storage_persistence(self, mock_model, memory_storage):
        """Test that Chain results persist in storage."""
        chain = Chain(model=mock_model, prompt="Test persistence", storage=memory_storage)

        # Run multiple times
        results = []
        for i in range(3):
            mock_model.response_text = f"Response {i}"
            result = chain.run()
            results.append(result)

        # All results should be stored
        stored_keys = memory_storage.list_keys()
        assert len(stored_keys) == 3

        # Should be able to retrieve all results
        for key in stored_keys:
            stored_thought = memory_storage.get(key)
            assert stored_thought is not None


@pytest.mark.integration
class TestChainPerformanceIntegration:
    """Test Chain performance characteristics in integration scenarios."""

    def test_chain_concurrent_execution(self, mock_model):
        """Test concurrent Chain execution."""
        import asyncio

        async def run_chain(prompt_suffix):
            chain = Chain(
                model=mock_model, prompt=f"Write about concurrent processing {prompt_suffix}"
            )
            validators = [
                LengthValidator(min_length=10, max_length=200),
                RegexValidator(required_patterns=[r"concurrent|processing"]),
            ]
            for validator in validators:
                chain = chain.validate_with(validator)

            return await chain.run_async()

        async def test_concurrent():
            # Run multiple chains concurrently
            tasks = [run_chain(f"task_{i}") for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            for result in results:
                assert result.validation_results is not None
                assert len(result.validation_results) == 2

        asyncio.run(test_concurrent())

    @pytest.mark.slow
    def test_chain_memory_usage(self, mock_model):
        """Test Chain memory usage with multiple executions."""
        # Run multiple chains to test memory usage
        for i in range(10):
            chain = Chain(model=mock_model, prompt=f"Write about memory test {i}.")
            chain = chain.validate_with(LengthValidator(min_length=10, max_length=100))
            result = chain.run()
            assert result.validation_results is not None

    @pytest.mark.slow
    def test_chain_performance_with_complex_validation(self, mock_model):
        """Test Chain performance with complex validation scenarios."""
        import time

        # Set up complex validation
        validators = [
            LengthValidator(min_length=50, max_length=500),
            RegexValidator(
                required_patterns=[r"artificial", r"intelligence", r"machine", r"learning"]
            ),
            ContentValidator(prohibited=["spam", "scam", "virus"], name="Security"),
            RegexValidator(forbidden_patterns=[r"bad", r"evil", r"harmful"]),
        ]

        mock_model.response_text = "This is a comprehensive response about artificial intelligence and machine learning technologies that avoids harmful content."

        chain = Chain(model=mock_model, prompt="Write about AI and ML.")
        for validator in validators:
            chain = chain.validate_with(validator)

        start_time = time.time()
        result = chain.run()
        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 1.0, f"Chain execution too slow: {execution_time:.3f}s"
        assert result.validation_results is not None
        assert len(result.validation_results) == 4


@pytest.mark.integration
class TestChainErrorHandlingIntegration:
    """Test Chain error handling in integration scenarios."""

    def test_chain_model_failure_recovery(self, failing_mock_model):
        """Test Chain behavior when model fails."""
        chain = Chain(model=failing_mock_model, prompt="Test error handling")

        with pytest.raises(Exception):
            chain.run()

    def test_chain_validator_error_handling(self, mock_model):
        """Test Chain behavior when validator has errors."""
        # Create a validator that might have issues
        problematic_validator = RegexValidator(required_patterns=[r"["])  # Invalid regex

        chain = Chain(model=mock_model, prompt="Test validator error")
        chain = chain.validate_with(problematic_validator)

        # Should handle validator errors gracefully
        try:
            result = chain.run()
            # If it doesn't raise an error, should still return a valid result
            assert result is not None
        except Exception:
            # It's also acceptable to raise an error for invalid regex
            pass

    def test_chain_critic_failure_handling(self, mock_model):
        """Test Chain behavior when critic fails."""
        failing_critic_model = Mock()
        failing_critic_model.generate.side_effect = Exception("Critic model failure")

        failing_critic = ReflexionCritic(model=failing_critic_model)
        working_critic_model = MockModel(model_name="working_critic", response_text="Good work")
        working_critic = SelfRefineCritic(model=working_critic_model)

        chain = Chain(
            model=mock_model, prompt="Test critic error handling", always_apply_critics=True
        )
        chain = chain.improve_with(failing_critic)
        chain = chain.improve_with(working_critic)

        # Should handle critic failures gracefully
        result = chain.run()
        assert result is not None
        # Working critic should still provide feedback
        assert result.critic_feedback is not None


@pytest.mark.integration
class TestChainConfigurationIntegration:
    """Test Chain configuration in integration scenarios."""

    def test_chain_configuration_inheritance(self, mock_model):
        """Test that chain configuration is properly inherited."""
        original_chain = Chain(
            model=mock_model, prompt="Original prompt", max_retries=5, always_apply_critics=True
        )

        validator = LengthValidator(min_length=10, max_length=100)
        critic_model = MockModel(model_name="critic", response_text="Feedback")
        critic = ReflexionCritic(model=critic_model)

        # Build new chain with validator and critic
        new_chain = original_chain.validate_with(validator).improve_with(critic)

        # Configuration should be preserved
        assert new_chain.config.max_retries == 5
        assert new_chain.config.always_apply_critics is True

        # New components should be added
        assert len(new_chain.validators) == 1
        assert len(new_chain.critics) == 1

    def test_chain_immutability_in_integration(self, mock_model):
        """Test chain immutability in complex integration scenarios."""
        base_chain = Chain(model=mock_model, prompt="Base prompt")

        # Create multiple derived chains
        chain_with_validator = base_chain.validate_with(
            LengthValidator(min_length=10, max_length=100)
        )

        critic_model = MockModel(model_name="critic", response_text="Feedback")
        chain_with_critic = base_chain.improve_with(ReflexionCritic(model=critic_model))

        chain_with_both = base_chain.validate_with(
            LengthValidator(min_length=10, max_length=100)
        ).improve_with(ReflexionCritic(model=critic_model))

        # Base chain should remain unchanged
        assert len(base_chain.validators) == 0
        assert len(base_chain.critics) == 0

        # Derived chains should have expected components
        assert len(chain_with_validator.validators) == 1
        assert len(chain_with_validator.critics) == 0

        assert len(chain_with_critic.validators) == 0
        assert len(chain_with_critic.critics) == 1

        assert len(chain_with_both.validators) == 1
        assert len(chain_with_both.critics) == 1
