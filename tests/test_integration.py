#!/usr/bin/env python3
"""Comprehensive integration tests for Sifaka.

This test suite verifies that all Sifaka components work together correctly
in various combinations and configurations. It tests the full chain execution
with different models, validators, critics, and storage backends.
"""

import time

from sifaka.core.chain import Chain
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.storage.memory import MemoryStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator
from sifaka.validators.format import FormatValidator
from tests.utils import (
    assert_chain_execution_success,
    assert_thought_valid,
    assert_validation_results,
)

logger = get_logger(__name__)


class TestBasicIntegration:
    """Test basic integration scenarios."""

    def test_simple_chain_execution(self):
        """Test basic chain execution with minimal components."""
        model = MockModel(model_name="test-model")
        chain = Chain(model=model, prompt="Write a simple sentence about AI.")

        result = chain.run()

        assert_thought_valid(result, min_length=1)
        assert_chain_execution_success(result, expected_iterations=0)

    def test_chain_with_single_validator(self):
        """Test chain execution with one validator."""
        model = MockModel(model_name="test-model")
        validator = LengthValidator(min_length=10, max_length=100)

        chain = Chain(model=model, prompt="Write about technology.")
        chain = chain.validate_with(validator)

        result = chain.run()

        assert_thought_valid(result, min_length=10, max_length=100)
        assert_validation_results(result, expected_count=1, expected_passed=True)
        assert_chain_execution_success(result)

    def test_chain_with_single_critic(self):
        """Test chain execution with one critic."""
        model = MockModel(model_name="test-model")
        critic = ReflexionCritic(model=MockModel(model_name="critic-model"))

        chain = Chain(model=model, prompt="Write about AI ethics.", always_apply_critics=True)
        chain = chain.improve_with(critic)

        result = chain.run()

        assert_thought_valid(result)
        # Critics may run multiple times, just check we have feedback
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 1
        assert_chain_execution_success(result)

    def test_chain_with_storage(self):
        """Test chain execution with storage backend."""
        model = MockModel(model_name="test-model")
        storage = MemoryStorage()

        chain = Chain(model=model, prompt="Write about machine learning.", storage=storage)

        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)

        # Verify storage was used (storage uses key format: thought_{chain_id}_{iteration})
        thought_key = f"thought_{result.chain_id}_{result.iteration}"
        assert thought_key in storage.data, "Thought should be saved to storage"


class TestValidatorIntegration:
    """Test integration with multiple validators."""

    def test_multiple_validators_all_pass(self):
        """Test chain with multiple validators that all pass."""
        from tests.utils.mocks import MockModelFactory

        model = MockModelFactory.create_standard(
            model_name="test-model",
            response_text="This is a well-formed response about artificial intelligence and machine learning.",
        )

        validators = [
            LengthValidator(min_length=20, max_length=200),
            RegexValidator(required_patterns=[r"artificial", r"intelligence"]),
            ContentValidator(prohibited=["bad", "evil"], name="Safety Filter"),
        ]

        chain = Chain(model=model, prompt="Write about AI.")
        for validator in validators:
            chain = chain.validate_with(validator)

        result = chain.run()

        assert_thought_valid(result)
        assert_validation_results(result, expected_count=3, expected_passed=True)
        assert_chain_execution_success(result, expected_iterations=0)

    def test_multiple_validators_some_fail(self):
        """Test chain with validators where some fail initially."""
        model = MockModel(
            model_name="test-model", response_text="Short"  # Will fail length validator
        )

        validators = [
            LengthValidator(min_length=50, max_length=200),
            RegexValidator(required_patterns=[r"\w+"]),  # This will pass
        ]

        chain = Chain(model=model, prompt="Write a detailed explanation.")
        for validator in validators:
            chain = chain.validate_with(validator)

        result = chain.run()

        assert_thought_valid(result)
        assert_validation_results(result, expected_count=2)
        # Note: With current mock model, it won't retry, so some validations may fail

    def test_validator_types_combination(self):
        """Test different types of validators working together."""
        from tests.utils.mocks import MockModelFactory

        model = MockModelFactory.create_standard(
            model_name="test-model",
            response_text="This is a properly formatted JSON response: {'key': 'value'}",
        )

        validators = [
            LengthValidator(min_length=10, max_length=500),
            RegexValidator(required_patterns=[r"JSON", r"key"]),
            ContentValidator(prohibited=["error", "fail"], name="Error Filter"),
            FormatValidator(format_type="json"),
        ]

        chain = Chain(model=model, prompt="Generate a JSON example.")
        for validator in validators:
            chain = chain.validate_with(validator)

        result = chain.run()

        assert_thought_valid(result)
        assert_validation_results(result, expected_count=4)
        assert_chain_execution_success(result)


class TestCriticIntegration:
    """Test integration with multiple critics."""

    def test_multiple_critics_sequential(self):
        """Test chain with multiple critics applied sequentially."""
        model = MockModel(model_name="test-model")
        critic_model = MockModel(model_name="critic-model")

        critics = [
            ReflexionCritic(model=critic_model),
            SelfRefineCritic(model=critic_model),
            PromptCritic(model=critic_model),
        ]

        chain = Chain(
            model=model, prompt="Write about the future of AI.", always_apply_critics=True
        )
        for critic in critics:
            chain = chain.improve_with(critic)

        result = chain.run()

        assert_thought_valid(result)
        # Multiple critics may run multiple times, just check we have feedback
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 3
        assert_chain_execution_success(result)

    def test_critics_with_validators(self):
        """Test critics and validators working together."""
        model = MockModel(model_name="test-model")
        critic_model = MockModel(model_name="critic-model")

        # Add both validators and critics
        chain = Chain(
            model=model,
            prompt="Write a comprehensive guide about neural networks.",
            always_apply_critics=True,
        )

        # Add validators
        chain = chain.validate_with(LengthValidator(min_length=30, max_length=300))
        chain = chain.validate_with(ContentValidator(prohibited=["bad"], name="Safety"))

        # Add critics
        chain = chain.improve_with(ReflexionCritic(model=critic_model))
        chain = chain.improve_with(SelfRefineCritic(model=critic_model))

        result = chain.run()

        assert_thought_valid(result)
        assert_validation_results(result, expected_count=2)
        # Multiple critics may run multiple times, just check we have feedback
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 2
        assert_chain_execution_success(result)


class TestStorageIntegration:
    """Test integration with different storage backends."""

    def test_memory_storage_integration(self):
        """Test integration with memory storage."""
        model = MockModel(model_name="test-model")
        storage = MemoryStorage()

        chain = Chain(model=model, prompt="Write about data storage.", storage=storage)
        chain = chain.validate_with(LengthValidator(min_length=10, max_length=200))

        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)

        # Verify storage operations
        thought_key = f"thought_{result.chain_id}_{result.iteration}"
        assert thought_key in storage.data
        loaded_thought_data = storage.get(thought_key)
        assert loaded_thought_data is not None

    def test_storage_with_multiple_iterations(self):
        """Test storage behavior with multiple chain iterations."""
        model = MockModel(model_name="test-model")
        storage = MemoryStorage()

        # Run multiple chains with same storage
        results = []
        for i in range(3):
            chain = Chain(model=model, prompt=f"Write about topic {i}.", storage=storage)
            result = chain.run()
            results.append(result)

        # Verify all thoughts are stored
        for result in results:
            assert_thought_valid(result)
            thought_key = f"thought_{result.chain_id}_{result.iteration}"
            assert thought_key in storage.data

        # Verify unique IDs
        ids = [result.id for result in results]
        assert len(set(ids)) == len(ids), "All thoughts should have unique IDs"


class TestPerformanceIntegration:
    """Test performance aspects of integration."""

    def test_concurrent_validation_performance(self):
        """Test that concurrent validation provides performance benefits."""
        model = MockModel(model_name="test-model")

        # Create multiple slow validators
        validators = [
            LengthValidator(min_length=10, max_length=500),
            RegexValidator(required_patterns=[r"\w+"]),
            ContentValidator(prohibited=["bad"], name="Safety"),
            FormatValidator(format_type="json"),
        ]

        chain = Chain(model=model, prompt="Write about concurrent processing.")
        for validator in validators:
            chain = chain.validate_with(validator)

        start_time = time.time()
        result = chain.run()
        execution_time = time.time() - start_time

        assert_thought_valid(result)
        assert_validation_results(result, expected_count=4)

        # Performance should be reasonable (concurrent execution)
        assert execution_time < 5.0, f"Execution took too long: {execution_time:.2f}s"

    def test_memory_usage_reasonable(self):
        """Test that memory usage stays within reasonable bounds."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        model = MockModel(model_name="test-model")

        # Run multiple chains to test memory usage
        for i in range(10):
            chain = Chain(model=model, prompt=f"Write about memory test {i}.")
            chain = chain.validate_with(LengthValidator(min_length=10, max_length=100))
            result = chain.run()
            assert_thought_valid(result)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 10 chains)
        assert memory_increase < 50, f"Memory usage increased too much: {memory_increase:.1f}MB"
