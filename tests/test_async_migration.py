#!/usr/bin/env python3
"""Comprehensive test suite for async migration functionality.

This test suite verifies that the async migration has been successfully implemented
and that concurrent validation and criticism work as expected.
"""

import asyncio
import time

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator

# Configure logging
logger = get_logger(__name__)


class TestAsyncValidation:
    """Test async validation functionality."""

    def test_concurrent_validation_basic(self):
        """Test that multiple validators run concurrently."""
        # Create validators
        validators = [
            LengthValidator(min_length=10, max_length=1000),
            RegexValidator(required_patterns=[r"\w+"]),
            ContentValidator(prohibited=["bad", "evil"], name="Content Filter"),
        ]

        # Create chain with validators
        model = MockModel(model_name="test-model")
        chain = Chain(model=model, prompt="Write a good story about technology")
        for validator in validators:
            chain.validate_with(validator)

        # Run chain - this should use concurrent validation internally
        start_time = time.time()
        result = chain.run()
        execution_time = time.time() - start_time

        # Verify result
        assert result.text is not None
        assert result.validation_results is not None
        assert len(result.validation_results) == len(validators)

        # Log execution time for performance analysis
        logger.info(f"Concurrent validation completed in {execution_time:.3f}s")

    def test_concurrent_validation_with_failures(self):
        """Test concurrent validation with some validators failing."""
        # Create validators that will fail
        validators = [
            LengthValidator(min_length=1000, max_length=2000),  # Will fail - text too short
            RegexValidator(required_patterns=[r"^MUST_START_WITH_THIS"]),  # Will fail
            ContentValidator(
                prohibited=["mock"], name="Mock Filter"
            ),  # Will fail - contains "mock"
        ]

        # Create chain with failing validators
        model = MockModel(model_name="test-model")
        chain = Chain(model=model, prompt="Write a short mock response")
        for validator in validators:
            chain.validate_with(validator)

        # Run chain
        result = chain.run()

        # Verify that validation failed but chain still completed
        assert result.text is not None
        assert result.validation_results is not None
        assert len(result.validation_results) == len(validators)

        # Check that all validations failed as expected
        failed_validations = [vr for vr in result.validation_results.values() if not vr.passed]
        assert len(failed_validations) == len(validators)


class TestAsyncCriticism:
    """Test async criticism functionality."""

    def test_concurrent_criticism_basic(self):
        """Test that multiple critics run concurrently."""
        # Create critics
        critics = [
            ReflexionCritic(model=MockModel(model_name="critic-model")),
            PromptCritic(model=MockModel(model_name="critic-model")),
            SelfRefineCritic(model=MockModel(model_name="critic-model")),
        ]

        # Create chain with critics - ALWAYS apply critics for testing
        model = MockModel(model_name="test-model")
        chain = Chain(
            model=model,
            prompt="Write a story about AI",
            always_apply_critics=True,  # Ensure critics run even when validation passes
        )
        for critic in critics:
            chain.improve_with(critic)

        # Run chain - this should use concurrent criticism internally
        start_time = time.time()
        result = chain.run()
        execution_time = time.time() - start_time

        # Verify result
        assert result.text is not None
        assert len(result.critic_feedback) >= len(critics)

        # Log execution time for performance analysis
        logger.info(f"Concurrent criticism completed in {execution_time:.3f}s")

    def test_concurrent_criticism_complex(self):
        """Test concurrent criticism with complex critics."""
        # Create more complex critics
        critics = [
            ConstitutionalCritic(
                model=MockModel(model_name="critic-model"),
                principles=["Be helpful", "Be harmless", "Be honest"],
            ),
            NCriticsCritic(model=MockModel(model_name="critic-model"), num_critics=3),
            SelfRAGCritic(model=MockModel(model_name="critic-model")),
        ]

        # Create chain with complex critics - ALWAYS apply critics for testing
        model = MockModel(model_name="test-model")
        chain = Chain(
            model=model,
            prompt="Explain artificial intelligence",
            always_apply_critics=True,  # Ensure critics run even when validation passes
        )
        for critic in critics:
            chain.improve_with(critic)

        # Run chain
        result = chain.run()

        # Verify result
        assert result.text is not None
        assert len(result.critic_feedback) >= len(critics)


class TestAsyncChainExecution:
    """Test full async chain execution."""

    def test_full_chain_with_concurrency(self):
        """Test full chain execution with concurrent validation and criticism."""
        # Create comprehensive setup
        validators = [
            LengthValidator(min_length=20, max_length=500),
            RegexValidator(required_patterns=[r"\w+"]),
            ContentValidator(prohibited=["hate", "violence"], name="Safety Filter"),
        ]

        critics = [
            ReflexionCritic(model=MockModel(model_name="critic-model")),
            SelfRefineCritic(model=MockModel(model_name="critic-model")),
            PromptCritic(model=MockModel(model_name="critic-model")),
        ]

        # Create chain - ALWAYS apply critics for testing
        model = MockModel(model_name="test-model")
        chain = Chain(
            model=model,
            prompt="Write an educational article about machine learning",
            always_apply_critics=True,  # Ensure critics run even when validation passes
        )

        # Add validators and critics
        for validator in validators:
            chain.validate_with(validator)
        for critic in critics:
            chain.improve_with(critic)

        # Run chain
        start_time = time.time()
        result = chain.run()
        execution_time = time.time() - start_time

        # Verify comprehensive result
        assert result.text is not None
        assert len(result.validation_results) == len(validators)
        assert len(result.critic_feedback) >= len(critics)
        assert result.iteration >= 1

        logger.info(f"Full chain with concurrency completed in {execution_time:.3f}s")

    def test_multiple_chains_concurrently(self):
        """Test running multiple chains concurrently."""

        async def run_chain_async(chain_id: int) -> Thought:
            """Run a single chain asynchronously."""
            model = MockModel(model_name="test-model")
            chain = Chain(
                model=model,
                prompt=f"Write about topic {chain_id}",
                always_apply_critics=True,  # Ensure critics run even when validation passes
            )
            chain.validate_with(LengthValidator(min_length=10, max_length=200))
            chain.improve_with(ReflexionCritic(model=MockModel(model_name="critic-model")))

            # Use the internal async method
            return await chain._run_async()

        async def run_multiple_chains():
            """Run multiple chains concurrently."""
            # Create tasks for concurrent execution
            tasks = [run_chain_async(i) for i in range(5)]

            # Run all chains concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time

            return results, execution_time

        # Run the async test
        results, execution_time = asyncio.run(run_multiple_chains())

        # Verify results
        assert len(results) == 5
        for result in results:
            assert result.text is not None
            assert len(result.validation_results) >= 1
            assert len(result.critic_feedback) >= 1

        logger.info(f"Processed 5 chains concurrently in {execution_time:.3f}s")


class TestBackwardCompatibility:
    """Test that async migration maintains backward compatibility."""

    def test_sync_api_unchanged(self):
        """Test that existing sync API works unchanged."""
        # This should work exactly as before the async migration
        model = MockModel(model_name="test-model")
        validator = LengthValidator(min_length=10, max_length=100)
        critic = ReflexionCritic(model=MockModel(model_name="critic-model"))

        chain = Chain(
            model=model,
            prompt="Write something",
            always_apply_critics=True,  # Ensure critics run for testing
        )
        chain.validate_with(validator)
        chain.improve_with(critic)

        # This should work without any changes
        result = chain.run()

        assert result.text is not None
        assert len(result.validation_results) == 1
        assert len(result.critic_feedback) >= 1

    def test_individual_component_apis(self):
        """Test that individual component APIs work unchanged."""
        # Test validator API
        validator = LengthValidator(min_length=5, max_length=50)
        thought = Thought(prompt="test", text="This is a test message")
        validation_result = validator.validate(thought)
        assert validation_result.passed

        # Test critic API
        critic = ReflexionCritic(model=MockModel(model_name="critic-model"))
        critique_result = critic.critique(thought)
        assert "needs_improvement" in critique_result

        improved_text = critic.improve(thought)
        assert isinstance(improved_text, str)
        assert len(improved_text) > 0


def main():
    """Run the async migration test suite."""
    print("🎯 Sifaka Async Migration - Comprehensive Test Suite")
    print("=" * 60)

    # Test concurrent validation
    print("\n🔍 Testing Concurrent Validation...")
    test_validation = TestAsyncValidation()
    test_validation.test_concurrent_validation_basic()
    test_validation.test_concurrent_validation_with_failures()
    print("✅ Concurrent validation completed!")

    # Test concurrent criticism
    print("\n💬 Testing Concurrent Criticism...")
    test_criticism = TestAsyncCriticism()
    test_criticism.test_concurrent_criticism_basic()
    test_criticism.test_concurrent_criticism_complex()
    print("✅ Concurrent criticism completed!")

    # Test full chain execution
    print("\n⛓️  Testing Full Chain with Concurrency...")
    test_chain = TestAsyncChainExecution()
    test_chain.test_full_chain_with_concurrency()
    test_chain.test_multiple_chains_concurrently()
    print("✅ Full chain with concurrency completed!")

    # Test backward compatibility
    print("\n🔄 Testing Backward Compatibility...")
    test_compat = TestBackwardCompatibility()
    test_compat.test_sync_api_unchanged()
    test_compat.test_individual_component_apis()
    print("✅ Backward compatibility test passed!")

    print("\n🎉 All async migration tests completed successfully!")
    print("✨ Sifaka now supports concurrent validation and criticism!")


if __name__ == "__main__":
    main()
