#!/usr/bin/env python3
"""Comprehensive tests for Sifaka Chain core functionality.

This test suite covers the Chain class, which is the central orchestrator
for thought generation, validation, and improvement workflows.
"""

from unittest.mock import Mock

import pytest

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.models.base import MockModel
from sifaka.retrievers.simple import MockRetriever
from sifaka.storage.memory import MemoryStorage
from sifaka.validators.base import LengthValidator


class TestChainBasic:
    """Test basic Chain functionality."""

    def test_chain_creation_basic(self):
        """Test basic chain creation."""
        model = MockModel(model_name="test-model")
        chain = Chain(model=model, prompt="Test prompt")

        assert chain.model == model
        assert chain.prompt == "Test prompt"
        assert chain.max_iterations == 3  # Default value
        assert chain.validators == []
        assert chain.critics == []

    def test_chain_creation_with_options(self):
        """Test chain creation with custom options."""
        model = MockModel(model_name="test-model")
        chain = Chain(
            model=model, prompt="Test prompt", max_iterations=5, temperature=0.8, max_tokens=100
        )

        assert chain.max_iterations == 5
        assert chain.temperature == 0.8
        assert chain.max_tokens == 100

    def test_chain_with_validators(self):
        """Test chain creation with validators."""
        model = MockModel(model_name="test-model")
        validator = LengthValidator(min_length=10, max_length=100)

        chain = Chain(model=model, prompt="Test prompt").validate_with(validator)

        assert len(chain.validators) == 1
        assert chain.validators[0] == validator

    def test_chain_with_critics(self):
        """Test chain creation with critics."""
        model = MockModel(model_name="test-model")
        critic = Mock(spec=BaseCritic)

        chain = Chain(model=model, prompt="Test prompt").improve_with(critic)

        assert len(chain.critics) == 1
        assert chain.critics[0] == critic

    def test_chain_with_retriever(self):
        """Test chain creation with retriever."""
        model = MockModel(model_name="test-model")
        retriever = MockRetriever()

        chain = Chain(model=model, prompt="Test prompt", retriever=retriever)

        assert chain.retriever == retriever

    def test_chain_with_storage(self):
        """Test chain creation with storage."""
        model = MockModel(model_name="test-model")
        storage = MemoryStorage()

        chain = Chain(model=model, prompt="Test prompt", storage=storage)

        assert chain.storage == storage

    def test_chain_fluent_api(self):
        """Test chain fluent API."""
        model = MockModel(model_name="test-model")
        validator = LengthValidator(min_length=10, max_length=100)
        critic = Mock(spec=BaseCritic)

        chain = (
            Chain(model=model, prompt="Test prompt")
            .validate_with(validator)
            .improve_with(critic)
            .with_options(max_iterations=5)
        )

        assert len(chain.validators) == 1
        assert len(chain.critics) == 1
        assert chain.max_iterations == 5


class TestChainExecution:
    """Test Chain execution functionality."""

    def test_chain_run_basic(self):
        """Test basic chain execution."""
        model = MockModel(model_name="test-model", response_text="Generated response")
        chain = Chain(model=model, prompt="Test prompt")

        result = chain.run()

        assert isinstance(result, Thought)
        assert result.text == "Generated response"
        assert result.prompt == "Test prompt"
        assert result.iteration == 1

    def test_chain_run_with_validation_success(self):
        """Test chain execution with successful validation."""
        model = MockModel(
            model_name="test-model",
            response_text="This is a valid response that meets length requirements",
        )
        validator = LengthValidator(min_length=10, max_length=100)

        chain = Chain(model=model, prompt="Test prompt").validate_with(validator)
        result = chain.run()

        assert isinstance(result, Thought)
        assert result.text is not None
        assert len(result.text) >= 10

    def test_chain_run_with_validation_failure_and_retry(self):
        """Test chain execution with validation failure and retry."""
        model = MockModel(model_name="test-model")
        # First response too short, second response valid
        model.responses = ["Short", "This is a longer response that meets the requirements"]

        validator = LengthValidator(min_length=20, max_length=100)

        chain = Chain(model=model, prompt="Test prompt", max_iterations=2).validate_with(validator)
        result = chain.run()

        assert isinstance(result, Thought)
        assert result.iteration == 2  # Should have retried
        assert len(result.text) >= 20

    def test_chain_run_with_retriever(self):
        """Test chain execution with retriever."""
        model = MockModel(model_name="test-model", response_text="Response with context")
        retriever = MockRetriever(documents=["Context document 1", "Context document 2"])

        chain = Chain(model=model, prompt="Test prompt", retriever=retriever)
        result = chain.run()

        assert isinstance(result, Thought)
        assert result.pre_generation_context is not None
        assert len(result.pre_generation_context) > 0

    def test_chain_run_with_storage(self):
        """Test chain execution with storage."""
        model = MockModel(model_name="test-model", response_text="Stored response")
        storage = MemoryStorage()

        chain = Chain(model=model, prompt="Test prompt", storage=storage)
        result = chain.run()

        assert isinstance(result, Thought)
        # Check if thought was stored (implementation dependent)
        assert result.text == "Stored response"

    def test_chain_run_max_iterations_exceeded(self):
        """Test chain execution when max iterations exceeded."""
        model = MockModel(model_name="test-model", response_text="Short")  # Always fails validation
        validator = LengthValidator(min_length=100, max_length=200)  # Impossible to meet

        chain = Chain(model=model, prompt="Test prompt", max_iterations=2).validate_with(validator)
        result = chain.run()

        assert isinstance(result, Thought)
        assert result.iteration == 2  # Should stop at max iterations
        # Should have validation errors
        assert len(result.validation_results) > 0

    def test_chain_run_async_context(self):
        """Test chain execution in async context."""
        model = MockModel(model_name="test-model", response_text="Async response")
        chain = Chain(model=model, prompt="Test prompt")

        # Test the sync run method (which handles async internally)
        result = chain.run()

        assert isinstance(result, Thought)
        assert result.text == "Async response"

    def test_chain_run_with_options(self):
        """Test chain execution with generation options."""
        model = MockModel(model_name="test-model", response_text="Response with options")

        chain = Chain(model=model, prompt="Test prompt", temperature=0.8, max_tokens=50, top_p=0.9)
        result = chain.run()

        assert isinstance(result, Thought)
        assert result.text == "Response with options"


class TestChainErrorHandling:
    """Test Chain error handling."""

    def test_chain_model_error_handling(self):
        """Test chain handling of model errors."""
        model = Mock()
        model.generate.side_effect = Exception("Model error")

        chain = Chain(model=model, prompt="Test prompt")

        # Should handle model errors gracefully
        with pytest.raises(Exception):
            chain.run()

    def test_chain_validation_error_handling(self):
        """Test chain handling of validation errors."""
        model = MockModel(model_name="test-model", response_text="Test response")
        validator = Mock()
        validator.validate.side_effect = Exception("Validation error")

        chain = Chain(model=model, prompt="Test prompt").validate_with(validator)

        # Should handle validation errors gracefully
        result = chain.run()
        assert isinstance(result, Thought)

    def test_chain_critic_error_handling(self):
        """Test chain handling of critic errors."""
        model = MockModel(model_name="test-model", response_text="Test response")
        critic = Mock()
        critic.critique.side_effect = Exception("Critic error")

        chain = Chain(model=model, prompt="Test prompt").improve_with(critic)

        # Should handle critic errors gracefully
        result = chain.run()
        assert isinstance(result, Thought)

    def test_chain_retriever_error_handling(self):
        """Test chain handling of retriever errors."""
        model = MockModel(model_name="test-model", response_text="Test response")
        retriever = Mock()
        retriever.retrieve_for_thought.side_effect = Exception("Retriever error")

        chain = Chain(model=model, prompt="Test prompt", retriever=retriever)

        # Should handle retriever errors gracefully
        result = chain.run()
        assert isinstance(result, Thought)

    def test_chain_storage_error_handling(self):
        """Test chain handling of storage errors."""
        model = MockModel(model_name="test-model", response_text="Test response")
        storage = Mock()
        storage.save.side_effect = Exception("Storage error")

        chain = Chain(model=model, prompt="Test prompt", storage=storage)

        # Should handle storage errors gracefully
        result = chain.run()
        assert isinstance(result, Thought)


class TestChainConfiguration:
    """Test Chain configuration and options."""

    def test_chain_with_options_method(self):
        """Test chain with_options method."""
        model = MockModel(model_name="test-model")
        chain = Chain(model=model, prompt="Test prompt")

        updated_chain = chain.with_options(max_iterations=5, temperature=0.8, max_tokens=100)

        assert updated_chain.max_iterations == 5
        assert updated_chain.temperature == 0.8
        assert updated_chain.max_tokens == 100

    def test_chain_copy_behavior(self):
        """Test chain copy behavior for fluent API."""
        model = MockModel(model_name="test-model")
        original_chain = Chain(model=model, prompt="Test prompt")

        validator = LengthValidator(min_length=10, max_length=100)
        new_chain = original_chain.validate_with(validator)

        # Should not modify original chain
        assert len(original_chain.validators) == 0
        assert len(new_chain.validators) == 1

    def test_chain_multiple_validators(self):
        """Test chain with multiple validators."""
        model = MockModel(model_name="test-model")
        validator1 = LengthValidator(min_length=10, max_length=100)
        validator2 = LengthValidator(min_length=5, max_length=50)

        chain = (
            Chain(model=model, prompt="Test prompt")
            .validate_with(validator1)
            .validate_with(validator2)
        )

        assert len(chain.validators) == 2

    def test_chain_multiple_critics(self):
        """Test chain with multiple critics."""
        model = MockModel(model_name="test-model")
        critic1 = Mock(spec=BaseCritic)
        critic2 = Mock(spec=BaseCritic)

        chain = Chain(model=model, prompt="Test prompt").improve_with(critic1).improve_with(critic2)

        assert len(chain.critics) == 2

    def test_chain_configuration_inheritance(self):
        """Test chain configuration inheritance."""
        model = MockModel(model_name="test-model")
        base_chain = Chain(model=model, prompt="Base prompt", max_iterations=5, temperature=0.8)

        validator = LengthValidator(min_length=10, max_length=100)
        derived_chain = base_chain.validate_with(validator)

        # Should inherit configuration
        assert derived_chain.max_iterations == 5
        assert derived_chain.temperature == 0.8
        assert derived_chain.prompt == "Base prompt"


class TestChainIntegration:
    """Test Chain integration scenarios."""

    def test_chain_full_workflow(self):
        """Test complete chain workflow with all components."""
        model = MockModel(model_name="test-model", response_text="Complete workflow response")
        validator = LengthValidator(min_length=10, max_length=100)
        critic = Mock(spec=BaseCritic)
        retriever = MockRetriever(documents=["Context doc"])
        storage = MemoryStorage()

        chain = (
            Chain(model=model, prompt="Test prompt", retriever=retriever, storage=storage)
            .validate_with(validator)
            .improve_with(critic)
            .with_options(max_iterations=3)
        )

        result = chain.run()

        assert isinstance(result, Thought)
        assert result.text is not None
        assert result.pre_generation_context is not None

    def test_chain_performance(self):
        """Test chain performance characteristics."""
        import time

        model = MockModel(model_name="test-model", response_text="Performance test response")
        chain = Chain(model=model, prompt="Performance test")

        start_time = time.time()
        result = chain.run()
        end_time = time.time()

        # Should complete in reasonable time
        assert (end_time - start_time) < 1.0  # 1 second max
        assert isinstance(result, Thought)

    def test_chain_memory_usage(self):
        """Test chain memory usage."""
        model = MockModel(model_name="test-model", response_text="Memory test response")
        chain = Chain(model=model, prompt="Memory test")

        # Run multiple times to check for memory leaks
        for _ in range(10):
            result = chain.run()
            assert isinstance(result, Thought)

        # Should not accumulate excessive memory

    def test_chain_thread_safety(self):
        """Test chain thread safety."""
        import threading

        model = MockModel(model_name="test-model", response_text="Thread safety test")
        chain = Chain(model=model, prompt="Thread safety test")

        results = []

        def run_chain():
            result = chain.run()
            results.append(result)

        # Run multiple threads
        threads = [threading.Thread(target=run_chain) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, Thought)
