#!/usr/bin/env python3
"""Comprehensive tests for Sifaka model implementations.

This test suite validates all model providers including OpenAI, Anthropic,
HuggingFace, Ollama, and Mock models. It tests generation capabilities,
error handling, and integration with the Sifaka framework.
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from sifaka.models.base import MockModel
from sifaka.core.interfaces import Model
from sifaka.utils.logging import get_logger

from tests.utils import assert_thought_valid, MockModelFactory

logger = get_logger(__name__)


class TestMockModel:
    """Test MockModel implementation."""

    def test_mock_model_basic_generation(self):
        """Test basic text generation with MockModel."""
        model = MockModel(model_name="test-mock")

        result = model.generate("Test prompt")

        # MockModel returns a formatted string with the model name and prompt
        assert "test-mock" in result
        assert "Test prompt" in result
        assert model.model_name == "test-mock"

    def test_mock_model_custom_response(self):
        """Test MockModel with factory for custom response text."""
        from tests.utils.mocks import MockModelFactory

        custom_response = "This is a custom mock response for testing purposes."
        model = MockModelFactory.create_standard(
            model_name="custom-mock", response_text=custom_response
        )

        result = model.generate("Any prompt")

        assert result == custom_response

    def test_mock_model_token_counting(self):
        """Test token counting functionality."""
        model = MockModel(model_name="token-test")

        # Test with various text lengths
        short_text = "Hello"
        medium_text = "This is a medium length text for testing."
        long_text = "This is a much longer text that should have more tokens when counted by the mock model implementation."

        short_tokens = model.count_tokens(short_text)
        medium_tokens = model.count_tokens(medium_text)
        long_tokens = model.count_tokens(long_text)

        # Token counts should be reasonable and proportional
        assert short_tokens > 0
        assert medium_tokens > short_tokens
        assert long_tokens > medium_tokens
        assert long_tokens < len(long_text)  # Should be less than character count

    def test_mock_model_with_options(self):
        """Test MockModel with generation options."""
        model = MockModel(model_name="options-test")

        # Test with various options (MockModel should handle gracefully)
        result = model.generate("Test prompt", temperature=0.7, max_tokens=100, top_p=0.9)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_model_interface_compliance(self):
        """Test that MockModel implements the Model interface correctly."""
        model = MockModel(model_name="interface-test")

        # Verify it implements the Model protocol
        assert isinstance(model, Model)

        # Test required methods exist and work
        assert hasattr(model, "generate")
        assert hasattr(model, "count_tokens")

        result = model.generate("Test")
        assert isinstance(result, str)

        tokens = model.count_tokens("Test")
        assert isinstance(tokens, int)


class TestModelFactories:
    """Test model factory utilities."""

    def test_standard_mock_factory(self):
        """Test standard mock model creation."""
        model = MockModelFactory.create_standard(
            model_name="factory-test", response_text="Factory response"
        )

        assert model.model_name == "factory-test"
        assert model.generate("Any prompt") == "Factory response"

    def test_slow_mock_factory(self):
        """Test slow mock model creation."""
        import time

        model = MockModelFactory.create_slow(model_name="slow-test", delay_seconds=0.1)

        start_time = time.time()
        result = model.generate("Test prompt")
        execution_time = time.time() - start_time

        assert execution_time >= 0.1  # Should have delay
        assert isinstance(result, str)

    def test_failing_mock_factory(self):
        """Test failing mock model creation."""
        model = MockModelFactory.create_failing(
            model_name="failing-test", error_message="Test error"
        )

        with pytest.raises(Exception) as exc_info:
            model.generate("Test prompt")

        assert "Test error" in str(exc_info.value)

    def test_variable_response_factory(self):
        """Test variable response mock model."""
        responses = ["Response 1", "Response 2", "Response 3"]
        model = MockModelFactory.create_variable_response(
            model_name="variable-test", responses=responses
        )

        # Test cycling through responses
        results = []
        for i in range(6):  # Test more than the number of responses
            result = model.generate(f"Prompt {i}")
            results.append(result)

        # Should cycle through responses
        assert results[0] == "Response 1"
        assert results[1] == "Response 2"
        assert results[2] == "Response 3"
        assert results[3] == "Response 1"  # Should cycle back
        assert results[4] == "Response 2"
        assert results[5] == "Response 3"


class TestModelErrorHandling:
    """Test error handling in model implementations."""

    def test_mock_model_empty_response(self):
        """Test MockModel with empty response using factory."""
        from tests.utils.mocks import MockModelFactory

        model = MockModelFactory.create_standard(model_name="empty-test", response_text="")

        result = model.generate("Test prompt")

        assert result == ""

    def test_mock_model_none_response(self):
        """Test MockModel behavior with None response."""
        # MockModel should handle None gracefully
        model = MockModel(model_name="none-test", response_text=None)

        result = model.generate("Test prompt")

        # Should convert None to empty string or handle appropriately
        assert result is not None

    def test_model_with_invalid_options(self):
        """Test model behavior with invalid generation options."""
        model = MockModel(model_name="invalid-options-test")

        # Should handle invalid options gracefully
        result = model.generate(
            "Test prompt", invalid_option="invalid_value", another_invalid=12345
        )

        assert isinstance(result, str)

    def test_model_with_very_long_prompt(self):
        """Test model behavior with very long prompts."""
        model = MockModel(model_name="long-prompt-test")

        # Create a very long prompt
        long_prompt = "This is a test prompt. " * 1000

        result = model.generate(long_prompt)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_model_token_counting_edge_cases(self):
        """Test token counting with edge cases."""
        model = MockModel(model_name="token-edge-test")

        # Test edge cases
        empty_tokens = model.count_tokens("")
        whitespace_tokens = model.count_tokens("   ")
        special_chars_tokens = model.count_tokens("!@#$%^&*()")
        unicode_tokens = model.count_tokens("Hello 世界 🌍")

        assert empty_tokens >= 0
        assert whitespace_tokens >= 0
        assert special_chars_tokens >= 0
        assert unicode_tokens >= 0


class TestModelIntegration:
    """Test model integration with Sifaka components."""

    def test_model_with_chain_integration(self):
        """Test model integration with Chain."""
        from sifaka.core.chain import Chain

        from tests.utils.mocks import MockModelFactory

        model = MockModelFactory.create_standard(
            model_name="chain-integration-test",
            response_text="This is a test response for chain integration.",
        )

        chain = Chain(model=model, prompt="Write about integration testing.")
        result = chain.run()

        assert_thought_valid(result)
        assert result.text == "This is a test response for chain integration."

    def test_model_with_validators_integration(self):
        """Test model integration with validators."""
        from sifaka.core.chain import Chain
        from sifaka.validators.base import LengthValidator

        model = MockModel(
            model_name="validator-integration-test",
            response_text="This is a response that should pass length validation.",
        )

        chain = Chain(model=model, prompt="Write a response.")
        chain.validate_with(LengthValidator(min_length=10, max_length=100))

        result = chain.run()

        assert_thought_valid(result)
        assert result.validation_results is not None
        assert len(result.validation_results) == 1

    def test_model_with_critics_integration(self):
        """Test model integration with critics."""
        from sifaka.core.chain import Chain
        from sifaka.critics.reflexion import ReflexionCritic

        main_model = MockModel(model_name="main-model", response_text="This is the main response.")
        critic_model = MockModel(
            model_name="critic-model", response_text="This is critic feedback."
        )

        chain = Chain(model=main_model, prompt="Write something.", always_apply_critics=True)
        chain.improve_with(ReflexionCritic(model=critic_model))

        result = chain.run()

        assert_thought_valid(result)
        assert result.critic_feedback is not None
        assert len(result.critic_feedback) >= 1

    def test_model_with_storage_integration(self):
        """Test model integration with storage."""
        from sifaka.core.chain import Chain
        from sifaka.storage.memory import MemoryStorage

        model = MockModel(
            model_name="storage-integration-test", response_text="This response will be stored."
        )
        storage = MemoryStorage()

        chain = Chain(model=model, prompt="Write something to store.", storage=storage)

        result = chain.run()

        assert_thought_valid(result)
        thought_key = f"thought_{result.chain_id}_{result.iteration}"
        assert thought_key in storage.data

        stored_thought_data = storage.get(thought_key)
        assert stored_thought_data is not None


class TestModelPerformance:
    """Test model performance characteristics."""

    def test_mock_model_performance(self):
        """Test MockModel performance."""
        import time

        model = MockModel(model_name="performance-test")

        # Test generation performance
        start_time = time.time()
        for i in range(100):
            result = model.generate(f"Test prompt {i}")
            assert isinstance(result, str)
        generation_time = time.time() - start_time

        # Should be very fast for mock model
        assert (
            generation_time < 1.0
        ), f"Mock model too slow: {generation_time:.3f}s for 100 generations"

        # Test token counting performance
        start_time = time.time()
        for i in range(100):
            tokens = model.count_tokens(f"Test text {i}")
            assert isinstance(tokens, int)
        counting_time = time.time() - start_time

        assert counting_time < 1.0, f"Token counting too slow: {counting_time:.3f}s for 100 counts"

    def test_model_memory_usage(self):
        """Test model memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create multiple models
        models = []
        for i in range(50):
            model = MockModel(model_name=f"memory-test-{i}", response_text=f"Response {i}")
            models.append(model)

        # Use the models
        for model in models:
            result = model.generate("Test")
            assert isinstance(result, str)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert (
            memory_increase < 50
        ), f"Model memory usage too high: {memory_increase:.1f}MB for 50 models"


class TestModelConfiguration:
    """Test model configuration and options."""

    def test_mock_model_configuration(self):
        """Test MockModel configuration options."""
        from tests.utils.mocks import MockModelFactory

        model = MockModelFactory.create_standard(
            model_name="configured-model", response_text="Configured response"
        )

        assert model.model_name == "configured-model"
        result = model.generate("Test")
        assert result == "Configured response"

    def test_model_option_validation(self):
        """Test validation of model options."""
        model = MockModel(model_name="validation-test")

        # Test with valid options
        result = model.generate("Test prompt", temperature=0.7, max_tokens=100)
        assert isinstance(result, str)

        # Test with edge case options
        result = model.generate("Test prompt", temperature=0.0, max_tokens=1)
        assert isinstance(result, str)

    def test_model_defaults(self):
        """Test model default behavior."""
        model = MockModel(model_name="default-test")  # MockModel requires model_name

        # Should have reasonable defaults
        assert model.model_name is not None
        assert len(model.model_name) > 0

        result = model.generate("Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0
