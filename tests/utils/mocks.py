"""Mock factories and utilities for Sifaka testing.

This module provides factory classes and utilities for creating
mock objects with configurable behavior for testing.
"""

import time
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, MagicMock

from sifaka.core.thought import Thought, ValidationResult, CriticFeedback
from sifaka.models.base import MockModel
from sifaka.storage.memory import MemoryStorage


class MockModelFactory:
    """Factory for creating MockModel instances with various behaviors."""

    @staticmethod
    def create_standard(
        model_name: str = "test-model", response_text: str = "Mock generated response"
    ) -> MockModel:
        """Create a standard mock model."""

        class CustomResponseMockModel(MockModel):
            def __init__(self, model_name: str, response_text: str = "Mock response", **kwargs):
                super().__init__(model_name, **kwargs)
                self.response_text = response_text

            def generate(self, prompt: str, **options: Any) -> str:
                return self.response_text

            def generate_with_thought(self, thought: Any, **options: Any) -> tuple[str, str]:
                # Use mixin to build contextualized prompt
                full_prompt = self._build_contextualized_prompt(thought, max_docs=5)
                return self.response_text, full_prompt

        return CustomResponseMockModel(model_name=model_name, response_text=response_text)

    @staticmethod
    def create_slow(
        model_name: str = "slow-model",
        response_text: str = "Slow mock response",
        delay_seconds: float = 1.0,
    ) -> MockModel:
        """Create a mock model that simulates slow responses."""

        class SlowMockModel(MockModel):
            def generate(self, prompt: str, **options: Any) -> str:
                time.sleep(delay_seconds)
                return super().generate(prompt, **options)

        return SlowMockModel(model_name=model_name, response_text=response_text)

    @staticmethod
    def create_failing(
        model_name: str = "failing-model", error_message: str = "Mock model error"
    ) -> MockModel:
        """Create a mock model that always fails."""

        class FailingMockModel(MockModel):
            def generate(self, prompt: str, **options: Any) -> str:
                raise Exception(error_message)

        return FailingMockModel(model_name=model_name)

    @staticmethod
    def create_variable_response(
        model_name: str = "variable-model", responses: List[str] = None
    ) -> MockModel:
        """Create a mock model that cycles through different responses."""
        if responses is None:
            responses = ["Response 1", "Response 2", "Response 3"]

        class VariableResponseModel(MockModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._responses = responses
                self._call_count = 0

            def generate(self, prompt: str, **options: Any) -> str:
                response = self._responses[self._call_count % len(self._responses)]
                self._call_count += 1
                return response

            def generate_with_thought(self, thought: Any, **options: Any) -> tuple[str, str]:
                # Use mixin to build contextualized prompt
                full_prompt = self._build_contextualized_prompt(thought, max_docs=5)
                # Get the variable response
                response = self._responses[self._call_count % len(self._responses)]
                self._call_count += 1
                return response, full_prompt

        return VariableResponseModel(model_name=model_name)


class MockStorageFactory:
    """Factory for creating mock storage instances."""

    @staticmethod
    def create_memory() -> MemoryStorage:
        """Create a standard memory storage."""
        return MemoryStorage()

    @staticmethod
    def create_failing() -> Mock:
        """Create a mock storage that fails operations."""
        storage = Mock()
        storage.set.side_effect = Exception("Storage set failed")
        storage.get.side_effect = Exception("Storage get failed")
        storage._set_async.side_effect = Exception("Storage set async failed")
        storage._get_async.side_effect = Exception("Storage get async failed")
        storage.search.side_effect = Exception("Storage search failed")
        storage.clear.side_effect = Exception("Storage clear failed")
        return storage

    @staticmethod
    def create_slow(delay_seconds: float = 0.5) -> Mock:
        """Create a mock storage with slow operations."""
        storage = Mock()

        def slow_set(*args, **kwargs):
            time.sleep(delay_seconds)
            return None

        def slow_get(*args, **kwargs):
            time.sleep(delay_seconds)
            return None

        storage.set.side_effect = slow_set
        storage.get.side_effect = slow_get
        storage._set_async.side_effect = slow_set
        storage._get_async.side_effect = slow_get
        storage.search.return_value = []
        storage.clear.return_value = None
        return storage


class MockRetrieverFactory:
    """Factory for creating mock retriever instances."""

    @staticmethod
    def create_standard(documents: List[str] = None) -> Mock:
        """Create a standard mock retriever."""
        if documents is None:
            documents = [
                "Document 1: Relevant information about the topic.",
                "Document 2: Additional context and details.",
                "Document 3: Supporting evidence and examples.",
            ]

        retriever = Mock()
        retriever.retrieve.return_value = documents
        return retriever

    @staticmethod
    def create_empty() -> Mock:
        """Create a mock retriever that returns no documents."""
        retriever = Mock()
        retriever.retrieve.return_value = []
        return retriever

    @staticmethod
    def create_failing() -> Mock:
        """Create a mock retriever that fails."""
        retriever = Mock()
        retriever.retrieve.side_effect = Exception("Retriever failed")
        return retriever


class MockValidatorFactory:
    """Factory for creating mock validators."""

    @staticmethod
    def create_passing(name: str = "passing-validator") -> Mock:
        """Create a validator that always passes."""
        validator = Mock()
        validator.validate.return_value = ValidationResult(passed=True, message="Validation passed")
        return validator

    @staticmethod
    def create_failing(
        name: str = "failing-validator", error_message: str = "Validation failed"
    ) -> Mock:
        """Create a validator that always fails."""
        validator = Mock()
        validator.validate.return_value = ValidationResult(passed=False, message=error_message)
        return validator

    @staticmethod
    def create_slow(name: str = "slow-validator", delay_seconds: float = 0.5) -> Mock:
        """Create a validator with slow validation."""
        validator = Mock()

        def slow_validate(*args, **kwargs):
            time.sleep(delay_seconds)
            return ValidationResult(passed=True, message="Slow validation passed")

        validator.validate.side_effect = slow_validate
        return validator


class MockCriticFactory:
    """Factory for creating mock critics."""

    @staticmethod
    def create_standard(name: str = "standard-critic") -> Mock:
        """Create a standard mock critic."""
        critic = Mock()
        critic.critique.return_value = {
            "needs_improvement": True,
            "feedback": "The text could be improved with more detail.",
            "score": 0.7,
        }
        critic.improve.return_value = "Improved text with more detail and clarity."
        return critic

    @staticmethod
    def create_satisfied(name: str = "satisfied-critic") -> Mock:
        """Create a critic that's always satisfied."""
        critic = Mock()
        critic.critique.return_value = {
            "needs_improvement": False,
            "feedback": "The text is excellent as is.",
            "score": 0.95,
        }
        critic.improve.return_value = "Text is already excellent."
        return critic

    @staticmethod
    def create_failing(name: str = "failing-critic", error_message: str = "Critic failed") -> Mock:
        """Create a critic that fails."""
        critic = Mock()
        critic.critique.side_effect = Exception(error_message)
        critic.improve.side_effect = Exception(error_message)
        return critic


# Utility functions for creating test scenarios
def create_test_scenario(
    name: str,
    model: Optional[Any] = None,
    validators: Optional[List[Any]] = None,
    critics: Optional[List[Any]] = None,
    storage: Optional[Any] = None,
    expected_success: bool = True,
) -> Dict[str, Any]:
    """Create a test scenario configuration."""
    return {
        "name": name,
        "model": model or MockModelFactory.create_standard(),
        "validators": validators or [],
        "critics": critics or [],
        "storage": storage or MockStorageFactory.create_memory(),
        "expected_success": expected_success,
    }


def create_performance_scenario(
    name: str,
    complexity: str = "medium",
    expected_max_time: float = 5.0,
    expected_max_memory: float = 100.0,
) -> Dict[str, Any]:
    """Create a performance test scenario."""
    prompts = {
        "simple": "Write a sentence.",
        "medium": "Write a paragraph about AI.",
        "complex": "Write a detailed essay about machine learning." * 5,
    }

    return {
        "name": name,
        "prompt": prompts.get(complexity, prompts["medium"]),
        "expected_max_time": expected_max_time,
        "expected_max_memory": expected_max_memory,
        "complexity": complexity,
    }
