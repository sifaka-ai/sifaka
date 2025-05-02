"""
Tests for the core critic implementation.
"""

import unittest
from unittest.mock import Mock

from sifaka.critics.models import CriticConfig
from sifaka.critics.core import CriticCore
from sifaka.critics.managers.memory import MemoryManager
from sifaka.critics.managers.prompt import DefaultPromptManager
from sifaka.critics.managers.response import ResponseParser
from sifaka.critics.services.critique import CritiqueService


class MockLanguageModel:
    """Mock language model for testing."""

    def __init__(self):
        """Initialize the mock language model."""
        self.invoke = Mock(
            return_value={
                "score": 0.8,
                "feedback": "Good text",
                "issues": ["Minor issue"],
                "suggestions": ["Minor suggestion"],
            }
        )

    def invoke(self, prompt: str) -> dict:
        """Mock implementation of invoke."""
        return {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Minor issue"],
            "suggestions": ["Minor suggestion"],
        }


class TestCriticCore(unittest.TestCase):
    """Tests for the CriticCore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CriticConfig(
            name="Test Critic",
            description="Test critic for testing",
            min_confidence=0.7,
            max_attempts=3,
        )
        self.model = MockLanguageModel()
        self.critic = CriticCore(
            config=self.config,
            llm_provider=self.model,
        )

    def test_initialization(self):
        """Test that the critic initializes correctly."""
        self.assertEqual(self.critic.config, self.config)

    def test_validate(self):
        """Test that validate works correctly."""
        self.model.invoke = Mock(return_value={"valid": True})
        result = self.critic.validate("Test text")
        self.assertTrue(result)

    def test_critique(self):
        """Test that critique works correctly."""
        self.model.invoke = Mock(
            return_value={
                "score": 0.8,
                "feedback": "Good text",
                "issues": ["Minor issue"],
                "suggestions": ["Minor suggestion"],
            }
        )
        result = self.critic.critique("Test text")
        self.assertEqual(result.score, 0.8)
        self.assertEqual(result.feedback, "Good text")
        self.assertEqual(result.issues, ["Minor issue"])
        self.assertEqual(result.suggestions, ["Minor suggestion"])

    def test_improve(self):
        """Test that improve works correctly."""
        self.model.invoke = Mock(return_value={"improved_text": "Improved text"})
        result = self.critic.improve(
            "Test text", [{"rule_name": "Test Rule", "message": "Test message"}]
        )
        self.assertEqual(result, "Improved text")

    def test_improve_with_feedback(self):
        """Test that improve_with_feedback works correctly."""
        self.model.invoke = Mock(return_value={"improved_text": "Improved text"})
        result = self.critic.improve_with_feedback("Test text", "Test feedback")
        self.assertEqual(result, "Improved text")


class TestCritiqueService(unittest.TestCase):
    """Tests for the CritiqueService class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CriticConfig(
            name="Test Critic",
            description="Test critic for testing",
            min_confidence=0.7,
            max_attempts=3,
        )
        self.model = MockLanguageModel()
        self.prompt_manager = DefaultPromptManager(self.config)
        self.response_parser = ResponseParser()
        self.memory_manager = MemoryManager(buffer_size=5)
        self.service = CritiqueService(
            llm_provider=self.model,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            memory_manager=self.memory_manager,
        )

    def test_validate(self):
        """Test that validate works correctly."""
        self.model.invoke = Mock(return_value={"valid": True})
        result = self.service.validate("Test text")
        self.assertTrue(result)

    def test_critique(self):
        """Test that critique works correctly."""
        self.model.invoke = Mock(
            return_value={
                "score": 0.8,
                "feedback": "Good text",
                "issues": ["Minor issue"],
                "suggestions": ["Minor suggestion"],
            }
        )
        result = self.service.critique("Test text")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")
        self.assertEqual(result["issues"], ["Minor issue"])
        self.assertEqual(result["suggestions"], ["Minor suggestion"])

    def test_improve(self):
        """Test that improve works correctly."""
        self.model.invoke = Mock(return_value={"improved_text": "Improved text"})
        result = self.service.improve("Test text", "Test feedback")
        self.assertEqual(result, "Improved text")

    def test_memory_manager(self):
        """Test that memory manager works correctly."""
        self.model.invoke = Mock(return_value={"reflection": "Test reflection"})
        self.service._generate_reflection("Test text", "Test feedback", "Improved text")
        self.assertEqual(self.memory_manager.memory_size, 1)
        self.assertEqual(self.memory_manager.get_memory()[0], "Test reflection")


if __name__ == "__main__":
    unittest.main()
