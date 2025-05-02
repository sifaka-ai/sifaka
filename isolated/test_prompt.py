"""
Tests for the prompt critic module.

This module contains tests for the prompt critic components defined in sifaka/critics/prompt.py,
including DefaultPromptFactory for creating prompts and other functionality.
"""

import unittest
from unittest.mock import MagicMock


class TestPromptCriticConfig(unittest.TestCase):
    """Tests for the PromptCriticConfig."""

    def test_valid_config(self):
        """Test creating a valid config."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCriticConfig

        config = PromptCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="Test system prompt",
            temperature=0.5,
            max_tokens=100,
        )
        self.assertEqual(config.name, "test_critic")
        self.assertEqual(config.description, "Test critic")
        self.assertEqual(config.system_prompt, "Test system prompt")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 100)


class TestDefaultPromptFactory(unittest.TestCase):
    """Tests for the DefaultPromptFactory prompt creation methods."""

    def setUp(self):
        """Set up test environment."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import DefaultPromptFactory
        self.factory = DefaultPromptFactory()

    def test_create_validation_prompt(self):
        """Test creating a validation prompt."""
        prompt = self.factory.create_validation_prompt("Text to validate")
        self.assertIn("TEXT TO VALIDATE:", prompt)
        self.assertIn("Text to validate", prompt)
        self.assertIn("VALID:", prompt)
        self.assertIn("REASON:", prompt)

    def test_create_critique_prompt(self):
        """Test creating a critique prompt."""
        prompt = self.factory.create_critique_prompt("Text to critique")
        self.assertIn("TEXT TO CRITIQUE:", prompt)
        self.assertIn("Text to critique", prompt)
        self.assertIn("SCORE:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("ISSUES:", prompt)
        self.assertIn("SUGGESTIONS:", prompt)

    def test_create_improvement_prompt(self):
        """Test creating an improvement prompt."""
        prompt = self.factory.create_improvement_prompt("Text to improve", "Feedback")
        self.assertIn("TEXT TO IMPROVE:", prompt)
        self.assertIn("Text to improve", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("Feedback", prompt)
        self.assertIn("IMPROVED_TEXT:", prompt)


if __name__ == "__main__":
    unittest.main()