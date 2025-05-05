"""
Tests for the constitutional critic.

This module contains tests for the ConstitutionalCritic class and related functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from sifaka.critics.constitutional import (
    ConstitutionalCritic,
    ConstitutionalCriticConfig,
    create_constitutional_critic,
)


class TestConstitutionalCritic(unittest.TestCase):
    """Tests for the ConstitutionalCritic class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock language model provider
        self.mock_llm_provider = MagicMock()

        # Set up default principles
        self.principles = ["Do not provide harmful content.", "Explain reasoning clearly."]

        # Create a default configuration
        self.config = ConstitutionalCriticConfig(
            name="test_critic", description="A test critic", principles=self.principles
        )

        # Create a critic with the mock provider
        self.critic = ConstitutionalCritic(config=self.config, llm_provider=self.mock_llm_provider)

    def test_initialization(self):
        """Test that the critic initializes correctly."""
        # Check that the critic has the correct attributes
        self.assertEqual(self.critic._state.cache.get("principles"), self.principles)
        self.assertEqual(self.critic._state.model, self.mock_llm_provider)
        self.assertTrue(self.critic._state.initialized)

    def test_format_principles(self):
        """Test that principles are formatted correctly."""
        formatted = self.critic._format_principles()
        expected = "- Do not provide harmful content.\n- Explain reasoning clearly."
        self.assertEqual(formatted, expected)

    def test_get_task_from_metadata(self):
        """Test that task is extracted correctly from metadata."""
        metadata = {"task": "Test task"}
        task = self.critic._get_task_from_metadata(metadata)
        self.assertEqual(task, "Test task")

        # Test with missing task
        with self.assertRaises(ValueError):
            self.critic._get_task_from_metadata({})

        # Test with None metadata
        with self.assertRaises(ValueError):
            self.critic._get_task_from_metadata(None)

    def test_critique_with_no_issues(self):
        """Test critique when no issues are found."""
        # Configure mock to return a response with no issues
        self.mock_llm_provider.generate.return_value = (
            "No issues found. The response aligns with all principles."
        )

        # Call critique
        result = self.critic.critique("Test response", {"task": "Test task"})

        # Check result
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["feedback"], "Response aligns with all principles.")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

        # Verify mock was called correctly
        self.mock_llm_provider.generate.assert_called_once()

    def test_critique_with_issues(self):
        """Test critique when issues are found."""
        # Configure mock to return a response with issues
        self.mock_llm_provider.generate.return_value = (
            "The response has issues:\n"
            "- Does not explain reasoning clearly\n"
            "- Could provide more context"
        )

        # Call critique
        result = self.critic.critique("Test response", {"task": "Test task"})

        # Check result
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(
            result["feedback"],
            "The response has issues:\n- Does not explain reasoning clearly\n- Could provide more context",
        )
        self.assertEqual(result["issues"], ["Does not explain reasoning clearly"])
        self.assertEqual(result["suggestions"], ["Could provide more context"])

        # Verify mock was called correctly
        self.mock_llm_provider.generate.assert_called_once()

    def test_validate(self):
        """Test validate method."""
        # Configure mock for no issues
        self.mock_llm_provider.generate.return_value = (
            "No issues found. The response aligns with all principles."
        )

        # Test with valid response
        result = self.critic.validate("Valid response", {"task": "Test task"})
        self.assertTrue(result)

        # Configure mock for issues
        self.mock_llm_provider.generate.return_value = (
            "The response has issues:\n- Does not explain reasoning clearly"
        )

        # Test with invalid response
        result = self.critic.validate("Invalid response", {"task": "Test task"})
        self.assertFalse(result)

    def test_improve(self):
        """Test improve method."""
        # Configure mocks for the first test
        self.mock_llm_provider.generate.reset_mock()
        self.mock_llm_provider.generate.side_effect = None

        # First call (critique) returns issues
        self.mock_llm_provider.generate.return_value = (
            "The response has issues:\n- Does not explain reasoning clearly"
        )

        # Test critique
        critique_result = self.critic.critique("Test response", {"task": "Test task"})
        self.assertTrue(critique_result["issues"])

        # Reset mock
        self.mock_llm_provider.generate.reset_mock()

        # Second call (improve) returns improved text
        self.mock_llm_provider.generate.return_value = "Improved response with clear reasoning"

        # Test improve with issues
        result = self.critic.improve_with_feedback(
            "Test response", "Does not explain reasoning clearly"
        )
        self.assertEqual(result, "Improved response with clear reasoning")

        # Verify mock was called
        self.mock_llm_provider.generate.assert_called_once()

        # Reset mock
        self.mock_llm_provider.generate.reset_mock()

        # Configure mock for no issues
        self.mock_llm_provider.generate.return_value = (
            "No issues found. The response aligns with all principles."
        )

        # Test critique with no issues
        critique_result = self.critic.critique("Valid response", {"task": "Test task"})
        self.assertFalse(critique_result["issues"])

        # Test improve with no issues
        result = self.critic.improve("Valid response", {"task": "Test task"})
        self.assertEqual(result, "Valid response")


class TestCreateConstitutionalCritic(unittest.TestCase):
    """Tests for the create_constitutional_critic factory function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock language model provider
        self.mock_llm_provider = MagicMock()

        # Set up default principles
        self.principles = ["Do not provide harmful content.", "Explain reasoning clearly."]

    def test_create_with_defaults(self):
        """Test creating a critic with default parameters."""
        critic = create_constitutional_critic(
            llm_provider=self.mock_llm_provider, principles=self.principles
        )

        # Check that the critic was created correctly
        self.assertIsInstance(critic, ConstitutionalCritic)
        self.assertEqual(critic._state.cache.get("principles"), self.principles)
        self.assertEqual(critic._state.model, self.mock_llm_provider)

    def test_create_with_custom_params(self):
        """Test creating a critic with custom parameters."""
        critic = create_constitutional_critic(
            llm_provider=self.mock_llm_provider,
            principles=self.principles,
            name="custom_critic",
            description="Custom description",
            min_confidence=0.8,
            system_prompt="Custom system prompt",
            temperature=0.5,
            max_tokens=2000,
        )

        # Check that the critic was created with custom parameters
        self.assertIsInstance(critic, ConstitutionalCritic)
        self.assertEqual(critic._state.cache.get("principles"), self.principles)
        self.assertEqual(critic._state.cache.get("system_prompt"), "Custom system prompt")
        self.assertEqual(critic._state.cache.get("temperature"), 0.5)
        self.assertEqual(critic._state.cache.get("max_tokens"), 2000)

    def test_create_with_config(self):
        """Test creating a critic with a config object."""
        config = ConstitutionalCriticConfig(
            name="config_critic",
            description="Config description",
            principles=self.principles,
            system_prompt="Config system prompt",
        )

        critic = create_constitutional_critic(
            llm_provider=self.mock_llm_provider,
            principles=[],  # Should be ignored since config has principles
            config=config,
        )

        # Check that the critic was created with the config
        self.assertIsInstance(critic, ConstitutionalCritic)
        self.assertEqual(critic._state.cache.get("principles"), self.principles)
        self.assertEqual(critic._state.cache.get("system_prompt"), "Config system prompt")


if __name__ == "__main__":
    unittest.main()
