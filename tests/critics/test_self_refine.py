"""
Tests for the SelfRefineCritic.
"""

import unittest
from unittest.mock import MagicMock, patch

from sifaka.critics.self_refine import (
    SelfRefineCritic,
    SelfRefineCriticConfig,
    create_self_refine_critic,
)


class TestSelfRefineCritic(unittest.TestCase):
    """Test cases for the SelfRefineCritic."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock language model provider
        self.mock_llm_provider = MagicMock()
        self.mock_llm_provider.generate.side_effect = self._mock_generate
        self.mock_llm_provider.agenerate = self._mock_agenerate

        # Create a basic configuration
        self.config = SelfRefineCriticConfig(
            name="test_critic",
            description="Test critic",
            max_iterations=2,
            system_prompt="You are a test critic",
        )

    def _mock_generate(self, prompt, **kwargs):
        """Mock the generate method of the language model provider."""
        if "Critique" in prompt:
            if "needs improvement" in prompt:
                return "The text needs more detail and better structure."
            else:
                return "No issues found. The text looks good."
        elif "Revise" in prompt:
            if "The text needs more detail and better structure." in prompt:
                return (
                    "This is an improved version of the text with more detail and better structure."
                )
            else:
                return (
                    "Improved: " + prompt.split("Original Response:")[-1].split("\n\n")[0].strip()
                )
        else:
            return "Default response"

    async def _mock_agenerate(self, prompt, **kwargs):
        """Mock the agenerate method of the language model provider."""
        return self._mock_generate(prompt, **kwargs)

    def test_initialization(self):
        """Test that the critic initializes correctly."""
        critic = SelfRefineCritic(config=self.config, llm_provider=self.mock_llm_provider)

        # Check that the critic was initialized correctly
        self.assertEqual(critic._state.model, self.mock_llm_provider)
        self.assertEqual(critic._state.cache.get("max_iterations"), 2)
        self.assertEqual(critic._state.cache.get("system_prompt"), "You are a test critic")
        self.assertTrue(critic._state.initialized)

    def test_validate(self):
        """Test the validate method."""
        critic = SelfRefineCritic(config=self.config, llm_provider=self.mock_llm_provider)

        # Test with text that needs improvement
        result = critic.validate("This text needs improvement", {"task": "Improve the text"})
        self.assertFalse(result)

        # Test with text that doesn't need improvement
        result = critic.validate("This text is good", {"task": "Improve the text"})
        self.assertTrue(result)

    def test_critique(self):
        """Test the critique method."""
        critic = SelfRefineCritic(config=self.config, llm_provider=self.mock_llm_provider)

        # Test critique
        result = critic.critique("This text needs improvement", {"task": "Improve the text"})

        # Check the result
        self.assertIn("score", result)
        self.assertIn("feedback", result)
        self.assertIn("issues", result)
        self.assertIn("suggestions", result)

    def test_improve(self):
        """Test the improve method."""
        # Create a simple mock that returns predefined responses
        mock_llm = MagicMock()

        # Define the responses for each call
        responses = [
            "The text needs more detail and better structure.",  # First critique
            "This is an improved version of the text with more detail and better structure.",  # First revision
            "No issues found. The text looks good.",  # Second critique (for the improved text)
        ]

        # Set up the mock to return the predefined responses
        mock_llm.generate = MagicMock(side_effect=responses)

        # Create critic with our mock
        critic = SelfRefineCritic(config=self.config, llm_provider=mock_llm)

        # Test improvement
        result = critic.improve("This text needs improvement", {"task": "Improve the text"})

        # Check that the text was improved
        self.assertEqual(
            result, "This is an improved version of the text with more detail and better structure."
        )

        # Create a new mock for the second test
        mock_llm2 = MagicMock()
        mock_llm2.generate = MagicMock(return_value="No issues found. The text looks good.")

        # Create a new critic with the second mock
        critic2 = SelfRefineCritic(config=self.config, llm_provider=mock_llm2)

        # Test with text that doesn't need improvement
        result = critic2.improve("This text is good", {"task": "Improve the text"})

        # Check that the text was not changed
        self.assertEqual(result, "This text is good")

    def test_create_self_refine_critic(self):
        """Test the create_self_refine_critic factory function."""
        # Create with parameters
        critic = create_self_refine_critic(
            llm_provider=self.mock_llm_provider,
            name="factory_critic",
            description="Factory description",
            max_iterations=3,
        )

        # Check that the critic was created correctly
        self.assertIsInstance(critic, SelfRefineCritic)
        self.assertEqual(critic._state.cache.get("max_iterations"), 3)

        # Create with config
        config = SelfRefineCriticConfig(
            name="config_critic",
            description="Config description",
            max_iterations=4,
        )

        critic = create_self_refine_critic(
            llm_provider=self.mock_llm_provider,
            config=config,
        )

        # Check that the critic was created with the config
        self.assertIsInstance(critic, SelfRefineCritic)
        self.assertEqual(critic._state.cache.get("max_iterations"), 4)


if __name__ == "__main__":
    unittest.main()
