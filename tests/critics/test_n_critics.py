"""
Tests for the N-Critics critic.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from sifaka.models.base import Model
from sifaka.critics.n_critics import NCriticsCritic, create_n_critics_critic
from sifaka.errors import ImproverError


class TestNCriticsCritic(unittest.TestCase):
    """Tests for the N-Critics critic."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MagicMock(spec=Model)
        self.model.generate.return_value = '{"role": "Test Critic", "needs_improvement": true, "score": 5, "issues": ["Test issue"], "suggestions": ["Test suggestion"], "explanation": "Test explanation"}'
        self.critic = NCriticsCritic(model=self.model, num_critics=2)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.critic.model, self.model)
        self.assertEqual(self.critic.num_critics, 2)
        self.assertEqual(self.critic.max_refinement_iterations, 2)
        self.assertEqual(len(self.critic.critic_roles), 2)

    def test_init_with_invalid_values(self):
        """Test initialization with invalid values."""
        # Test with num_critics < 1
        critic = NCriticsCritic(model=self.model, num_critics=0)
        self.assertEqual(critic.num_critics, 1)  # Should be clamped to 1

        # Test with num_critics > 5
        critic = NCriticsCritic(model=self.model, num_critics=10)
        self.assertEqual(critic.num_critics, 5)  # Should be clamped to 5

        # Test with max_refinement_iterations < 1
        critic = NCriticsCritic(model=self.model, max_refinement_iterations=0)
        self.assertEqual(critic.max_refinement_iterations, 1)  # Should be clamped to 1

    def test_critique(self):
        """Test the _critique method."""
        critique = self.critic._critique("Test text")

        # Check that the model was called for each critic
        self.assertEqual(self.model.generate.call_count, 3)  # 2 critics + 1 aggregation

        # Check the structure of the critique
        self.assertIn("needs_improvement", critique)
        self.assertIn("message", critique)
        self.assertIn("critic_critiques", critique)
        self.assertIn("aggregated_critique", critique)
        self.assertIn("issues", critique)
        self.assertIn("suggestions", critique)

        # Check the critic critiques
        self.assertEqual(len(critique["critic_critiques"]), 2)
        for critic_critique in critique["critic_critiques"]:
            self.assertIn("role", critic_critique)
            self.assertIn("needs_improvement", critic_critique)
            self.assertIn("score", critic_critique)
            self.assertIn("issues", critic_critique)
            self.assertIn("suggestions", critic_critique)
            self.assertIn("explanation", critic_critique)

    def test_critique_with_json_error(self):
        """Test the _critique method with JSON parsing error."""
        self.model.generate.return_value = "Invalid JSON"
        critique = self.critic._critique("Test text")

        # Check that the model was called for each critic
        self.assertEqual(self.model.generate.call_count, 3)  # 2 critics + 1 aggregation

        # Check that default values were used for the critiques
        for critic_critique in critique["critic_critiques"]:
            self.assertIn("role", critic_critique)
            self.assertIn("needs_improvement", critic_critique)
            self.assertEqual(critic_critique["needs_improvement"], True)
            self.assertIn("score", critic_critique)
            self.assertEqual(critic_critique["score"], 5)
            self.assertIn("issues", critic_critique)
            self.assertIn("suggestions", critic_critique)
            self.assertIn("explanation", critic_critique)

    def test_critique_with_model_error(self):
        """Test the _critique method with model error."""
        self.model.generate.side_effect = Exception("Model error")

        with self.assertRaises(ImproverError):
            self.critic._critique("Test text")

    def test_improve(self):
        """Test the _improve method."""
        # Mock the _critique method to return a predefined critique
        with patch.object(self.critic, "_critique") as mock_critique:
            # First critique (original text)
            first_critique = {
                "needs_improvement": True,
                "message": "Test message",
                "critic_critiques": [
                    {
                        "role": "Test Critic 1",
                        "needs_improvement": True,
                        "score": 5,
                        "issues": ["Test issue 1"],
                        "suggestions": ["Test suggestion 1"],
                        "explanation": "Test explanation 1",
                    },
                    {
                        "role": "Test Critic 2",
                        "needs_improvement": True,
                        "score": 6,
                        "issues": ["Test issue 2"],
                        "suggestions": ["Test suggestion 2"],
                        "explanation": "Test explanation 2",
                    },
                ],
                "aggregated_critique": {
                    "summary": "Test summary",
                    "issues": ["Test issue 1", "Test issue 2"],
                    "suggestions": ["Test suggestion 1", "Test suggestion 2"],
                    "average_score": 5.5,
                },
                "issues": ["Test issue 1", "Test issue 2"],
                "suggestions": ["Test suggestion 1", "Test suggestion 2"],
            }

            # Second critique (improved text)
            second_critique = {
                "needs_improvement": False,
                "message": "Test message 2",
                "critic_critiques": [
                    {
                        "role": "Test Critic 1",
                        "needs_improvement": False,
                        "score": 8,
                        "issues": [],
                        "suggestions": [],
                        "explanation": "Much better",
                    },
                    {
                        "role": "Test Critic 2",
                        "needs_improvement": False,
                        "score": 9,
                        "issues": [],
                        "suggestions": [],
                        "explanation": "Great improvement",
                    },
                ],
                "aggregated_critique": {
                    "summary": "Great improvement",
                    "issues": [],
                    "suggestions": [],
                    "average_score": 8.5,
                },
                "issues": [],
                "suggestions": [],
            }

            # Set up the mock to return different values on successive calls
            mock_critique.side_effect = [second_critique, second_critique]

            # Mock the _generate_improved_text method to return a predefined text
            with patch.object(self.critic, "_generate_improved_text") as mock_generate:
                mock_generate.return_value = "Improved text"

                # Test the _improve method
                improved_text = self.critic._improve("Test text", first_critique)

                # Check that the methods were called
                self.assertTrue(mock_critique.call_count >= 1)
                self.assertTrue(mock_generate.call_count >= 1)

                # Check the improved text
                self.assertEqual(improved_text, "Improved text")

    def test_factory_function(self):
        """Test the factory function."""
        critic = create_n_critics_critic(
            model=self.model,
            system_prompt="Test prompt",
            temperature=0.5,
            num_critics=3,
            max_refinement_iterations=2,
        )

        self.assertIsInstance(critic, NCriticsCritic)
        self.assertEqual(critic.model, self.model)
        self.assertEqual(critic.system_prompt, "Test prompt")
        self.assertEqual(critic.temperature, 0.5)
        self.assertEqual(critic.num_critics, 3)
        self.assertEqual(critic.max_refinement_iterations, 2)


if __name__ == "__main__":
    unittest.main()
