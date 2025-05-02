"""
Tests for the prompt critic module.

This module contains tests for the prompt critic components defined in sifaka/critics/prompt.py,
including DefaultPromptFactory for creating prompts and other functionality.
"""

import unittest
from unittest.mock import MagicMock, patch


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

    def test_invalid_system_prompt(self):
        """Test creating a config with an invalid system prompt."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCriticConfig

        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="",  # Empty system prompt
                temperature=0.5,
                max_tokens=100,
            )

    def test_invalid_temperature(self):
        """Test creating a config with an invalid temperature."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCriticConfig

        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=1.5,  # Temperature > 1
                max_tokens=100,
            )

    def test_invalid_max_tokens(self):
        """Test creating a config with invalid max_tokens."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCriticConfig

        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=0.5,
                max_tokens=0,  # Max tokens must be positive
            )


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

    def test_create_critic(self):
        """Test creating a critic."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCritic

        mock_llm = MagicMock()
        mock_llm.model_name = "test_model"

        # Mock the PromptCritic class
        with patch('sifaka.critics.prompt.PromptCritic') as mock_critic:
            mock_critic.return_value = MagicMock()

            # Call the method
            critic = self.factory.create_critic(mock_llm)

            # Check that PromptCritic was called
            mock_critic.assert_called_once()

    def test_create_with_custom_prompt(self):
        """Test creating a critic with a custom prompt."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

        mock_llm = MagicMock()
        mock_llm.model_name = "test_model"

        # Mock the required classes
        with patch('sifaka.critics.prompt.PromptCritic') as mock_critic, \
             patch('sifaka.critics.prompt.PromptCriticConfig') as mock_config:
            mock_config.return_value = MagicMock()
            mock_critic.return_value = MagicMock()

            # Call the method
            critic = self.factory.create_with_custom_prompt(
                mock_llm,
                "Custom system prompt",
                min_confidence=0.8,
                temperature=0.5
            )

            # Check that the required classes were called
            mock_config.assert_called_once()
            mock_critic.assert_called_once()


class TestPromptCritic(unittest.TestCase):
    """Tests for the PromptCritic class."""

    def setUp(self):
        """Set up test environment."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

        # Create a mock LLM provider
        self.mock_llm = MagicMock()
        self.mock_llm.model_name = "test_model"
        self.mock_llm.generate.return_value = "IMPROVED_TEXT: Improved text"

        # Create a mock critique service
        self.mock_critique_service = MagicMock()
        self.mock_critique_service.improve.return_value = "Improved text"
        self.mock_critique_service.validate.return_value = True
        self.mock_critique_service.critique.return_value = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": [],
            "suggestions": []
        }

        # Create a config
        self.config = PromptCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="Test system prompt",
            temperature=0.5,
            max_tokens=100,
        )

        # Patch the components
        with patch('sifaka.critics.managers.prompt_factories.PromptCriticPromptManager'), \
             patch('sifaka.critics.managers.response.ResponseParser'), \
             patch('sifaka.critics.services.critique.CritiqueService') as mock_service_class:
            mock_service_class.return_value = self.mock_critique_service
            self.critic = PromptCritic(
                name="test_critic",
                description="Test critic",
                llm_provider=self.mock_llm,
                config=self.config
            )

    def test_init_without_llm(self):
        """Test initializing without an LLM provider."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCritic
        from pydantic import ValidationError

        # Patch ValidationError
        with patch('sifaka.critics.prompt.ValidationError') as mock_validation_error:
            mock_validation_error.from_exception_data.return_value = Exception("Field required")

            # Try to create a critic without an LLM
            with self.assertRaises(Exception):
                PromptCritic(name="test_critic", description="Test critic")

    def test_improve(self):
        """Test the improve method."""
        result = self.critic.improve("Text to improve", "Feedback")
        self.assertEqual(result, "Improved text")
        self.mock_critique_service.improve.assert_called_once_with("Text to improve", "Feedback")

    def test_improve_with_empty_text(self):
        """Test the improve method with empty text."""
        with self.assertRaises(ValueError):
            self.critic.improve("", "Feedback")

    def test_improve_with_feedback(self):
        """Test the improve_with_feedback method."""
        result = self.critic.improve_with_feedback("Text to improve", "Feedback")
        self.assertEqual(result, "Improved text")
        self.mock_critique_service.improve.assert_called_once_with("Text to improve", "Feedback")

    def test_improve_with_feedback_empty_text(self):
        """Test the improve_with_feedback method with empty text."""
        with self.assertRaises(ValueError):
            self.critic.improve_with_feedback("", "Feedback")

    def test_improve_with_feedback_empty_feedback(self):
        """Test the improve_with_feedback method with empty feedback."""
        with self.assertRaises(ValueError):
            self.critic.improve_with_feedback("Text to improve", "")

    def test_critique(self):
        """Test the critique method."""
        result = self.critic.critique("Text to critique")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")
        self.mock_critique_service.critique.assert_called_once_with("Text to critique")

    def test_critique_empty_text(self):
        """Test the critique method with empty text."""
        with self.assertRaises(ValueError):
            self.critic.critique("")

    def test_validate(self):
        """Test the validate method."""
        result = self.critic.validate("Text to validate")
        self.assertTrue(result)
        self.mock_critique_service.validate.assert_called_once_with("Text to validate")

    def test_validate_empty_text(self):
        """Test the validate method with empty text."""
        with self.assertRaises(ValueError):
            self.critic.validate("")

    @patch('asyncio.run')
    def test_async_methods(self, mock_run):
        """Test the async methods."""
        mock_run.return_value = True

        # Setup async methods
        self.mock_critique_service.avalidate.return_value = True
        self.mock_critique_service.acritique.return_value = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": [],
            "suggestions": []
        }
        self.mock_critique_service.aimprove.return_value = "Improved text"

        # Test avalidate
        self.critic.avalidate("Text to validate")
        self.mock_critique_service.avalidate.assert_called_once_with("Text to validate")

        # Test acritique
        self.critic.acritique("Text to critique")
        self.mock_critique_service.acritique.assert_called_once_with("Text to critique")

        # Test aimprove
        self.critic.aimprove("Text to improve", "Feedback")
        self.mock_critique_service.aimprove.assert_called_once_with("Text to improve", "Feedback")


class TestCreatePromptCritic(unittest.TestCase):
    """Tests for the create_prompt_critic function."""

    def setUp(self):
        """Set up test environment."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import PromptCritic

        # Create a mock LLM provider
        self.mock_llm = MagicMock()
        self.mock_llm.model_name = "test_model"

    def test_create_with_defaults(self):
        """Test creating a critic with default parameters."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import create_prompt_critic, PromptCritic, PromptCriticConfig

        # Mock the required classes
        with patch('sifaka.critics.prompt.PromptCritic') as mock_critic, \
             patch('sifaka.critics.prompt.PromptCriticConfig') as mock_config:
            mock_config.return_value = MagicMock()
            mock_critic.return_value = MagicMock()

            # Call the function
            critic = create_prompt_critic(self.mock_llm)

            # Check that the required classes were called
            mock_config.assert_called_once()
            mock_critic.assert_called_once()

    def test_create_with_custom_parameters(self):
        """Test creating a critic with custom parameters."""
        # Import here to use the mock imports
        from sifaka.critics.prompt import create_prompt_critic, PromptCritic, PromptCriticConfig

        # Mock the required classes
        with patch('sifaka.critics.prompt.PromptCritic') as mock_critic, \
             patch('sifaka.critics.prompt.PromptCriticConfig') as mock_config:
            mock_config.return_value = MagicMock()
            mock_critic.return_value = MagicMock()

            # Call the function with custom parameters
            critic = create_prompt_critic(
                self.mock_llm,
                name="custom_critic",
                description="Custom description",
                system_prompt="Custom system prompt",
                temperature=0.3,
                max_tokens=500,
                min_confidence=0.8
            )

            # Check that the required classes were called with the right parameters
            mock_config.assert_called_once_with(
                name="custom_critic",
                description="Custom description",
                system_prompt="Custom system prompt",
                temperature=0.3,
                max_tokens=500,
                min_confidence=0.8
            )
            mock_critic.assert_called_once()


if __name__ == "__main__":
    unittest.main()