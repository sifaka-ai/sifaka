"""
Tests for the prompt critic module.

This module contains tests for the prompt critic components defined in sifaka/critics/prompt.py,
including PromptCritic, DefaultPromptFactory, and the create_prompt_critic function.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from sifaka.critics.prompt import (
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_SYSTEM_PROMPT,
    DefaultPromptFactory,
    LanguageModel,
    PromptCritic,
    PromptCriticConfig,
    create_prompt_critic,
)


class TestPromptCriticConfig(unittest.TestCase):
    """Tests for the PromptCriticConfig."""

    def test_valid_config(self):
        """Test creating a valid config."""
        config = PromptCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="Test system prompt",
            temperature=0.5,
            max_tokens=100,
            min_confidence=0.8,
            max_attempts=2,
        )
        self.assertEqual(config.name, "test_critic")
        self.assertEqual(config.description, "Test critic")
        self.assertEqual(config.system_prompt, "Test system prompt")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 100)
        self.assertEqual(config.min_confidence, 0.8)
        self.assertEqual(config.max_attempts, 2)

    def test_empty_system_prompt(self):
        """Test that an empty system prompt raises an error."""
        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="",
                temperature=0.5,
                max_tokens=100,
            )

    def test_invalid_temperature(self):
        """Test that an invalid temperature raises an error."""
        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=1.5,
                max_tokens=100,
            )

    def test_invalid_max_tokens(self):
        """Test that an invalid max_tokens raises an error."""
        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test_critic",
                description="Test critic",
                system_prompt="Test system prompt",
                temperature=0.5,
                max_tokens=0,
            )


class TestDefaultPromptFactory(unittest.TestCase):
    """Tests for the DefaultPromptFactory class."""

    def setUp(self):
        """Set up a DefaultPromptFactory instance for testing."""
        self.factory = DefaultPromptFactory()
        self.mock_model = MagicMock()
        self.mock_model.model_name = "test_model"

    def test_create_validation_prompt(self):
        """Test creating a validation prompt."""
        prompt = self.factory.create_validation_prompt("Text to validate")
        self.assertIn("TEXT TO VALIDATE:", prompt)
        self.assertIn("Text to validate", prompt)
        self.assertIn("VALID: [true/false]", prompt)
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


class TestCreatePromptCritic(unittest.TestCase):
    """Tests for the create_prompt_critic function."""

    def setUp(self):
        """Set up a mock model for testing."""
        self.mock_model = MagicMock()
        self.mock_model.model_name = "test_model"

    def test_create_with_defaults(self):
        """Test creating a critic with default parameters."""
        with patch("sifaka.critics.prompt.PromptCritic") as mock_critic_class:
            mock_instance = MagicMock()
            mock_critic_class.return_value = mock_instance

            critic = create_prompt_critic(self.mock_model)

            # Check that PromptCritic was called with expected arguments
            mock_critic_class.assert_called_once()
            # Extract the config passed to PromptCritic
            config = mock_critic_class.call_args[1]["config"]

            self.assertEqual(config.name, "factory_critic")
            self.assertEqual(config.description, "Evaluates and improves text using language models")
            self.assertEqual(config.system_prompt, DEFAULT_SYSTEM_PROMPT)

    def test_create_with_custom_parameters(self):
        """Test creating a critic with custom parameters."""
        with patch("sifaka.critics.prompt.PromptCritic") as mock_critic_class:
            mock_instance = MagicMock()
            mock_critic_class.return_value = mock_instance

            critic = create_prompt_critic(
                self.mock_model,
                name="custom_critic",
                description="Custom description",
                system_prompt="Custom system prompt",
                temperature=0.3,
                max_tokens=500,
            )

            # Check that PromptCritic was called with expected arguments
            mock_critic_class.assert_called_once()
            # Extract the config passed to PromptCritic
            config = mock_critic_class.call_args[1]["config"]

            self.assertEqual(config.name, "custom_critic")
            self.assertEqual(config.description, "Custom description")
            self.assertEqual(config.system_prompt, "Custom system prompt")


class TestCreatePrompt(unittest.TestCase):
    """Test create prompt methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = DefaultPromptFactory()

    def test_create_validation_prompt(self):
        """Test create_validation_prompt method."""
        prompt = self.factory.create_validation_prompt("Test text")
        self.assertIn("Test text", prompt)
        self.assertIn("VALID:", prompt)
        self.assertIn("REASON:", prompt)

    def test_create_critique_prompt(self):
        """Test create_critique_prompt method."""
        prompt = self.factory.create_critique_prompt("Test text")
        self.assertIn("Test text", prompt)
        self.assertIn("SCORE:", prompt)
        self.assertIn("FEEDBACK:", prompt)
        self.assertIn("ISSUES:", prompt)
        self.assertIn("SUGGESTIONS:", prompt)

    def test_create_improvement_prompt(self):
        """Test create_improvement_prompt method."""
        prompt = self.factory.create_improvement_prompt("Test text", "Test feedback")
        self.assertIn("Test text", prompt)
        self.assertIn("Test feedback", prompt)
        self.assertIn("IMPROVED_TEXT:", prompt)

    def test_create_critic_with_defaults(self):
        """Test create_critic method with defaults."""
        mock_llm = MagicMock()
        with patch('sifaka.critics.prompt.PromptCritic') as mock_critic_cls:
            self.factory.create_critic(mock_llm)
            mock_critic_cls.assert_called_once()

    def test_create_critic_with_config(self):
        """Test create_critic method with custom config."""
        mock_llm = MagicMock()
        mock_config = MagicMock()
        with patch('sifaka.critics.prompt.PromptCritic') as mock_critic_cls:
            self.factory.create_critic(mock_llm, mock_config)
            mock_critic_cls.assert_called_once_with(config=mock_config, llm_provider=mock_llm)

    def test_create_with_custom_prompt(self):
        """Test create_with_custom_prompt method."""
        mock_llm = MagicMock()
        with patch('sifaka.critics.prompt.PromptCritic') as mock_critic_cls:
            with patch('sifaka.critics.prompt.PromptCriticConfig') as mock_config_cls:
                self.factory.create_with_custom_prompt(
                    mock_llm,
                    "Custom system prompt",
                    min_confidence=0.8,
                    temperature=0.5
                )
                mock_config_cls.assert_called_once()
                mock_critic_cls.assert_called_once()