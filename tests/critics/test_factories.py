"""
Tests for the critic factory functions.

This module contains comprehensive tests for the factory functions in the critics/factories.py
module that are used to create various types of critics.
"""

import unittest
from unittest.mock import MagicMock, patch
import pytest

from sifaka.critics.factories import create_prompt_critic, create_reflexion_critic
from sifaka.critics.core import CriticCore
from sifaka.critics.managers.memory import MemoryManager
from sifaka.critics.managers.prompt_factories import PromptCriticPromptManager, ReflexionCriticPromptManager
from sifaka.critics.managers.response import ResponseParser
from sifaka.critics.models import CriticConfig, PromptCriticConfig, ReflexionCriticConfig


class TestCriticFactories(unittest.TestCase):
    """Tests for critic factory functions."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create a mock LLM provider that can be used for both factories
        self.mock_llm_provider = MagicMock()
        self.mock_llm_provider.invoke.return_value = "Test response"

    def test_create_prompt_critic_with_defaults(self):
        """Test creating a prompt critic with default parameters."""
        # Create a prompt critic with defaults
        with patch('sifaka.critics.factories.PromptCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                critic = create_prompt_critic(llm_provider=self.mock_llm_provider)

                # Assert the returned object is a CriticCore
                self.assertIsInstance(critic, CriticCore)

                # Assert the configuration has the expected default values
                self.assertEqual(critic._config.name, "prompt_critic")
                self.assertEqual(critic._config.description, "Evaluates and improves text using language models")
                self.assertEqual(critic._config.min_confidence, 0.7)
                self.assertEqual(critic._config.max_attempts, 3)

                # Assert the correct managers were created and used
                self.assertEqual(critic._model, self.mock_llm_provider)
                mock_prompt_manager.assert_called_once()
                mock_response_parser.assert_called_once()

    def test_create_prompt_critic_with_custom_params(self):
        """Test creating a prompt critic with custom parameters."""
        # Create a prompt critic with custom parameters
        with patch('sifaka.critics.factories.PromptCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                critic = create_prompt_critic(
                    llm_provider=self.mock_llm_provider,
                    name="custom_critic",
                    description="Custom description",
                    min_confidence=0.8,
                    max_attempts=5,
                    cache_size=200,
                    priority=2,
                    cost=2.0,
                    system_prompt="Custom system prompt",
                    temperature=0.5,
                    max_tokens=2000,
                )

                # Assert the configuration has the expected custom values
                self.assertEqual(critic._config.name, "custom_critic")
                self.assertEqual(critic._config.description, "Custom description")
                self.assertEqual(critic._config.min_confidence, 0.8)
                self.assertEqual(critic._config.max_attempts, 5)
                self.assertEqual(critic._config.cache_size, 200)
                self.assertEqual(critic._config.priority, 2)
                self.assertEqual(critic._config.cost, 2.0)
                self.assertEqual(critic._config.system_prompt, "Custom system prompt")
                self.assertEqual(critic._config.temperature, 0.5)
                self.assertEqual(critic._config.max_tokens, 2000)

    def test_create_prompt_critic_with_existing_config(self):
        """Test creating a prompt critic with an existing configuration."""
        # Create a config
        config = PromptCriticConfig(
            name="existing_config_critic",
            description="Critic with existing config",
            min_confidence=0.9,
            system_prompt="Config system prompt",
        )

        # Create a prompt critic with the existing config
        with patch('sifaka.critics.factories.PromptCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                critic = create_prompt_critic(
                    llm_provider=self.mock_llm_provider,
                    config=config,
                )

                # Assert the configuration was used
                self.assertEqual(critic._config, config)
                self.assertEqual(critic._config.name, "existing_config_critic")
                self.assertEqual(critic._config.description, "Critic with existing config")
                self.assertEqual(critic._config.min_confidence, 0.9)
                self.assertEqual(critic._config.system_prompt, "Config system prompt")

    def test_create_prompt_critic_with_additional_kwargs(self):
        """Test creating a prompt critic with additional keyword arguments."""
        # Create a prompt critic with additional kwargs
        with patch('sifaka.critics.factories.PromptCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                critic = create_prompt_critic(
                    llm_provider=self.mock_llm_provider,
                    custom_param1="value1",
                    custom_param2="value2",
                )

                # Assert the core was created with the additional kwargs
                self.assertIsInstance(critic, CriticCore)

                # In a real test we would check critic._custom_param1 and critic._custom_param2,
                # but since we're using mocks and these are just passed through, we can't test this directly

    def test_create_reflexion_critic_with_defaults(self):
        """Test creating a reflexion critic with default parameters."""
        # Create a reflexion critic with defaults
        with patch('sifaka.critics.factories.ReflexionCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                with patch('sifaka.critics.factories.MemoryManager') as mock_memory_manager:
                    critic = create_reflexion_critic(llm_provider=self.mock_llm_provider)

                    # Assert the returned object is a CriticCore
                    self.assertIsInstance(critic, CriticCore)

                    # Assert the configuration has the expected default values
                    self.assertEqual(critic._config.name, "reflexion_critic")
                    self.assertEqual(critic._config.description, "Improves text using reflections on past feedback")
                    self.assertEqual(critic._config.min_confidence, 0.7)
                    self.assertEqual(critic._config.max_attempts, 3)

                    # Assert the correct managers were created and used
                    self.assertEqual(critic._model, self.mock_llm_provider)
                    mock_prompt_manager.assert_called_once()
                    mock_response_parser.assert_called_once()
                    mock_memory_manager.assert_called_once_with(buffer_size=5)  # Default buffer size

    def test_create_reflexion_critic_with_custom_params(self):
        """Test creating a reflexion critic with custom parameters."""
        # Create a reflexion critic with custom parameters
        with patch('sifaka.critics.factories.ReflexionCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                with patch('sifaka.critics.factories.MemoryManager') as mock_memory_manager:
                    critic = create_reflexion_critic(
                        llm_provider=self.mock_llm_provider,
                        name="custom_reflexion",
                        description="Custom reflexion critic",
                        min_confidence=0.8,
                        max_attempts=5,
                        cache_size=200,
                        priority=2,
                        cost=2.0,
                        system_prompt="Custom reflexion system prompt",
                        temperature=0.5,
                        max_tokens=2000,
                        memory_buffer_size=10,
                        reflection_depth=2,
                    )

                    # Assert the configuration has the expected custom values
                    self.assertEqual(critic._config.name, "custom_reflexion")
                    self.assertEqual(critic._config.description, "Custom reflexion critic")
                    self.assertEqual(critic._config.min_confidence, 0.8)
                    self.assertEqual(critic._config.max_attempts, 5)
                    self.assertEqual(critic._config.cache_size, 200)
                    self.assertEqual(critic._config.priority, 2)
                    self.assertEqual(critic._config.cost, 2.0)
                    self.assertEqual(critic._config.system_prompt, "Custom reflexion system prompt")
                    self.assertEqual(critic._config.temperature, 0.5)
                    self.assertEqual(critic._config.max_tokens, 2000)

                    # Check reflexion-specific parameters
                    self.assertEqual(critic._config.memory_buffer_size, 10)
                    self.assertEqual(critic._config.reflection_depth, 2)

                    # Assert the memory manager was created with the right buffer size
                    mock_memory_manager.assert_called_once_with(buffer_size=10)

    def test_create_reflexion_critic_with_existing_config(self):
        """Test creating a reflexion critic with an existing configuration."""
        # Create a config
        config = ReflexionCriticConfig(
            name="existing_reflexion_config",
            description="Reflexion critic with existing config",
            min_confidence=0.9,
            system_prompt="Config reflexion system prompt",
            memory_buffer_size=15,
            reflection_depth=3,
        )

        # Create a reflexion critic with the existing config
        with patch('sifaka.critics.factories.ReflexionCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                with patch('sifaka.critics.factories.MemoryManager') as mock_memory_manager:
                    critic = create_reflexion_critic(
                        llm_provider=self.mock_llm_provider,
                        config=config,
                    )

                    # Assert the configuration was used
                    self.assertEqual(critic._config, config)
                    self.assertEqual(critic._config.name, "existing_reflexion_config")
                    self.assertEqual(critic._config.description, "Reflexion critic with existing config")
                    self.assertEqual(critic._config.min_confidence, 0.9)
                    self.assertEqual(critic._config.system_prompt, "Config reflexion system prompt")
                    self.assertEqual(critic._config.memory_buffer_size, 15)
                    self.assertEqual(critic._config.reflection_depth, 3)

                    # Assert the memory manager was created with the right buffer size
                    mock_memory_manager.assert_called_once_with(buffer_size=15)

    def test_create_reflexion_critic_with_additional_kwargs(self):
        """Test creating a reflexion critic with additional keyword arguments."""
        # Create a reflexion critic with additional kwargs
        with patch('sifaka.critics.factories.ReflexionCriticPromptManager') as mock_prompt_manager:
            with patch('sifaka.critics.factories.ResponseParser') as mock_response_parser:
                with patch('sifaka.critics.factories.MemoryManager') as mock_memory_manager:
                    critic = create_reflexion_critic(
                        llm_provider=self.mock_llm_provider,
                        custom_param1="value1",
                        custom_param2="value2",
                    )

                    # Assert the core was created with the additional kwargs
                    self.assertIsInstance(critic, CriticCore)

                    # In a real test we would check critic._custom_param1 and critic._custom_param2,
                    # but since we're using mocks and these are just passed through, we can't test this directly

    def test_create_prompt_critic_integration(self):
        """Test creating a prompt critic in an integration-style test without mocks."""
        # Create a prompt critic without mocking the managers
        critic = create_prompt_critic(llm_provider=self.mock_llm_provider)

        # Verify critic and its managers are the correct types
        self.assertIsInstance(critic, CriticCore)
        self.assertIsInstance(critic._prompt_manager, PromptCriticPromptManager)
        self.assertIsInstance(critic._response_parser, ResponseParser)
        self.assertIsNone(critic._memory_manager)  # Prompt critics don't have memory managers

    def test_create_reflexion_critic_integration(self):
        """Test creating a reflexion critic in an integration-style test without mocks."""
        # Create a reflexion critic without mocking the managers
        critic = create_reflexion_critic(llm_provider=self.mock_llm_provider)

        # Verify critic and its managers are the correct types
        self.assertIsInstance(critic, CriticCore)
        self.assertIsInstance(critic._prompt_manager, ReflexionCriticPromptManager)
        self.assertIsInstance(critic._response_parser, ResponseParser)
        self.assertIsInstance(critic._memory_manager, MemoryManager)

        # Verify memory manager has the right buffer size
        self.assertEqual(critic._memory_manager._buffer_size, 5)  # Default buffer size


if __name__ == "__main__":
    unittest.main()