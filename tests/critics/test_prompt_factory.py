"""
Consolidated tests for DefaultPromptFactory and factory-related functionality.

This file combines factory-related tests previously distributed across multiple files.
"""

import pytest
from unittest.mock import MagicMock

from sifaka.critics.prompt import (
    PromptCritic,
    PromptCriticConfig,
    DefaultPromptFactory,
    create_prompt_critic,
    DEFAULT_SYSTEM_PROMPT,
)


class TestDefaultPromptFactory:
    """Test cases for DefaultPromptFactory."""

    def test_create_validation_prompt(self):
        """Test creation of validation prompt."""
        factory = DefaultPromptFactory()
        prompt = factory.create_validation_prompt("Test text")

        assert "Test text" in prompt
        assert "VALID:" in prompt
        assert "REASON:" in prompt

    def test_create_critique_prompt(self):
        """Test creation of critique prompt."""
        factory = DefaultPromptFactory()
        prompt = factory.create_critique_prompt("Test text")

        assert "Test text" in prompt
        assert "SCORE:" in prompt
        assert "FEEDBACK:" in prompt
        assert "ISSUES:" in prompt
        assert "SUGGESTIONS:" in prompt

    def test_create_improvement_prompt(self):
        """Test creation of improvement prompt."""
        factory = DefaultPromptFactory()
        prompt = factory.create_improvement_prompt("Test text", "Improve clarity")

        assert "Test text" in prompt
        assert "Improve clarity" in prompt
        assert "IMPROVED_TEXT:" in prompt

    def test_create_critic(self):
        """Test factory method for creating critics."""
        factory = DefaultPromptFactory()
        mock_model = MagicMock()
        mock_model.model_name = "test-model"

        critic = factory.create_critic(mock_model)

        assert isinstance(critic, PromptCritic)
        assert critic._model == mock_model
        assert critic.config.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_create_with_custom_prompt(self):
        """Test creating a critic with custom system prompt."""
        factory = DefaultPromptFactory()
        mock_model = MagicMock()
        mock_model.model_name = "test-model"

        critic = factory.create_with_custom_prompt(
            model=mock_model,
            system_prompt="You are a very specific critic",
            min_confidence=0.8,
            temperature=0.3,
        )

        assert isinstance(critic, PromptCritic)
        assert critic._model == mock_model
        assert critic.config.system_prompt == "You are a very specific critic"
        assert critic.config.min_confidence == 0.8
        assert critic.config.temperature == 0.3


class TestCreatePromptCritic:
    """Test cases for create_prompt_critic function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock language model."""
        mock = MagicMock()
        mock.model_name = "test-model"
        return mock

    def test_create_prompt_critic_factory_function(self, mock_model):
        """Test the create_prompt_critic factory function."""
        critic = create_prompt_critic(
            model=mock_model,
            name="factory_critic",
            description="Created with factory",
            system_prompt="Custom prompt",
            temperature=0.3,
            max_tokens=200,
            min_confidence=0.5,
        )

        assert isinstance(critic, PromptCritic)
        assert critic.config.name == "factory_critic"
        assert critic.config.description == "Created with factory"
        assert critic.config.system_prompt == "Custom prompt"
        assert critic.config.temperature == 0.3
        assert critic.config.max_tokens == 200
        assert critic.config.min_confidence == 0.5
