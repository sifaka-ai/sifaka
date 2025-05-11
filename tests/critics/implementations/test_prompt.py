"""
Tests for prompt critic implementation.

This module contains tests for the prompt critic implementation in
sifaka/critics/implementations/prompt.py.
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from sifaka.critics.implementations.prompt import (
    PromptCritic,
    create_prompt_critic,
)
from sifaka.critics.config import PromptCriticConfig
from sifaka.core.base import BaseResult as CriticResult


class TestPromptCritic:
    """Tests for PromptCritic."""

    def test_initialization(self):
        """Test that critic initializes correctly with state management."""
        # Create mock model provider
        mock_provider = MagicMock()
        
        # Create critic
        critic = PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=mock_provider,
        )
        
        # Check that state is initialized
        assert critic._state_manager.get("model") == mock_provider
        assert critic._state_manager.get("cache") == {}
        assert critic._state_manager.get_metadata("component_type") == "critic"
        assert critic._state_manager.get_metadata("name") == "test_critic"
        assert critic._state_manager.get_metadata("description") == "Test critic"

    def test_lazy_initialization(self):
        """Test that components are lazily initialized."""
        # Create mock model provider
        mock_provider = MagicMock()
        
        # Create critic
        critic = PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=mock_provider,
        )
        
        # Check that components are not initialized yet
        assert critic._state_manager.get("prompt_manager") is None
        assert critic._state_manager.get("response_parser") is None
        assert critic._state_manager.get("memory_manager") is None
        assert "critique_service" not in critic._state_manager.get("cache", {})
        
        # Initialize components
        critic._initialize_components()
        
        # Check that components are now initialized
        assert critic._state_manager.get("prompt_manager") is not None
        assert critic._state_manager.get("response_parser") is not None
        assert critic._state_manager.get("memory_manager") is not None
        assert "critique_service" in critic._state_manager.get("cache", {})

    @patch("sifaka.critics.services.critique.CritiqueService")
    def test_critique(self, mock_critique_service):
        """Test that critique method updates state correctly."""
        # Create mock model provider and critique service
        mock_provider = MagicMock()
        mock_service = MagicMock()
        mock_critique_service.return_value = mock_service
        
        # Mock critique result
        mock_service.critique.return_value = {
            "score": 0.8,
            "feedback": "Good text",
            "issues": [],
            "suggestions": [],
        }
        
        # Create critic
        critic = PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=mock_provider,
        )
        
        # Initialize components manually
        critic._initialize_components()
        
        # Call critique method
        result = critic.critique("Test text")
        
        # Check that state is updated
        assert critic._state_manager.get_metadata("critique_count") == 1
        assert critic._state_manager.get_metadata("last_critique_time") is not None
        
        # Check that result is correct
        assert result["score"] == 0.8
        assert result["feedback"] == "Good text"
        assert result["issues"] == []
        assert result["suggestions"] == []

    @patch("sifaka.critics.services.critique.CritiqueService")
    def test_improve(self, mock_critique_service):
        """Test that improve method updates state correctly."""
        # Create mock model provider and critique service
        mock_provider = MagicMock()
        mock_service = MagicMock()
        mock_critique_service.return_value = mock_service
        
        # Mock improve result
        mock_service.improve.return_value = "Improved text"
        
        # Create critic
        critic = PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=mock_provider,
        )
        
        # Initialize components manually
        critic._initialize_components()
        
        # Call improve method
        result = critic.improve("Test text", "Make it better")
        
        # Check that state is updated
        assert critic._state_manager.get_metadata("improvement_count") == 1
        assert critic._state_manager.get_metadata("last_improvement_time") is not None
        
        # Check that result is correct
        assert result == "Improved text"

    @patch("sifaka.critics.services.critique.CritiqueService")
    def test_validate(self, mock_critique_service):
        """Test that validate method updates state correctly."""
        # Create mock model provider and critique service
        mock_provider = MagicMock()
        mock_service = MagicMock()
        mock_critique_service.return_value = mock_service
        
        # Mock validate result
        mock_service.validate.return_value = True
        
        # Create critic
        critic = PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=mock_provider,
            config=PromptCriticConfig(min_confidence=0.7),
        )
        
        # Initialize components manually
        critic._initialize_components()
        
        # Call validate method
        result = critic.validate("Test text")
        
        # Check that state is updated
        assert critic._state_manager.get_metadata("validation_count") == 1
        
        # Check that result is correct
        assert result is True

    def test_factory_function(self):
        """Test that factory function creates critic correctly."""
        # Create mock model provider
        mock_provider = MagicMock()
        
        # Create critic using factory function
        critic = create_prompt_critic(
            llm_provider=mock_provider,
            name="test_critic",
            description="Test critic",
            system_prompt="You are a helpful critic",
            temperature=0.7,
            max_tokens=1000,
            min_confidence=0.8,
            max_attempts=3,
            cache_size=100,
            priority=1,
            cost=0.1,
            track_performance=True,
            track_errors=True,
        )
        
        # Check that critic is created correctly
        assert critic.name == "test_critic"
        assert critic.description == "Test critic"
        assert critic.config.system_prompt == "You are a helpful critic"
        assert critic.config.temperature == 0.7
        assert critic.config.max_tokens == 1000
        assert critic.config.min_confidence == 0.8
        assert critic.config.max_attempts == 3
        assert critic.config.cache_size == 100
        assert critic.config.priority == 1
        assert critic.config.cost == 0.1
        assert critic.config.track_performance is True
        assert critic.config.track_errors is True
        
        # Check that state is initialized
        assert critic._state_manager.get("model") == mock_provider
        assert critic._state_manager.get_metadata("component_type") == "critic"
