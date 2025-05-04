"""
Tests for state management in critics.
"""

import pytest
from unittest.mock import MagicMock

from sifaka.critics.core import CriticCore
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.reflexion import ReflexionCritic, ReflexionCriticConfig


def test_critic_core_state_management():
    """Test state management in CriticCore."""
    # Create mock components
    mock_llm = MagicMock()
    mock_config = MagicMock()
    
    # Create critic
    critic = CriticCore(config=mock_config, llm_provider=mock_llm)
    
    # Check state initialization
    assert critic._state is not None
    
    # Get state
    state = critic._state.get_state()
    
    # Check state properties
    assert state.initialized
    assert state.model is mock_llm
    assert "critique_service" in state.cache


def test_prompt_critic_state_management():
    """Test state management in PromptCritic."""
    # Create mock components
    mock_llm = MagicMock()
    
    # Create critic
    critic = PromptCritic(llm_provider=mock_llm)
    
    # Check state initialization
    assert critic._state is not None
    
    # Get state
    state = critic._state.get_state()
    
    # Check state properties
    assert state.initialized
    assert state.model is mock_llm
    assert "critique_service" in state.cache


def test_reflexion_critic_state_management():
    """Test state management in ReflexionCritic."""
    # Create mock components
    mock_llm = MagicMock()
    config = ReflexionCriticConfig(
        name="test_reflexion",
        description="Test reflexion critic",
        system_prompt="You are a test critic",
        memory_buffer_size=3,
    )
    
    # Create critic
    critic = ReflexionCritic(llm_provider=mock_llm, config=config)
    
    # Check state initialization
    assert critic._state is not None
    
    # Get state
    state = critic._state.get_state()
    
    # Check state properties
    assert state.initialized
    assert state.model is mock_llm
    assert "critique_service" in state.cache


def test_critic_methods_use_state():
    """Test that critic methods use state properly."""
    # Create mock components
    mock_llm = MagicMock()
    mock_critique_service = MagicMock()
    mock_critique_service.validate.return_value = True
    mock_critique_service.critique.return_value = {"score": 0.8, "feedback": "Good job!"}
    mock_critique_service.improve.return_value = "Improved text"
    
    # Create critic
    critic = CriticCore(config=MagicMock(), llm_provider=mock_llm)
    
    # Replace critique service in state
    state = critic._state.get_state()
    state.cache["critique_service"] = mock_critique_service
    
    # Test methods
    assert critic.validate("Test text") is True
    assert critic.critique("Test text")["score"] == 0.8
    assert critic.improve("Test text", "Make it better") == "Improved text"
    
    # Check that methods used the service from state
    mock_critique_service.validate.assert_called_once_with("Test text")
    mock_critique_service.critique.assert_called_once_with("Test text")
    mock_critique_service.improve.assert_called_once_with("Test text", "Make it better")


def test_reflexion_critic_cache():
    """Test that ReflexionCritic uses state cache properly."""
    # Create mock components
    mock_llm = MagicMock()
    mock_memory_manager = MagicMock()
    mock_memory_manager.get_memory.return_value = ["Previous reflection"]
    
    # Create critic
    critic = ReflexionCritic(llm_provider=mock_llm)
    
    # Replace memory manager in state
    state = critic._state.get_state()
    state.memory_manager = mock_memory_manager
    
    # Test method that uses cache
    reflections = critic._get_relevant_reflections()
    
    # Check result and cache
    assert reflections == ["Previous reflection"]
    assert state.cache["cache"]["last_reflections"] == ["Previous reflection"]
    
    # Test violations to feedback
    violations = [
        {"rule_name": "Clarity", "message": "Text is unclear"},
        {"rule_name": "Grammar", "message": "Grammar issues found"},
    ]
    feedback = critic._violations_to_feedback(violations)
    
    # Check that violations were stored in cache
    assert "last_violations" in state.cache["cache"]
    assert state.cache["cache"]["last_violations"] == violations
