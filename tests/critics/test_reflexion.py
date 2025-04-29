"""Tests for reflexion-based critics."""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.critics.reflexion import ReflexionCritic, ReflexionCriticConfig, ReflexionPromptFactory


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = MagicMock()
    mock.invoke.return_value = {
        "score": 0.8,
        "feedback": "Test feedback",
        "issues": ["Test issue"],
        "suggestions": ["Test suggestion"],
    }
    return mock


@pytest.fixture
def reflexion_critic(mock_llm):
    """Create a ReflexionCritic instance with a mock LLM."""
    return ReflexionCritic(
        name="test_reflexion_critic",
        description="A test reflexion critic",
        llm_provider=mock_llm,
        config=ReflexionCriticConfig(
            name="test_reflexion_critic",
            description="A test reflexion critic",
            system_prompt="You are a reflexion critic",
            memory_buffer_size=3,
            reflection_depth=1,
        ),
    )


def test_reflexion_critic_initialization(reflexion_critic):
    """Test that ReflexionCritic can be initialized with valid parameters."""
    assert reflexion_critic.config.name == "test_reflexion_critic"
    assert reflexion_critic.config.description == "A test reflexion critic"
    assert reflexion_critic.config.system_prompt == "You are a reflexion critic"
    assert reflexion_critic.config.memory_buffer_size == 3
    assert reflexion_critic.config.reflection_depth == 1
    assert reflexion_critic._memory_buffer == []


def test_reflexion_critic_config_validation():
    """Test that ReflexionCriticConfig validates parameters correctly."""
    # Test valid config
    config = ReflexionCriticConfig(
        name="test",
        description="test",
        system_prompt="test",
        memory_buffer_size=5,
        reflection_depth=1,
    )
    assert config.memory_buffer_size == 5
    assert config.reflection_depth == 1

    # Test invalid memory_buffer_size
    with pytest.raises(ValueError):
        ReflexionCriticConfig(
            name="test",
            description="test",
            system_prompt="test",
            memory_buffer_size=-1,  # Invalid negative value
        )

    # Test invalid reflection_depth
    with pytest.raises(ValueError):
        ReflexionCriticConfig(
            name="test",
            description="test",
            system_prompt="test",
            reflection_depth=0,  # Invalid zero value
        )


def test_reflexion_critic_critique(reflexion_critic, mock_llm):
    """Test that ReflexionCritic's critique method works as expected."""
    result = reflexion_critic.critique("test text")
    assert result.score == 0.8
    assert result.feedback == "Test feedback"
    assert result.issues == ["Test issue"]
    assert result.suggestions == ["Test suggestion"]
    mock_llm.invoke.assert_called_once()


def test_reflexion_critic_validate(reflexion_critic, mock_llm):
    """Test that ReflexionCritic's validate method works as expected."""
    mock_llm.invoke.return_value = {"valid": True}
    result = reflexion_critic.validate("test text")
    assert result is True
    mock_llm.invoke.assert_called_once()

    # Test with string response
    mock_llm.invoke.reset_mock()
    mock_llm.invoke.return_value = "VALID: true"
    result = reflexion_critic.validate("test text")
    assert result is True
    mock_llm.invoke.assert_called_once()


def test_reflexion_critic_improve(reflexion_critic, mock_llm):
    """Test that ReflexionCritic's improve method works as expected."""
    mock_llm.invoke.return_value = {"improved_text": "Improved text"}
    
    # First call to improve
    result = reflexion_critic.improve("test text", [{"rule_name": "test", "message": "test message"}])
    assert result == "Improved text"
    
    # Second call to invoke should be for reflection generation
    assert mock_llm.invoke.call_count == 2


def test_reflexion_critic_memory_buffer(reflexion_critic, mock_llm):
    """Test that ReflexionCritic's memory buffer works as expected."""
    # Set up mock to return a reflection
    mock_llm.invoke.side_effect = [
        {"improved_text": "Improved text 1"},
        "REFLECTION: First reflection",
        {"improved_text": "Improved text 2"},
        "REFLECTION: Second reflection",
        {"improved_text": "Improved text 3"},
        "REFLECTION: Third reflection",
        {"improved_text": "Improved text 4"},
        "REFLECTION: Fourth reflection",
    ]
    
    # Make multiple improve calls to fill the memory buffer
    reflexion_critic.improve("text 1", [{"rule_name": "test", "message": "message 1"}])
    assert len(reflexion_critic._memory_buffer) == 1
    assert "First reflection" in reflexion_critic._memory_buffer[0]
    
    reflexion_critic.improve("text 2", [{"rule_name": "test", "message": "message 2"}])
    assert len(reflexion_critic._memory_buffer) == 2
    
    reflexion_critic.improve("text 3", [{"rule_name": "test", "message": "message 3"}])
    assert len(reflexion_critic._memory_buffer) == 3
    
    # This should cause the oldest reflection to be removed
    reflexion_critic.improve("text 4", [{"rule_name": "test", "message": "message 4"}])
    assert len(reflexion_critic._memory_buffer) == 3
    assert "First reflection" not in reflexion_critic._memory_buffer[0]
    assert "Fourth reflection" in reflexion_critic._memory_buffer[2]


def test_reflexion_prompt_factory():
    """Test that ReflexionPromptFactory creates prompts correctly."""
    factory = ReflexionPromptFactory()
    
    # Test validation prompt
    validation_prompt = factory.create_validation_prompt("test text")
    assert "test text" in validation_prompt
    assert "VALID:" in validation_prompt
    
    # Test critique prompt
    critique_prompt = factory.create_critique_prompt("test text")
    assert "test text" in critique_prompt
    assert "SCORE:" in critique_prompt
    
    # Test improvement prompt without reflections
    improvement_prompt = factory.create_improvement_prompt("test text", "test feedback")
    assert "test text" in improvement_prompt
    assert "test feedback" in improvement_prompt
    assert "PREVIOUS REFLECTIONS" not in improvement_prompt
    
    # Test improvement prompt with reflections
    improvement_prompt = factory.create_improvement_prompt(
        "test text", "test feedback", ["reflection 1", "reflection 2"]
    )
    assert "test text" in improvement_prompt
    assert "test feedback" in improvement_prompt
    assert "PREVIOUS REFLECTIONS" in improvement_prompt
    assert "reflection 1" in improvement_prompt
    assert "reflection 2" in improvement_prompt
    
    # Test reflection prompt
    reflection_prompt = factory.create_reflection_prompt(
        "original text", "feedback", "improved text"
    )
    assert "original text" in reflection_prompt
    assert "feedback" in reflection_prompt
    assert "improved text" in reflection_prompt
    assert "REFLECTION:" in reflection_prompt


def test_create_reflexion_critic_factory_function(mock_llm):
    """Test the create_reflexion_critic factory function."""
    from sifaka.critics.reflexion import create_reflexion_critic
    
    critic = create_reflexion_critic(
        model=mock_llm,
        name="factory_critic",
        description="Created with factory function",
        memory_buffer_size=10,
        reflection_depth=2,
    )
    
    assert critic.config.name == "factory_critic"
    assert critic.config.description == "Created with factory function"
    assert critic.config.memory_buffer_size == 10
    assert critic.config.reflection_depth == 2
    assert isinstance(critic, ReflexionCritic)


def test_reflexion_critic_empty_text(reflexion_critic):
    """Test that ReflexionCritic handles empty text input."""
    with pytest.raises(ValueError):
        reflexion_critic.critique("")
    
    with pytest.raises(ValueError):
        reflexion_critic.validate("")
    
    with pytest.raises(ValueError):
        reflexion_critic.improve("", [])
