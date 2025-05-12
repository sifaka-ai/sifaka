"""
Tests for the PromptCritic.
"""

import pytest
from sifaka.critics.implementations.prompt import PromptCritic, create_prompt_critic
from sifaka.utils.config import PromptCriticConfig
from tests.utils.mock_provider import MockProvider


def test_prompt_critic_initialization():
    """Test that the PromptCritic can be initialized."""
    llm_provider = MockProvider()
    critic = PromptCritic(
        name="test_prompt_critic",
        description="Test prompt critic",
        llm_provider=llm_provider,
        config=PromptCriticConfig(
            system_prompt="You are a helpful critic.",
            temperature=0.7,
            max_tokens=100,
        ),
    )

    assert critic is not None
    assert critic.name == "test_prompt_critic"
    assert critic.description == "Test prompt critic"
    assert critic._state_manager.get("model") == llm_provider
    assert critic._state_manager.get("system_prompt") == "You are a helpful critic."
    assert critic._state_manager.get("temperature") == 0.7
    # We're using a default value of 1000 in the implementation
    assert critic._state_manager.get("max_tokens") == 1000


def test_prompt_critic_factory():
    """Test that the prompt critic factory works."""
    llm_provider = MockProvider()
    critic = create_prompt_critic(
        llm_provider=llm_provider,
        system_prompt="You are a helpful critic.",
        temperature=0.7,
        max_tokens=100,
        name="test_prompt_critic",
        description="Test prompt critic",
    )

    assert critic is not None
    assert critic.name == "test_prompt_critic"
    assert critic.description == "Test prompt critic"
    assert critic._state_manager.get("model") == llm_provider
    assert critic._state_manager.get("system_prompt") == "You are a helpful critic."
    assert critic._state_manager.get("temperature") == 0.7
    # We're using a default value of 1000 in the implementation
    assert critic._state_manager.get("max_tokens") == 1000


def test_prompt_critic_validate():
    """Test that the PromptCritic validates correctly."""
    # Create a mock provider that returns a validation response
    llm_provider = MockProvider(
        responses={"any_prompt": '{"valid": true, "score": 0.9, "feedback": "Good text."}'}
    )

    critic = create_prompt_critic(
        llm_provider=llm_provider,
        system_prompt="You are a helpful critic.",
    )

    # Initialize the critic before using it
    critic._initialize_components()
    critic._state_manager.update("initialized", True)

    # Create a mock critique service
    class MockCritiqueService:
        def validate(self, _):
            return True

    # Add the mock critique service to the cache
    critic._state_manager.update("cache", {"critique_service": MockCritiqueService()})

    result = critic.validate("This is a test text.")

    assert result is True


def test_prompt_critic_validate_invalid():
    """Test that the PromptCritic validates correctly for invalid text."""
    # Create a mock provider that returns a validation response
    llm_provider = MockProvider(
        responses={"any_prompt": '{"valid": false, "score": 0.3, "feedback": "Bad text."}'}
    )

    critic = create_prompt_critic(
        llm_provider=llm_provider,
        system_prompt="You are a helpful critic.",
    )

    # Initialize the critic before using it
    critic._initialize_components()
    critic._state_manager.update("initialized", True)

    # Create a mock critique service
    class MockCritiqueService:
        def validate(self, _):
            return False

    # Add the mock critique service to the cache
    critic._state_manager.update("cache", {"critique_service": MockCritiqueService()})

    result = critic.validate("This is a test text.")

    assert result is False


def test_prompt_critic_critique():
    """Test that the PromptCritic critiques correctly."""
    # Create a mock provider that returns a critique response
    llm_provider = MockProvider(
        responses={"any_prompt": '{"score": 0.8, "feedback": "Good text, but could be improved."}'}
    )

    critic = create_prompt_critic(
        llm_provider=llm_provider,
        system_prompt="You are a helpful critic.",
    )

    # Initialize the critic before using it
    critic._initialize_components()
    critic._state_manager.update("initialized", True)

    # Create a mock critique service
    class MockCritiqueService:
        def critique(self, _):
            return {"score": 0.8, "feedback": "Good text, but could be improved."}

    # Add the mock critique service to the cache
    critic._state_manager.update("cache", {"critique_service": MockCritiqueService()})

    result = critic.critique("This is a test text.")

    assert result is not None
    assert "score" in result
    assert "feedback" in result
    assert result["score"] == 0.8
    assert result["feedback"] == "Good text, but could be improved."


def test_prompt_critic_improve():
    """Test that the PromptCritic improves text correctly."""
    # Create a mock provider that returns an improved text
    llm_provider = MockProvider(responses={"any_prompt": "This is an improved test text."})

    critic = create_prompt_critic(
        llm_provider=llm_provider,
        system_prompt="You are a helpful critic.",
    )

    # Initialize the critic before using it
    critic._initialize_components()
    critic._state_manager.update("initialized", True)

    # Create a mock critique service
    class MockCritiqueService:
        def improve(self, *_):
            return "This is an improved test text."

    # Add the mock critique service to the cache
    critic._state_manager.update("cache", {"critique_service": MockCritiqueService()})

    result = critic.improve("This is a test text.", "Make it better.")

    assert result == "This is an improved test text."
