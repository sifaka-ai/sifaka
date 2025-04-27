"""Tests for prompt-based critics."""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock
from pydantic import ValidationError
from sifaka.critics.prompt import PromptCritic, DefaultPromptFactory
from sifaka.critics.protocols import LLMProvider, PromptFactory


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, should_fail: bool = False, response: Dict[str, Any] = None):
        self.should_fail = should_fail
        self.invoke_called = 0
        self.ainvoke_called = 0
        self.last_prompt = None

        self.response = response or {
            "valid": True,
            "score": 0.8,
            "feedback": "Test feedback",
            "issues": ["Test issue"],
            "suggestions": ["Test suggestion"],
            "improved_text": "Improved text",
        }

    def invoke(self, prompt: str) -> Dict[str, Any]:
        self.invoke_called += 1
        self.last_prompt = prompt
        if self.should_fail:
            raise ValueError("LLM invocation failed")
        return self.response

    async def ainvoke(self, prompt: str) -> Dict[str, Any]:
        self.ainvoke_called += 1
        self.last_prompt = prompt
        if self.should_fail:
            raise ValueError("LLM invocation failed")
        return self.response


class CustomPromptFactory(PromptFactory):
    """Custom prompt factory for testing."""

    def __init__(self):
        self.create_validation_called = 0
        self.create_critique_called = 0
        self.create_improvement_called = 0

    def create_validation_prompt(self, text: str) -> str:
        self.create_validation_called += 1
        return f"Custom validation: {text}"

    def create_critique_prompt(self, text: str) -> str:
        self.create_critique_called += 1
        return f"Custom critique: {text}"

    def create_improvement_prompt(self, text: str, feedback: str) -> str:
        self.create_improvement_called += 1
        return f"Custom improvement: {text} with {feedback}"


@pytest.fixture
def llm_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def prompt_factory():
    """Create a default prompt factory."""
    return DefaultPromptFactory()


@pytest.fixture
def custom_prompt_factory():
    """Create a custom prompt factory."""
    return CustomPromptFactory()


@pytest.fixture
def prompt_critic(llm_provider, prompt_factory):
    """Create a PromptCritic instance with mocks."""
    return PromptCritic(
        name="test_prompt_critic",
        description="A test prompt critic",
        llm_provider=llm_provider,
        prompt_factory=prompt_factory,
    )


# Initialization tests
def test_prompt_critic_initialization(prompt_critic, llm_provider, prompt_factory):
    """Test that PromptCritic can be initialized with valid parameters."""
    assert prompt_critic.name == "test_prompt_critic"
    assert prompt_critic.description == "A test prompt critic"
    assert prompt_critic.llm_provider == llm_provider
    assert isinstance(prompt_critic.prompt_factory, DefaultPromptFactory)


def test_prompt_critic_missing_llm():
    """Test that PromptCritic requires an LLM provider."""
    with pytest.raises(ValidationError) as exc_info:
        PromptCritic(name="test_critic", description="A test critic")
    assert "Field required" in str(exc_info.value)


# Validation tests
def test_prompt_critic_validate(prompt_critic, llm_provider):
    """Test that PromptCritic's validate method works as expected."""
    result = prompt_critic.validate("test text")
    assert result is True
    assert llm_provider.invoke_called == 1
    assert "Validate" in llm_provider.last_prompt
    assert "test text" in llm_provider.last_prompt


def test_prompt_critic_validate_failure(llm_provider, prompt_factory):
    """Test validation with failing LLM."""
    failing_llm = MockLLMProvider(should_fail=True)
    critic = PromptCritic(
        name="test_critic",
        description="A test critic",
        llm_provider=failing_llm,
        prompt_factory=prompt_factory,
    )

    with pytest.raises(ValueError, match="LLM invocation failed"):
        critic.validate("test text")
    assert failing_llm.invoke_called == 1


def test_prompt_critic_validate_invalid_response(llm_provider, prompt_factory):
    """Test validation with invalid LLM response."""
    invalid_llm = MockLLMProvider(response={"valid": False})
    critic = PromptCritic(
        name="test_critic",
        description="A test critic",
        llm_provider=invalid_llm,
        prompt_factory=prompt_factory,
    )

    assert critic.validate("test text") is False


# Critique tests
def test_prompt_critic_critique(prompt_critic, llm_provider):
    """Test that PromptCritic's critique method works as expected."""
    result = prompt_critic.critique("test text")
    assert isinstance(result, dict)
    assert result["score"] == 0.8
    assert result["feedback"] == "Test feedback"
    assert result["issues"] == ["Test issue"]
    assert result["suggestions"] == ["Test suggestion"]
    assert llm_provider.invoke_called == 1
    assert "Critique" in llm_provider.last_prompt
    assert "test text" in llm_provider.last_prompt


def test_prompt_critic_critique_custom_response(llm_provider, prompt_factory):
    """Test critique with custom LLM response."""
    custom_response = {
        "score": 0.9,
        "feedback": "Custom feedback",
        "issues": [],
        "suggestions": ["Perfect as is"],
    }
    custom_llm = MockLLMProvider(response=custom_response)
    critic = PromptCritic(
        name="test_critic",
        description="A test critic",
        llm_provider=custom_llm,
        prompt_factory=prompt_factory,
    )

    result = critic.critique("perfect text")
    assert result == custom_response


# Improvement tests
def test_prompt_critic_improve(prompt_critic, llm_provider):
    """Test that PromptCritic's improve method works as expected."""
    result = prompt_critic.improve("test text", "test feedback")
    assert result == "Improved text"
    assert llm_provider.invoke_called == 1
    assert "Improve" in llm_provider.last_prompt
    assert "test text" in llm_provider.last_prompt
    assert "test feedback" in llm_provider.last_prompt


def test_prompt_critic_improve_failure(llm_provider, prompt_factory):
    """Test improvement with failing LLM."""
    failing_llm = MockLLMProvider(should_fail=True)
    critic = PromptCritic(
        name="test_critic",
        description="A test critic",
        llm_provider=failing_llm,
        prompt_factory=prompt_factory,
    )

    with pytest.raises(ValueError, match="LLM invocation failed"):
        critic.improve("test text", "test feedback")
    assert failing_llm.invoke_called == 1


# Async tests
@pytest.mark.asyncio
async def test_prompt_critic_async_methods(prompt_critic, llm_provider):
    """Test async versions of PromptCritic methods."""
    # Test async validate
    result = await prompt_critic.avalidate("test text")
    assert result is True
    assert llm_provider.ainvoke_called == 1
    assert "Validate" in llm_provider.last_prompt
    llm_provider.last_prompt = None

    # Test async critique
    result = await prompt_critic.acritique("test text")
    assert isinstance(result, dict)
    assert result["score"] == 0.8
    assert llm_provider.ainvoke_called == 2
    assert "Critique" in llm_provider.last_prompt
    llm_provider.last_prompt = None

    # Test async improve
    result = await prompt_critic.aimprove("test text", "test feedback")
    assert result == "Improved text"
    assert llm_provider.ainvoke_called == 3
    assert "Improve" in llm_provider.last_prompt


@pytest.mark.asyncio
async def test_prompt_critic_async_failures(llm_provider, prompt_factory):
    """Test async methods with failures."""
    failing_llm = MockLLMProvider(should_fail=True)
    critic = PromptCritic(
        name="test_critic",
        description="A test critic",
        llm_provider=failing_llm,
        prompt_factory=prompt_factory,
    )

    # Test async validate failure
    with pytest.raises(ValueError, match="LLM invocation failed"):
        await critic.avalidate("test text")
    assert failing_llm.ainvoke_called == 1

    # Test async critique failure
    with pytest.raises(ValueError, match="LLM invocation failed"):
        await critic.acritique("test text")
    assert failing_llm.ainvoke_called == 2

    # Test async improve failure
    with pytest.raises(ValueError, match="LLM invocation failed"):
        await critic.aimprove("test text", "test feedback")
    assert failing_llm.ainvoke_called == 3


# Prompt factory tests
def test_default_prompt_factory():
    """Test DefaultPromptFactory methods."""
    factory = DefaultPromptFactory()

    # Test validation prompt
    validation_prompt = factory.create_validation_prompt("test text")
    assert "test text" in validation_prompt
    assert "Validate" in validation_prompt

    # Test critique prompt
    critique_prompt = factory.create_critique_prompt("test text")
    assert "test text" in critique_prompt
    assert "Critique" in critique_prompt

    # Test improvement prompt
    improvement_prompt = factory.create_improvement_prompt("test text", "test feedback")
    assert "test text" in improvement_prompt
    assert "test feedback" in improvement_prompt
    assert "Improve" in improvement_prompt


def test_prompt_critic_with_custom_factory(llm_provider, custom_prompt_factory):
    """Test PromptCritic with a custom prompt factory."""
    critic = PromptCritic(
        name="test_critic",
        description="A test critic",
        llm_provider=llm_provider,
        prompt_factory=custom_prompt_factory,
    )

    # Test validation
    critic.validate("test text")
    assert custom_prompt_factory.create_validation_called == 1
    assert llm_provider.last_prompt == "Custom validation: test text"

    # Test critique
    critic.critique("test text")
    assert custom_prompt_factory.create_critique_called == 1
    assert llm_provider.last_prompt == "Custom critique: test text"

    # Test improvement
    critic.improve("test text", "test feedback")
    assert custom_prompt_factory.create_improvement_called == 1
    assert llm_provider.last_prompt == "Custom improvement: test text with test feedback"
