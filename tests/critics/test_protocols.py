import unittest
import pytest
import inspect
from typing import Any, Dict, List, Protocol

from sifaka.critics.protocols import (
    TextValidator,
    TextImprover,
    TextCritic,
    LLMProvider,
    PromptFactory,
    SyncTextValidator,
    SyncTextImprover,
    SyncTextCritic,
    SyncLLMProvider,
    SyncPromptFactory,
    AsyncTextValidator,
    AsyncTextImprover,
    AsyncTextCritic,
    AsyncLLMProvider,
    AsyncPromptFactory,
    CritiqueResult
)


class TestProtocolDefinitions(unittest.TestCase):
    """Test the protocol definitions themselves."""

    def test_validator_protocol_methods(self):
        """Test TextValidator protocol has the correct methods."""
        # Get the methods defined in the protocol
        methods = [name for name, _ in inspect.getmembers(TextValidator, predicate=inspect.isfunction)]
        self.assertIn('validate', methods)

    def test_improver_protocol_methods(self):
        """Test TextImprover protocol has the correct methods."""
        methods = [name for name, _ in inspect.getmembers(TextImprover, predicate=inspect.isfunction)]
        self.assertIn('improve', methods)

    def test_critic_protocol_methods(self):
        """Test TextCritic protocol has the correct methods."""
        methods = [name for name, _ in inspect.getmembers(TextCritic, predicate=inspect.isfunction)]
        self.assertIn('critique', methods)

    def test_llm_provider_protocol_methods(self):
        """Test LLMProvider protocol has the correct methods."""
        methods = [name for name, _ in inspect.getmembers(LLMProvider, predicate=inspect.isfunction)]
        self.assertIn('generate', methods)

    def test_prompt_factory_protocol_methods(self):
        """Test PromptFactory protocol has the correct methods."""
        methods = [name for name, _ in inspect.getmembers(PromptFactory, predicate=inspect.isfunction)]
        self.assertIn('create_prompt', methods)

    def test_async_validator_protocol_methods(self):
        """Test AsyncTextValidator protocol has the correct methods."""
        methods = [name for name, _ in inspect.getmembers(AsyncTextValidator, predicate=inspect.isfunction)]
        self.assertIn('validate', methods)

    def test_async_improver_protocol_methods(self):
        """Test AsyncTextImprover protocol has the correct methods."""
        methods = [name for name, _ in inspect.getmembers(AsyncTextImprover, predicate=inspect.isfunction)]
        self.assertIn('improve', methods)

    def test_async_critic_protocol_methods(self):
        """Test AsyncTextCritic protocol has the correct methods."""
        methods = [name for name, _ in inspect.getmembers(AsyncTextCritic, predicate=inspect.isfunction)]
        self.assertIn('critique', methods)


class ConcreteValidator:
    """Concrete implementation of TextValidator for testing."""

    def validate(self, text: str) -> bool:
        """Validate text."""
        return len(text) > 0


class ConcreteImprover:
    """Concrete implementation of TextImprover for testing."""

    def improve(self, text: str, feedback: str) -> str:
        """Improve text."""
        return f"{text} (improved based on: {feedback})"


class ConcreteCritic:
    """Concrete implementation of TextCritic for testing."""

    def critique(self, text: str) -> Dict[str, Any]:
        """Critique text."""
        return {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Issue 1"],
            "suggestions": ["Suggestion 1"]
        }


class ConcreteLLMProvider:
    """Concrete implementation of LLMProvider for testing."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text."""
        return f"Generated: {prompt}"


class ConcretePromptFactory:
    """Concrete implementation of PromptFactory for testing."""

    def create_prompt(self, text: str, **kwargs: Any) -> str:
        """Create prompt."""
        return f"Prompt: {text}"


class AsyncConcreteValidator:
    """Async implementation of AsyncTextValidator for testing."""

    async def validate(self, text: str) -> bool:
        """Validate text asynchronously."""
        return len(text) > 0


class AsyncConcreteImprover:
    """Async implementation of AsyncTextImprover for testing."""

    async def improve(self, text: str, feedback: str) -> str:
        """Improve text asynchronously."""
        return f"{text} (improved based on: {feedback})"


class AsyncConcreteCritic:
    """Async implementation of AsyncTextCritic for testing."""

    async def critique(self, text: str) -> Dict[str, Any]:
        """Critique text asynchronously."""
        return {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Issue 1"],
            "suggestions": ["Suggestion 1"]
        }


class AsyncConcreteLLMProvider:
    """Async implementation of AsyncLLMProvider for testing."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text asynchronously."""
        return f"Generated: {prompt}"


class AsyncConcretePromptFactory:
    """Async implementation of AsyncPromptFactory for testing."""

    async def create_prompt(self, text: str, **kwargs: Any) -> str:
        """Create prompt asynchronously."""
        return f"Prompt: {text}"


class CompleteImplementation(TextValidator, TextImprover, TextCritic, LLMProvider, PromptFactory):
    """Class implementing all synchronous protocols for testing."""

    def validate(self, text: str) -> bool:
        """Validate text."""
        return len(text) > 0

    def improve(self, text: str, feedback: str) -> str:
        """Improve text."""
        return f"{text} (improved based on: {feedback})"

    def critique(self, text: str) -> Dict[str, Any]:
        """Critique text."""
        return {
            "score": 0.8,
            "feedback": "Good text",
            "issues": ["Issue 1"],
            "suggestions": ["Suggestion 1"]
        }

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text."""
        return f"Generated: {prompt}"

    def create_prompt(self, text: str, **kwargs: Any) -> str:
        """Create prompt."""
        return f"Prompt: {text}"


class TestProtocolImplementations(unittest.TestCase):
    """Test concrete implementations of protocols."""

    def test_validator_implementation(self):
        """Test concrete validator implementation."""
        validator = ConcreteValidator()
        self.assertIsInstance(validator, TextValidator)
        self.assertIsInstance(validator, SyncTextValidator)
        self.assertTrue(validator.validate("test"))
        self.assertFalse(validator.validate(""))

    def test_improver_implementation(self):
        """Test concrete improver implementation."""
        improver = ConcreteImprover()
        self.assertIsInstance(improver, TextImprover)
        self.assertIsInstance(improver, SyncTextImprover)
        self.assertEqual(improver.improve("text", "feedback"), "text (improved based on: feedback)")

    def test_critic_implementation(self):
        """Test concrete critic implementation."""
        critic = ConcreteCritic()
        self.assertIsInstance(critic, TextCritic)
        self.assertIsInstance(critic, SyncTextCritic)
        result = critic.critique("test")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")
        self.assertEqual(result["issues"], ["Issue 1"])
        self.assertEqual(result["suggestions"], ["Suggestion 1"])

    def test_llm_provider_implementation(self):
        """Test concrete LLM provider implementation."""
        provider = ConcreteLLMProvider()
        self.assertIsInstance(provider, LLMProvider)
        self.assertIsInstance(provider, SyncLLMProvider)
        self.assertEqual(provider.generate("test"), "Generated: test")

    def test_prompt_factory_implementation(self):
        """Test concrete prompt factory implementation."""
        factory = ConcretePromptFactory()
        self.assertIsInstance(factory, PromptFactory)
        self.assertIsInstance(factory, SyncPromptFactory)
        self.assertEqual(factory.create_prompt("test"), "Prompt: test")

    def test_multiple_protocol_implementation(self):
        """Test implementation of multiple protocols."""
        implementation = CompleteImplementation()
        self.assertIsInstance(implementation, TextValidator)
        self.assertIsInstance(implementation, TextImprover)
        self.assertIsInstance(implementation, TextCritic)
        self.assertIsInstance(implementation, LLMProvider)
        self.assertIsInstance(implementation, PromptFactory)
        self.assertTrue(implementation.validate("test"))
        self.assertEqual(implementation.improve("text", "feedback"), "text (improved based on: feedback)")
        self.assertEqual(implementation.generate("test"), "Generated: test")
        self.assertEqual(implementation.create_prompt("test"), "Prompt: test")


@pytest.mark.asyncio
async def test_async_validator():
    """Test async validator implementation."""
    validator = AsyncConcreteValidator()
    assert isinstance(validator, AsyncTextValidator)
    result = await validator.validate("test")
    assert result is True
    result = await validator.validate("")
    assert result is False


@pytest.mark.asyncio
async def test_async_improver():
    """Test async improver implementation."""
    improver = AsyncConcreteImprover()
    assert isinstance(improver, AsyncTextImprover)
    result = await improver.improve("text", "feedback")
    assert result == "text (improved based on: feedback)"


@pytest.mark.asyncio
async def test_async_critic():
    """Test async critic implementation."""
    critic = AsyncConcreteCritic()
    assert isinstance(critic, AsyncTextCritic)
    result = await critic.critique("test")
    assert result["score"] == 0.8
    assert result["feedback"] == "Good text"
    assert result["issues"] == ["Issue 1"]
    assert result["suggestions"] == ["Suggestion 1"]


@pytest.mark.asyncio
async def test_async_llm_provider():
    """Test async LLM provider implementation."""
    provider = AsyncConcreteLLMProvider()
    assert isinstance(provider, AsyncLLMProvider)
    result = await provider.generate("test")
    assert result == "Generated: test"


@pytest.mark.asyncio
async def test_async_prompt_factory():
    """Test async prompt factory implementation."""
    factory = AsyncConcretePromptFactory()
    assert isinstance(factory, AsyncPromptFactory)
    result = await factory.create_prompt("test")
    assert result == "Prompt: test"


def test_critique_result_type():
    """Test CritiqueResult type definition."""
    # Create a valid CritiqueResult
    result: CritiqueResult = {
        "score": 0.8,
        "feedback": "Good text",
        "issues": ["Issue 1"],
        "suggestions": ["Suggestion 1"]
    }

    # Check each field
    assert result["score"] == 0.8
    assert result["feedback"] == "Good text"
    assert result["issues"] == ["Issue 1"]
    assert result["suggestions"] == ["Suggestion 1"]