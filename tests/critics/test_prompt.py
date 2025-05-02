"""Tests for prompt critic functionality."""

import unittest
from unittest.mock import MagicMock, AsyncMock
from typing import Any
import pytest
from pydantic import ValidationError

from sifaka.critics.prompt import (
    PromptCritic,
    PromptCriticConfig,
    LanguageModel,
    DefaultPromptFactory,
    create_prompt_critic,
)


class MockLanguageModel(MagicMock):
    """Mock language model for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize with model name."""
        super().__init__(*args, **kwargs)
        self.model_name = "mock_model"

    def generate(self, prompt: str) -> str:
        """Mock implementation of generate."""
        return "Generated text"

    def invoke(self, prompt: str) -> Any:
        """Mock implementation of invoke."""
        if "critique" in prompt.lower():
            return {
                "score": 0.8,
                "feedback": "Good text",
                "issues": [],
                "suggestions": []
            }
        elif "improve" in prompt.lower():
            return "Improved text"
        else:
            return "Default response"

    async def ainvoke(self, prompt: str) -> Any:
        """Mock async implementation of invoke."""
        return self.invoke(prompt)


class ConcreteLanguageModel:
    """Concrete implementation of the LanguageModel protocol."""

    def __init__(self, model_name="concrete_model"):
        """Initialize with model name."""
        self._model_name = model_name

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        return f"Generated: {prompt}"

    def invoke(self, prompt: str) -> Any:
        """Invoke the model with a prompt."""
        return {"response": prompt}

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name


class TestPromptCriticConfig(unittest.TestCase):
    """Tests for PromptCriticConfig."""

    def test_valid_config(self):
        """Test valid configuration initialization."""
        config = PromptCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="You are an expert editor.",
            temperature=0.7,
            max_tokens=1000
        )
        self.assertEqual(config.name, "test_critic")
        self.assertEqual(config.description, "Test critic")
        self.assertEqual(config.system_prompt, "You are an expert editor.")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 1000)

    def test_invalid_config(self):
        """Test invalid configuration initialization."""
        # Test empty system prompt
        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test",
                description="Test",
                system_prompt=""
            )

        # Test invalid temperature
        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test",
                description="Test",
                temperature=1.5
            )

        # Test invalid max_tokens
        with self.assertRaises(ValueError):
            PromptCriticConfig(
                name="test",
                description="Test",
                max_tokens=0
            )


class TestPromptCritic(unittest.TestCase):
    """Tests for PromptCritic."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockLanguageModel()
        self.config = PromptCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="You are an expert editor.",
            temperature=0.7,
            max_tokens=1000
        )
        self.critic = PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=self.model,
            config=self.config
        )

    def test_critic_initialization(self):
        """Test critic initialization."""
        # Test basic configuration was passed correctly
        self.assertEqual(self.critic.config.name, "test_critic")
        self.assertEqual(self.critic.config.description, "Test critic")
        self.assertEqual(self.critic.config.system_prompt, "You are an expert editor.")
        self.assertEqual(self.critic.config.temperature, 0.7)
        self.assertEqual(self.critic.config.max_tokens, 1000)

        # Verify the model was stored
        self.assertEqual(self.critic._model, self.model)

    def test_critic_initialization_without_model(self):
        """Test critic initialization without model."""
        with self.assertRaises(Exception):  # Pydantic ValidationError
            PromptCritic(
                name="test_critic",
                description="Test critic"
            )

    def test_improve(self):
        """Test text improvement."""
        text = "Test text"
        improved = self.critic.improve(text)
        self.assertEqual(improved, "Improved text")

    def test_improve_with_feedback(self):
        """Test text improvement with feedback."""
        text = "Test text"
        feedback = "Make it better"
        improved = self.critic.improve(text, feedback)
        self.assertEqual(improved, "Improved text")

    def test_improve_invalid_text(self):
        """Test improvement with invalid text."""
        with self.assertRaises(ValueError):
            self.critic.improve("")

        with self.assertRaises(ValueError):
            self.critic.improve("   ")

    def test_critique(self):
        """Test text critique."""
        text = "Test text"
        result = self.critic.critique(text)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good text")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["suggestions"], [])

    def test_critique_invalid_text(self):
        """Test critique with invalid text."""
        with self.assertRaises(ValueError):
            self.critic.critique("")

        with self.assertRaises(ValueError):
            self.critic.critique("   ")

    def test_language_model_protocol(self):
        """Test the LanguageModel protocol implementation."""
        # Test that our mock implements the protocol
        self.assertIsInstance(self.model, LanguageModel)

        # Test protocol methods
        self.assertEqual(self.model.model_name, "mock_model")
        self.assertEqual(self.model.generate("test"), "Generated text")
        self.assertEqual(self.model.invoke("test"), "Default response")


class TestPromptCriticAsync:
    """Tests for async methods of PromptCritic."""

    @pytest.fixture
    def critic(self):
        """Create a critic for testing."""
        model = MockLanguageModel()
        config = PromptCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="You are an expert editor.",
            temperature=0.7,
            max_tokens=1000
        )
        return PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=model,
            config=config
        )

    @pytest.mark.asyncio
    async def test_avalidate(self, critic):
        """Test async validation."""
        # Create a mock for the critique service
        critic._critique_service.avalidate = AsyncMock()
        critic._critique_service.avalidate.return_value = True

        result = await critic.avalidate("Test text")
        assert result is True
        critic._critique_service.avalidate.assert_called_once_with("Test text")

    @pytest.mark.asyncio
    async def test_avalidate_empty_text(self, critic):
        """Test async validation with empty text."""
        with pytest.raises(ValueError):
            await critic.avalidate("")

        with pytest.raises(ValueError):
            await critic.avalidate("   ")

    @pytest.mark.asyncio
    async def test_acritique(self, critic):
        """Test async critique."""
        # Create a mock for the critique service
        expected_result = {
            "score": 0.8,
            "feedback": "Good content",
            "issues": [],
            "suggestions": []
        }
        critic._critique_service.acritique = AsyncMock()
        critic._critique_service.acritique.return_value = expected_result

        result = await critic.acritique("Test text")
        assert result == expected_result
        critic._critique_service.acritique.assert_called_once_with("Test text")

    @pytest.mark.asyncio
    async def test_acritique_empty_text(self, critic):
        """Test async critique with empty text."""
        with pytest.raises(ValueError):
            await critic.acritique("")

        with pytest.raises(ValueError):
            await critic.acritique("   ")

    @pytest.mark.asyncio
    async def test_aimprove(self, critic):
        """Test async improve."""
        # Create a mock for the critique service
        critic._critique_service.aimprove = AsyncMock()
        critic._critique_service.aimprove.return_value = "Improved text"

        result = await critic.aimprove("Test text", "Test feedback")
        assert result == "Improved text"
        critic._critique_service.aimprove.assert_called_once_with("Test text", "Test feedback")

    @pytest.mark.asyncio
    async def test_aimprove_empty_text(self, critic):
        """Test async improve with empty text."""
        with pytest.raises(ValueError):
            await critic.aimprove("", "Test feedback")

        with pytest.raises(ValueError):
            await critic.aimprove("   ", "Test feedback")


class TestDefaultPromptFactory(unittest.TestCase):
    """Tests for DefaultPromptFactory."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = DefaultPromptFactory()

    def test_create_validation_prompt(self):
        """Test validation prompt creation."""
        text = "Test text"
        prompt = self.factory.create_validation_prompt(text)
        self.assertIn(text, prompt)
        self.assertIn("validate", prompt.lower())

    def test_create_critique_prompt(self):
        """Test critique prompt creation."""
        text = "Test text"
        prompt = self.factory.create_critique_prompt(text)
        self.assertIn(text, prompt)
        self.assertIn("critique", prompt.lower())

    def test_create_improvement_prompt(self):
        """Test improvement prompt creation."""
        text = "Test text"
        feedback = "Make it better"
        prompt = self.factory.create_improvement_prompt(text, feedback)
        self.assertIn(text, prompt)
        self.assertIn(feedback, prompt)
        self.assertIn("improve", prompt.lower())


class TestCreatePromptCritic(unittest.TestCase):
    """Tests for create_prompt_critic factory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockLanguageModel()

    def test_create_prompt_critic(self):
        """Test creating a prompt critic with factory function."""
        # Test with custom parameters
        critic = create_prompt_critic(
            llm_provider=self.model,
            name="custom_critic",
            description="Custom critic",
            system_prompt="Custom system prompt",
            temperature=0.5,
            max_tokens=500,
            min_confidence=0.6
        )

        # Verify the critic was created with the right config
        self.assertEqual(critic.config.name, "custom_critic")
        self.assertEqual(critic.config.description, "Custom critic")
        self.assertEqual(critic.config.system_prompt, "Custom system prompt")
        self.assertEqual(critic.config.temperature, 0.5)
        self.assertEqual(critic.config.max_tokens, 500)
        self.assertEqual(critic.config.min_confidence, 0.6)

        # Verify the model was passed
        self.assertEqual(critic._model, self.model)

    def test_create_prompt_critic_with_defaults(self):
        """Test creating a prompt critic with default values."""
        from sifaka.critics.prompt import DEFAULT_SYSTEM_PROMPT

        # Create with just the required parameters
        critic = create_prompt_critic(llm_provider=self.model)

        # Verify default values were used
        self.assertEqual(critic.config.name, "factory_critic")
        self.assertEqual(critic.config.system_prompt, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(critic.config.temperature, 0.7)
        self.assertEqual(critic.config.max_tokens, 1000)
        self.assertEqual(critic.config.min_confidence, 0.7)


class TestDefaultPromptFactoryMethods(unittest.TestCase):
    """Additional tests for DefaultPromptFactory methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockLanguageModel()
        self.factory = DefaultPromptFactory()

    def test_create_critic_static_method(self):
        """Test create_critic static method."""
        # Test with default config
        critic = DefaultPromptFactory.create_critic(self.model)
        self.assertIsInstance(critic, PromptCritic)
        self.assertEqual(critic._model, self.model)

        # Test with custom config
        custom_config = PromptCriticConfig(
            name="custom_config",
            description="Custom config",
            system_prompt="Custom prompt",
            temperature=0.5
        )
        critic = DefaultPromptFactory.create_critic(self.model, custom_config)
        self.assertEqual(critic.config.name, "custom_config")
        self.assertEqual(critic.config.description, "Custom config")
        self.assertEqual(critic.config.system_prompt, "Custom prompt")
        self.assertEqual(critic.config.temperature, 0.5)

    def test_create_with_custom_prompt_static_method(self):
        """Test create_with_custom_prompt static method."""
        # Test with default parameters
        critic = DefaultPromptFactory.create_with_custom_prompt(
            self.model,
            "Custom system prompt"
        )
        self.assertIsInstance(critic, PromptCritic)
        self.assertEqual(critic.config.system_prompt, "Custom system prompt")
        self.assertEqual(critic.config.temperature, 0.7)  # Default value
        self.assertEqual(critic.config.min_confidence, 0.7)  # Default value

        # Test with custom parameters
        critic = DefaultPromptFactory.create_with_custom_prompt(
            self.model,
            "Custom system prompt",
            min_confidence=0.8,
            temperature=0.5
        )
        self.assertEqual(critic.config.system_prompt, "Custom system prompt")
        self.assertEqual(critic.config.temperature, 0.5)
        self.assertEqual(critic.config.min_confidence, 0.8)


class TestLanguageModelProtocol(unittest.TestCase):
    """Tests for LanguageModel protocol."""

    def test_concrete_implementation(self):
        """Test a concrete implementation of LanguageModel protocol."""
        model = ConcreteLanguageModel()

        # Verify it's recognized as implementing the protocol
        self.assertIsInstance(model, LanguageModel)

        # Test the methods
        self.assertEqual(model.model_name, "concrete_model")
        self.assertEqual(model.generate("test"), "Generated: test")
        self.assertEqual(model.invoke("test"), {"response": "test"})

    def test_protocol_structural_typing(self):
        """Test structural typing of the protocol."""
        # Create a structural match without inheriting
        class StructuralMatch:
            def generate(self, prompt: str) -> str:
                return "text"

            def invoke(self, prompt: str) -> Any:
                return {}

            @property
            def model_name(self) -> str:
                return "name"

        # Should be recognized as implementing the protocol
        model = StructuralMatch()
        self.assertIsInstance(model, LanguageModel)


class TestPromptCriticEdgeCases(unittest.TestCase):
    """Tests for edge cases in PromptCritic."""

    def test_error_handling_in_init(self):
        """Test error handling in __init__ method."""
        # Test the error path when llm_provider is None and we create a ValidationError
        with self.assertRaises(ValidationError):
            critic = PromptCritic(name="test", description="Test critic")


@pytest.mark.asyncio
class TestComplexAsyncBehavior:
    """Tests for complex async behaviors."""

    @pytest.fixture
    def critic(self):
        """Create a critic for testing."""
        model = MagicMock()
        # Remove ainvoke to test fallback
        model.ainvoke = None

        config = PromptCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="You are an expert editor.",
            temperature=0.7,
            max_tokens=1000
        )
        critic = PromptCritic(
            name="test_critic",
            description="Test critic",
            llm_provider=model,
            config=config
        )

        # Mock the critique service methods with AsyncMock
        critic._critique_service.avalidate = AsyncMock()
        critic._critique_service.avalidate.return_value = True

        critic._critique_service.acritique = AsyncMock()
        critic._critique_service.acritique.return_value = {
            "score": 0.9,
            "feedback": "Excellent",
            "issues": [],
            "suggestions": []
        }

        critic._critique_service.aimprove = AsyncMock()
        critic._critique_service.aimprove.return_value = "Improved text"

        return critic

    async def test_avalidate_with_sync_fallback(self, critic):
        """Test avalidate when only synchronous methods are available."""
        result = await critic.avalidate("Test text")
        assert result is True
        critic._critique_service.avalidate.assert_called_once_with("Test text")

    async def test_acritique_with_sync_fallback(self, critic):
        """Test acritique when only synchronous methods are available."""
        result = await critic.acritique("Test text")
        assert result["score"] == 0.9
        critic._critique_service.acritique.assert_called_once_with("Test text")

    async def test_aimprove_with_sync_fallback(self, critic):
        """Test aimprove when only synchronous methods are available."""
        result = await critic.aimprove("Test text", "Test feedback")
        assert result == "Improved text"
        critic._critique_service.aimprove.assert_called_once_with("Test text", "Test feedback")


if __name__ == "__main__":
    unittest.main()