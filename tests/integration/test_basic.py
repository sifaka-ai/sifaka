"""Basic tests for Sifaka functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from sifaka import improve
from sifaka.core.config import Config, LLMConfig
from sifaka.core.models import SifakaResult
from sifaka.validators import ContentValidator, LengthValidator


class TestBasicFunctionality:
    """Test basic Sifaka functionality."""

    @pytest.mark.asyncio
    async def test_improve_basic(self):
        """Test basic improve function."""
        # Mock OpenAI responses
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[
            0
        ].message.content = "This is improved text about renewable energy benefits with much better clarity and structure."

        with patch("openai.AsyncOpenAI") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            result = await improve("Write about renewable energy")

            assert isinstance(result, SifakaResult)
            assert result.original_text == "Write about renewable energy"
            assert result.final_text is not None
            assert result.iteration >= 0


class TestValidators:
    """Test validator functionality."""

    @pytest.mark.asyncio
    async def test_length_validator(self):
        """Test length validator."""
        validator = LengthValidator(min_length=10, max_length=100)
        result = SifakaResult(original_text="test", final_text="test")

        # Test short text
        validation = await validator.validate("short", result)
        assert validation.validator == "length"
        assert not validation.passed

        # Test good length
        validation = await validator.validate(
            "This is a good length text for testing", result
        )
        assert validation.passed

    @pytest.mark.asyncio
    async def test_content_validator(self):
        """Test content validator."""
        validator = ContentValidator(
            required_terms=["energy", "renewable"], forbidden_terms=["fossil", "coal"]
        )
        result = SifakaResult(original_text="test", final_text="test")

        # Test missing required terms
        validation = await validator.validate("This is about electricity", result)
        assert not validation.passed

        # Test forbidden terms
        validation = await validator.validate("Coal energy is renewable energy", result)
        assert not validation.passed

        # Test good content
        validation = await validator.validate(
            "Renewable energy sources are sustainable", result
        )
        assert validation.passed


class TestConfig:
    """Test configuration validation."""

    def test_config_validation(self):
        """Test config validation catches errors."""
        from sifaka.core.config import EngineConfig

        # Test invalid max_iterations
        with pytest.raises(ValueError):
            Config(engine=EngineConfig(max_iterations=0))

        # Test invalid temperature
        with pytest.raises(ValueError):
            Config(llm=LLMConfig(temperature=3.0))

        # Valid config should work
        config = Config(
            engine=EngineConfig(max_iterations=3), llm=LLMConfig(temperature=0.5)
        )
        assert config.engine.max_iterations == 3
        assert config.llm.temperature == 0.5

    def test_config_defaults(self):
        """Test default configuration values."""
        config = Config()
        assert config.llm.model == "gpt-4o-mini"
        assert config.llm.temperature == 0.7
        assert config.engine.max_iterations == 3
        assert config.critic.critics == ["reflexion"]


class TestMemoryBounds:
    """Test memory-bounded collections."""

    def test_generation_bounds(self):
        """Test that generations are memory-bounded."""
        result = SifakaResult(original_text="test", final_text="test")

        # Add more than max generations
        for i in range(15):
            result.add_generation(f"text {i}", "gpt-4")

        # Should only keep last 10
        assert len(result.generations) == 10
        assert result.generations[0].text == "text 5"  # Oldest kept
        assert list(result.generations)[-1].text == "text 14"  # Newest

    def test_validation_bounds(self):
        """Test that validations are memory-bounded."""
        result = SifakaResult(original_text="test", final_text="test")

        # Add more than max validations
        for i in range(25):
            result.add_validation(f"validator{i}", i % 2 == 0)

        # Should only keep last 20
        assert len(result.validations) == 20

    def test_critique_bounds(self):
        """Test that critiques are memory-bounded."""
        result = SifakaResult(original_text="test", final_text="test")

        # Add more than max critiques
        for i in range(25):
            result.add_critique(f"critic{i}", f"feedback {i}", [f"suggestion {i}"])

        # Should only keep last 20
        assert len(result.critiques) == 20


if __name__ == "__main__":
    pytest.main([__file__])
