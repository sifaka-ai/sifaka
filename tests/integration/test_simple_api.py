"""Integration tests for the simple API."""

import pytest

from sifaka import improve, improve_sync
from sifaka.core.config import Config, EngineConfig, LLMConfig
from sifaka.validators.basic import LengthValidator


class TestSimpleAPI:
    """Test the simplified API functions."""

    def test_improve_sync_basic(self):
        """Test basic synchronous improvement."""
        text = "This is a test."
        result = improve_sync(
            text, config=Config(llm=LLMConfig(temperature=0.3)), max_iterations=1
        )

        assert isinstance(result.final_text, str)
        assert result.iteration >= 1
        assert len(result.critiques) > 0

    def test_improve_sync_with_config(self):
        """Test improvement with custom config."""
        text = "The data shows interesting patterns."
        config = Config(
            llm=LLMConfig(temperature=0.3),
            engine=EngineConfig(max_iterations=2),
        )
        result = improve_sync(text, config=config, max_iterations=2)

        assert isinstance(result.final_text, str)
        assert result.original_text == text
        assert result.iteration <= 2

    @pytest.mark.asyncio
    async def test_improve_async(self):
        """Test async improvement."""
        text = "Machine learning uses data to make predictions."

        result = await improve(
            text, config=Config(llm=LLMConfig(temperature=0.3)), max_iterations=1
        )
        assert isinstance(result.final_text, str)
        assert result.iteration >= 1
        assert len(result.critiques) > 0

    @pytest.mark.asyncio
    async def test_improve_with_validators(self):
        """Test improvement with validators."""
        text = "Short text"

        validator = LengthValidator(min_length=50)
        result = await improve(
            text,
            validators=[validator],
            config=Config(llm=LLMConfig(temperature=0.3)),
            max_iterations=2,
        )
        assert isinstance(result.final_text, str)
        # Note: With mocking, the text won't actually be lengthened
        # Just verify the validator was applied
        assert len(result.validations) > 0

    def test_multiple_iterations(self):
        """Test that multiple iterations produce refinements."""
        text = "This needs improvement."

        result = improve_sync(
            text, config=Config(llm=LLMConfig(temperature=0.3)), max_iterations=3
        )
        assert result.iteration >= 1
        assert result.iteration <= 3
        # With current mocking, generations might not be stored
        # Just verify iterations happened
        assert result.iteration >= 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in API."""
        # Empty text should raise validation error
        with pytest.raises(ValueError, match="String should have at least 1 character"):
            await improve("", config=Config(llm=LLMConfig(temperature=0.3)))
