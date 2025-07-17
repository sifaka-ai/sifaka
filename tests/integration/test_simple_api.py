"""Integration tests for the simple API."""

import pytest

from sifaka import Config, improve, improve_sync
from sifaka.validators.basic import LengthValidator


class TestSimpleAPI:
    """Test the simplified API functions."""

    def test_improve_sync_basic(self):
        """Test basic synchronous improvement."""
        text = "This is a test."
        result = improve_sync(text, config=Config(temperature=0.3), max_iterations=1)

        assert isinstance(result.final_text, str)
        assert result.iteration >= 1
        assert len(result.critiques) > 0

    def test_improve_sync_with_config(self):
        """Test improvement with custom config."""
        text = "The data shows interesting patterns."
        config = Config(
            temperature=0.3,
            max_iterations=2,
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
            text,
            config=Config(temperature=0.3),
            max_iterations=1,
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
            config=Config(temperature=0.3),
            max_iterations=2,
        )

        assert isinstance(result.final_text, str)
        assert len(result.final_text) >= 50  # Should meet length requirement

    def test_multiple_iterations(self):
        """Test that multiple iterations produce refinements."""
        text = "This needs improvement."

        result = improve_sync(
            text,
            config=Config(temperature=0.3),
            max_iterations=3,
        )

        assert result.iteration >= 1
        assert result.iteration <= 3
        if result.iteration > 1:
            assert len(result.generations) == result.iteration

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in API."""
        # Empty text doesn't raise an error - it gets improved
        result = await improve("", config=Config(temperature=0.3))
        assert result.final_text != ""  # Should generate text from empty input
        assert len(result.final_text) >= 50  # Should meet minimum length requirement
