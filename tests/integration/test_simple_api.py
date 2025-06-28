"""Integration tests for the simple API."""

import pytest
from sifaka import improve_sync, improve, Config
from sifaka.validators.basic import LengthValidator


class TestSimpleAPI:
    """Test the simplified API functions."""

    def test_improve_sync_basic(self):
        """Test basic synchronous improvement."""
        text = "This is a test."
        result = improve_sync(text, temperature=0.3, max_iterations=1)

        assert isinstance(result.final_text, str)
        assert result.iteration_count >= 1
        assert len(result.critiques) > 0

    def test_improve_sync_with_config(self):
        """Test improvement with custom config."""
        text = "The data shows interesting patterns."
        config = Config(
            temperature=0.3,
            max_iterations=2,
            min_quality_score=0.7,
        )

        result = improve_sync(text, config=config)

        assert isinstance(result.final_text, str)
        assert result.original_text == text
        assert result.iteration_count <= 2

    @pytest.mark.asyncio
    async def test_improve_async(self):
        """Test async improvement."""
        text = "Machine learning uses data to make predictions."

        result = await improve(
            text,
            temperature=0.3,
            max_iterations=1,
        )

        assert isinstance(result.final_text, str)
        assert result.iteration_count >= 1
        assert len(result.critiques) > 0

    @pytest.mark.asyncio
    async def test_improve_with_validators(self):
        """Test improvement with validators."""
        text = "Short text"

        validator = LengthValidator(min_length=50)
        result = await improve(
            text,
            validators=[validator],
            temperature=0.3,
            max_iterations=2,
        )

        assert isinstance(result.final_text, str)
        assert len(result.final_text) >= 50  # Should meet length requirement

    def test_multiple_iterations(self):
        """Test that multiple iterations produce refinements."""
        text = "This needs improvement."

        result = improve_sync(
            text,
            temperature=0.3,
            max_iterations=3,
            min_quality_score=0.8,
        )

        assert result.iteration_count >= 1
        assert result.iteration_count <= 3
        if result.iteration_count > 1:
            assert len(result.improvements) == result.iteration_count

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in API."""
        with pytest.raises(ValueError):
            await improve("", temperature=0.3)  # Empty text should raise error
