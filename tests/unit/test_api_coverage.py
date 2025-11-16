"""Additional tests for api.py to improve coverage."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from sifaka import improve, improve_sync
from sifaka.core.exceptions import ValidationError
from sifaka.core.models import SifakaResult


class TestAsyncImprove:
    """Test the async improve function."""

    @pytest.mark.asyncio
    @patch("sifaka.api.SifakaEngine")
    async def test_improve_basic(self, mock_engine_class):
        """Test basic async improve call."""
        # Setup mock engine
        mock_engine = AsyncMock()
        mock_result = SifakaResult(
            original_text="Test text", final_text="Improved test text", iteration=1
        )
        mock_engine.improve.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        # Call improve
        result = await improve("Test text")

        # Verify
        assert result.original_text == "Test text"
        assert result.final_text == "Improved test text"
        mock_engine.improve.assert_called_once_with("Test text", None)

    @pytest.mark.asyncio
    @patch("sifaka.api.SifakaEngine")
    async def test_improve_with_all_options(self, mock_engine_class):
        """Test improve with all possible options."""
        mock_engine = AsyncMock()
        mock_result = SifakaResult(
            original_text="Test", final_text="Better test", iteration=3
        )
        mock_engine.improve.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        # Call with all options
        from sifaka.core.types import CriticType

        result = await improve(
            text="Test",
            max_iterations=5,
            critics=[CriticType.STYLE, CriticType.REFLEXION],
        )

        assert result.final_text == "Better test"

        # Verify engine was created with config
        engine_config = mock_engine_class.call_args[0][0]
        assert engine_config is not None

    @pytest.mark.asyncio
    @patch("sifaka.api.SifakaEngine")
    async def test_improve_empty_text_validation(self, mock_engine_class):
        """Test improve with empty text."""
        from pydantic import ValidationError as PydanticValidationError

        # Should raise validation error before creating engine
        with pytest.raises(
            PydanticValidationError, match="String should have at least 1 character"
        ):
            await improve("")

        with pytest.raises(ValueError, match="Text cannot be empty or just whitespace"):
            await improve("   ")

        # Engine should not be created
        mock_engine_class.assert_not_called()

    @pytest.mark.asyncio
    @patch("sifaka.api.SifakaEngine")
    async def test_improve_with_config(self, mock_engine_class):
        """Test improve with custom config."""
        from sifaka.core.config import Config

        mock_engine = AsyncMock()
        mock_result = SifakaResult(
            original_text="Test", final_text="Improved", iteration=1
        )
        mock_engine.improve.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        # With custom config
        config = Config()
        result = await improve("Test", config=config)
        assert result.final_text == "Improved"

    @pytest.mark.asyncio
    @patch("sifaka.api.SifakaEngine")
    async def test_improve_error_propagation(self, mock_engine_class):
        """Test that errors are properly propagated."""
        mock_engine = AsyncMock()
        mock_engine.improve.side_effect = RuntimeError("Engine failed")
        mock_engine_class.return_value = mock_engine

        with pytest.raises(RuntimeError, match="Engine failed"):
            await improve("Test text")


class TestImproveSync:
    """Test the synchronous improve wrapper."""

    @patch("sifaka.api.asyncio.run")
    def test_improve_sync_calls_async(self, mock_run):
        """Test that improve_sync properly calls async version."""
        mock_result = SifakaResult(
            original_text="Sync test", final_text="Sync improved", iteration=1
        )
        mock_run.return_value = mock_result

        result = improve_sync("Sync test")

        assert result.final_text == "Sync improved"

        # Verify asyncio.run was called
        mock_run.assert_called_once()

        # Get the coroutine that was passed to asyncio.run
        coro = mock_run.call_args[0][0]
        assert asyncio.iscoroutine(coro)
        # Clean up the coroutine
        coro.close()

    @patch("sifaka.api.asyncio.run")
    def test_improve_sync_with_kwargs(self, mock_run):
        """Test improve_sync passes all kwargs correctly."""
        mock_result = SifakaResult(
            original_text="Test", final_text="Better", iteration=2
        )
        mock_run.return_value = mock_result

        from sifaka.core.types import CriticType

        result = improve_sync("Test", critics=[CriticType.STYLE], max_iterations=3)

        assert result.final_text == "Better"
        mock_run.assert_called_once()

        # Clean up coroutine
        coro = mock_run.call_args[0][0]
        coro.close()

    @patch("sifaka.api.asyncio.run")
    def test_improve_sync_error_handling(self, mock_run):
        """Test error handling in sync wrapper."""
        mock_run.side_effect = ValidationError("Invalid input", "test_validator")

        with pytest.raises(ValidationError, match="Invalid input"):
            improve_sync("Test")

    @patch("sifaka.api.asyncio.run")
    def test_improve_sync_timeout(self, mock_run):
        """Test timeout handling."""
        mock_run.side_effect = asyncio.TimeoutError()

        with pytest.raises(asyncio.TimeoutError):
            improve_sync("Test")


class TestAPIHelpers:
    """Test helper functions in api module."""

    def test_import_structure(self):
        """Test that all expected functions are exported."""
        from sifaka import api

        assert hasattr(api, "improve")
        assert hasattr(api, "improve_sync")
        assert callable(api.improve)
        assert callable(api.improve_sync)

    @pytest.mark.asyncio
    async def test_async_context_manager_pattern(self):
        """Test using improve in async context."""
        with patch("sifaka.api.SifakaEngine") as mock_engine_class:
            mock_engine = AsyncMock()
            mock_result = SifakaResult(
                original_text="Context test", final_text="Context improved", iteration=1
            )
            mock_engine.improve.return_value = mock_result
            mock_engine_class.return_value = mock_engine

            # Should work in async context
            async def async_function():
                result = await improve("Context test")
                return result

            result = await async_function()
            assert result.final_text == "Context improved"


@pytest.mark.asyncio
async def test_improve_with_middleware():
    """Test improve with middleware to cover lines 126-136."""
    from sifaka.api import improve
    from sifaka.core.middleware import MiddlewarePipeline
    from sifaka.core.models import SifakaResult

    with patch("sifaka.api.SifakaEngine") as mock_engine_class:
        # Setup mock engine
        mock_engine = AsyncMock()
        mock_result = SifakaResult(
            original_text="test", final_text="improved", iteration=1
        )
        mock_engine.improve.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        # Create middleware
        middleware = MiddlewarePipeline()

        # This should execute the middleware path (lines 126-136)
        result = await improve("test text", middleware=middleware)

        assert result.final_text == "improved"
        # Verify engine was created
        mock_engine_class.assert_called_once()


@pytest.mark.asyncio
async def test_improve_config_validation_error():
    """Test config validation error path to cover lines 113-114."""
    from sifaka.api import improve
    from sifaka.core.config import Config

    with patch("sifaka.api.validate_config_params") as mock_validate:
        mock_validate.side_effect = ValueError("Invalid config parameter")

        # This should trigger the ValueError handling
        with pytest.raises(
            ValueError,
            match="Configuration validation failed: Invalid config parameter",
        ):
            await improve("test", config=Config())
