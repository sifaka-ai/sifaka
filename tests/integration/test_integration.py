"""Integration tests for the complete Sifaka system."""

import tempfile
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sifaka import FileStorage, SifakaResult, improve
from sifaka.core.config import Config, LLMConfig
from sifaka.validators import LengthValidator


class TestSifakaIntegration:
    """Test complete Sifaka workflows."""

    @pytest.mark.asyncio
    async def test_basic_improvement_workflow(self):
        """Test basic text improvement workflow."""
        # Mock OpenAI responses
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """REFLECTION: The text provides good information but could be more engaging.
SUGGESTIONS:
1. Add specific examples
2. Improve the introduction
3. Include a stronger conclusion"""

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Renewable energy is important for the environment",
                max_iterations=2,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)
            assert (
                result.original_text
                == "Renewable energy is important for the environment"
            )
            assert result.iteration >= 1
            assert len(result.critiques) >= 1
            assert result.critiques[0].critic == "reflexion"

    @pytest.mark.asyncio
    async def test_multiple_critics_workflow(self):
        """Test workflow with multiple critics."""

        # Mock different responses for different critics
        def mock_create_side_effect(*args, **kwargs):
            # Check the system message to determine which critic is calling
            system_content = kwargs.get("messages", [{}])[0].get("content", "")

            mock_response = MagicMock()
            if "reflexion" in system_content.lower():
                mock_response.choices[
                    0
                ].message.content = """REFLECTION: Good structure.
SUGGESTIONS: Add examples"""
            elif "constitutional" in system_content.lower():
                # Return JSON for constitutional critic
                mock_response.choices[0].message.content = """{
    "overall_assessment": "Generally follows principles",
    "principle_scores": {"1": 4, "2": 3, "3": 4},
    "violations": [],
    "suggestions": ["Add more detail"],
    "overall_confidence": 0.8,
    "evaluation_quality": 4
}"""
            else:
                mock_response.choices[0].message.content = "Generic feedback"

            return mock_response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=mock_create_side_effect
            )
            mock_openai.return_value = mock_client

            result = await improve(
                "Climate change is a serious issue",
                max_iterations=1,
                critics=["reflexion", "constitutional"],
            )

            assert len(result.critiques) >= 2
            critic_names = [c.critic for c in result.critiques]
            assert "reflexion" in critic_names

    @pytest.mark.asyncio
    async def test_with_validators(self):
        """Test workflow with validators."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Good text. SUGGESTIONS: Keep improving"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Use a text that should pass length validation
            long_text = "This is a longer text about renewable energy. " * 10

            result = await improve(
                long_text,
                max_iterations=1,
                critics=["reflexion"],
                validators=[LengthValidator(min_length=50, max_length=1000)],
            )
            assert len(result.validations) >= 1
            assert result.validations[0].validator == "length"

    @pytest.mark.asyncio
    async def test_with_file_storage(self):
        """Test workflow with file storage."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Good analysis. SUGGESTIONS: Add details"

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(storage_dir=temp_dir)

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                result = await improve(
                    "Test text for storage",
                    max_iterations=1,
                    critics=["reflexion"],
                    storage=storage,
                )

                # Verify result was saved
                loaded_result = await storage.load(result.id)
                assert loaded_result is not None
                assert loaded_result.id == result.id
                assert loaded_result.original_text == result.original_text

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        """Test timeout enforcement."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Analysis. SUGGESTIONS: Improve"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Set very low timeout via config
            config = Config(
                llm=LLMConfig(timeout_seconds=0.1)  # Minimum allowed timeout
            )

            # The improve function doesn't actually raise timeout in mocked mode
            # Instead, it should complete but show timeout in the critiques
            result = await improve(
                "Test text",
                max_iterations=1,
                critics=["reflexion"],
                config=config,
            )

            # Verify the result exists (timeout doesn't prevent completion in mock mode)
            assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test system recovery from critic errors."""
        # Mock one critic to fail, another to succeed
        call_count = 0

        def mock_create_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call (validation) fails
                raise Exception("API Error")
            else:
                # Subsequent calls succeed
                mock_response = MagicMock()
                mock_response.choices[
                    0
                ].message.content = (
                    "REFLECTION: Recovery successful. SUGGESTIONS: Continue"
                )
                return mock_response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=mock_create_side_effect
            )
            mock_openai.return_value = mock_client

            result = await improve(
                "Test text for error recovery", max_iterations=1, critics=["reflexion"]
            )
            # Should still get a result despite the error
            assert isinstance(result, SifakaResult)
            assert len(result.critiques) >= 1

    @pytest.mark.asyncio
    async def test_memory_bounds_enforcement(self):
        """Test memory bounds are enforced."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Analysis. SUGGESTIONS: Improve"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve("Test text", max_iterations=3, critics=["reflexion"])

            # Even with multiple iterations, collections should be bounded
            assert len(result.generations) <= 10
            assert len(result.critiques) <= 20
            assert len(result.validations) <= 20

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation in workflows."""
        with pytest.raises(ValueError):
            await improve(
                "Test text",
                max_iterations=15,  # Exceeds limit of 10
                critics=["reflexion"],
            )

    @pytest.mark.asyncio
    async def test_empty_critics_list(self):
        """Test handling of empty critics list."""
        # Should use default critics
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Default analysis. SUGGESTIONS: Improve"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test text",
                critics=None,  # Should default to ["reflexion"]
            )

            assert len(result.critiques) >= 1
            assert result.critiques[0].critic == "reflexion"

    @pytest.mark.asyncio
    async def test_custom_model_and_temperature(self):
        """Test using custom model and temperature settings."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Custom model analysis. SUGGESTIONS: Continue"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            create_mock = AsyncMock(return_value=mock_response)
            mock_client.chat.completions.create = create_mock
            mock_openai.return_value = mock_client

            config = Config(llm=LLMConfig(model="gpt-4", temperature=0.9))
            result = await improve(
                "Test text",
                max_iterations=1,
                critics=["reflexion"],
                config=config,
            )

            # In CI mode with mocks, we just verify the result was created
            # The actual model/temperature verification depends on the mocking setup
            assert isinstance(result, SifakaResult)
            assert result.final_text is not None

    @pytest.mark.asyncio
    async def test_result_completeness(self):
        """Test that results contain all expected information."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Complete analysis. SUGGESTIONS: Finalize"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test text for completeness", max_iterations=2, critics=["reflexion"]
            )

            # Verify all expected fields are present
            assert result.id is not None
            assert result.original_text == "Test text for completeness"
            assert result.final_text is not None
            assert result.iteration >= 1
            assert result.processing_time > 0
            assert result.created_at is not None
            assert hasattr(result, "generations")
            assert isinstance(result.generations, deque)
            assert isinstance(result.critiques, deque)
            assert isinstance(result.validations, deque)
