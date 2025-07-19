"""Comprehensive edge case tests for all modules."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sifaka import improve
from sifaka.core.exceptions import (
    CriticError,
    TimeoutError,
)
from sifaka.core.models import (
    CritiqueResult,
    Generation,
    SifakaResult,
    ValidationResult,
)
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.storage import FileStorage, MemoryStorage
from sifaka.validators import ContentValidator, LengthValidator


class TestTextEdgeCases:
    """Test edge cases related to text input."""

    @pytest.mark.asyncio
    async def test_empty_text_input(self):
        """Test handling of empty text input."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Empty text provided."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "",
                max_iterations=1,
                critics=["reflexion"],  # Empty text
            )

            assert isinstance(result, SifakaResult)
            assert result.original_text == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Whitespace text."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            whitespace_text = "   \n\t  \r\n  "
            result = await improve(
                whitespace_text, max_iterations=1, critics=["reflexion"]
            )

            assert isinstance(result, SifakaResult)
            assert result.original_text == whitespace_text

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test handling of extremely long text."""
        # Create 100KB text
        long_text = "This is a very long text for testing edge cases. " * 2000

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Long text processed."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(long_text, max_iterations=1, critics=["reflexion"])

            assert isinstance(result, SifakaResult)
            assert result.original_text == long_text

    @pytest.mark.asyncio
    async def test_unicode_edge_cases(self):
        """Test handling of various Unicode edge cases."""
        unicode_texts = [
            "Hello 世界! 🌍",  # Mixed languages and emoji
            "𝕌𝕟𝕚𝕔𝕠𝕕𝕖 𝓂𝒶𝓉𝒽 𝓈𝓎𝓂𝒷𝑜𝓁𝓈",  # Mathematical symbols
            "עברית ﷻ العربية",  # RTL languages
            "🎉🎊🎈🎁🎀💝🎂🍰",  # Many emoji
            "café naïve résumé",  # Accented characters
            "\u0000\u0001\u0002",  # Control characters
            "🇺🇸🇬🇧🇫🇷🇩🇪",  # Flag emoji
        ]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Unicode processed."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            for text in unicode_texts:
                result = await improve(text, max_iterations=1, critics=["reflexion"])

                assert isinstance(result, SifakaResult)
                assert result.original_text == text

    @pytest.mark.asyncio
    async def test_special_characters_and_escape_sequences(self):
        """Test handling of special characters and escape sequences."""
        special_texts = [
            "Line 1\nLine 2\rLine 3\r\n",  # Different newlines
            "Tab\tSeparated\tValues",  # Tabs
            "Quote: \"Hello\" and 'World'",  # Quotes
            "Backslash: \\ and forward: /",  # Slashes
            "Null byte: \x00 and bell: \x07",  # Control chars
            'JSON: {"key": "value"}',  # JSON-like
            "HTML: <div>content</div>",  # HTML-like
            "SQL: SELECT * FROM table;",  # SQL-like
            "Regex: ^[a-zA-Z0-9]+$",  # Regex pattern
        ]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Special chars handled."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            for text in special_texts:
                result = await improve(text, max_iterations=1, critics=["reflexion"])

                assert isinstance(result, SifakaResult)
                assert result.original_text == text


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    @pytest.mark.asyncio
    async def test_zero_max_iterations(self):
        """Test configuration with zero max iterations."""
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            await improve("Test text", max_iterations=0, critics=["reflexion"])

    @pytest.mark.asyncio
    async def test_negative_max_iterations(self):
        """Test configuration with negative max iterations."""
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            await improve("Test text", max_iterations=-1, critics=["reflexion"])

    @pytest.mark.asyncio
    async def test_excessive_max_iterations(self):
        """Test configuration with too many max iterations."""
        with pytest.raises(ValueError, match="max_iterations cannot exceed 10"):
            await improve("Test text", max_iterations=15, critics=["reflexion"])

    @pytest.mark.asyncio
    async def test_invalid_temperature(self):
        """Test configuration with invalid temperature values."""
        # Temperature below 0
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            await improve("Test text", temperature=-0.1, critics=["reflexion"])

        # Temperature above 2
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            await improve("Test text", temperature=2.1, critics=["reflexion"])

    @pytest.mark.asyncio
    async def test_empty_critics_list(self):
        """Test configuration with empty critics list."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Default critic used."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Should default to reflexion critic
            result = await improve(
                "Test text",
                critics=[],
                max_iterations=1,  # Empty list
            )

            assert isinstance(result, SifakaResult)
            assert len(result.critiques) >= 1

    @pytest.mark.asyncio
    async def test_unknown_critic(self):
        """Test configuration with unknown critic."""
        with pytest.raises(ValueError, match="Unknown critic"):
            await improve("Test text", critics=["unknown_critic"], max_iterations=1)

    @pytest.mark.asyncio
    async def test_unknown_model(self):
        """Test configuration with unknown model."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Unknown model test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            # Simulate model not found error
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Model not found")
            )
            mock_openai.return_value = mock_client

            with pytest.raises(Exception):
                await improve(
                    "Test text",
                    model="unknown-model-123",
                    critics=["reflexion"],
                    max_iterations=1,
                )


class TestCriticEdgeCases:
    """Test edge cases in critic behavior."""

    @pytest.mark.asyncio
    async def test_critic_empty_response(self):
        """Test critic handling of empty responses."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""  # Empty response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            critic = ReflexionCritic()

            # Should handle empty response gracefully
            with pytest.raises(CriticError):
                await critic.critique(
                    "Test text",
                    SifakaResult(
                        original_text="Test",
                        final_text="Test",
                        iteration=1,
                        generations=[],
                        critiques=[],
                        validations=[],
                        processing_time=1.0,
                    ),
                )

    @pytest.mark.asyncio
    async def test_critic_malformed_response(self):
        """Test critic handling of malformed responses."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Not a valid reflection format"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            critic = ReflexionCritic()

            with pytest.raises(CriticError):
                await critic.critique(
                    "Test text",
                    SifakaResult(
                        original_text="Test",
                        final_text="Test",
                        iteration=1,
                        generations=[],
                        critiques=[],
                        validations=[],
                        processing_time=1.0,
                    ),
                )

    @pytest.mark.asyncio
    async def test_constitutional_critic_invalid_json(self):
        """Test Constitutional critic with invalid JSON response."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Invalid JSON response"

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            critic = ConstitutionalCritic()

            with pytest.raises(CriticError):
                await critic.critique(
                    "Test text",
                    SifakaResult(
                        original_text="Test",
                        final_text="Test",
                        iteration=1,
                        generations=[],
                        critiques=[],
                        validations=[],
                        processing_time=1.0,
                    ),
                )

    @pytest.mark.asyncio
    async def test_critic_api_error(self):
        """Test critic handling of API errors."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_openai.return_value = mock_client

            critic = ReflexionCritic()

            with pytest.raises(CriticError):
                await critic.critique(
                    "Test text",
                    SifakaResult(
                        original_text="Test",
                        final_text="Test",
                        iteration=1,
                        generations=[],
                        critiques=[],
                        validations=[],
                        processing_time=1.0,
                    ),
                )


class TestValidatorEdgeCases:
    """Test edge cases in validator behavior."""

    @pytest.mark.asyncio
    async def test_length_validator_boundary_conditions(self):
        """Test length validator at exact boundaries."""
        validator = LengthValidator(min_length=10, max_length=10)

        # Exactly 10 characters
        result = await validator.validate("1234567890")
        assert result.passed is True

        # 9 characters (too short)
        result = await validator.validate("123456789")
        assert result.passed is False

        # 11 characters (too long)
        result = await validator.validate("12345678901")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_content_validator_case_sensitivity(self):
        """Test content validator case sensitivity edge cases."""
        validator = ContentValidator(
            required_terms=["HELLO", "world"], forbidden_terms=["BAD", "evil"]
        )

        # Test mixed case
        result = await validator.validate("hello WORLD is good, not bad or EVIL")
        assert result.passed is False  # Contains forbidden terms

    @pytest.mark.asyncio
    async def test_content_validator_partial_matches(self):
        """Test content validator partial word matching."""
        validator = ContentValidator(required_terms=["test"], forbidden_terms=["bad"])

        # Partial matches should be found
        result = await validator.validate("testing is good, not badly done")
        assert result.passed is False  # "badly" contains "bad"

    @pytest.mark.asyncio
    async def test_content_validator_unicode_terms(self):
        """Test content validator with Unicode terms."""
        validator = ContentValidator(
            required_terms=["🎉", "世界"], forbidden_terms=["😈"]
        )

        # Test with Unicode content
        result = await validator.validate("Hello 世界! 🎉 Great day, no 😈 here")
        assert result.passed is False  # Contains forbidden emoji

    @pytest.mark.asyncio
    async def test_validator_with_none_input(self):
        """Test validator behavior with None input."""
        validator = LengthValidator(min_length=1)

        with pytest.raises((TypeError, AttributeError)):
            await validator.validate(None)


class TestStorageEdgeCases:
    """Test edge cases in storage behavior."""

    @pytest.mark.asyncio
    async def test_storage_load_nonexistent_id(self):
        """Test loading non-existent result ID."""
        storage = MemoryStorage()

        result = await storage.load("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_storage_delete_nonexistent_id(self):
        """Test deleting non-existent result ID."""
        storage = MemoryStorage()

        deleted = await storage.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_file_storage_invalid_directory(self):
        """Test file storage with invalid directory."""
        # Try to create storage in a file (not directory)
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises((OSError, IOError, FileNotFoundError)):
                storage = FileStorage(storage_dir=temp_file.name)
                # Try to save something
                result = SifakaResult(
                    original_text="test",
                    final_text="test",
                    iteration=1,
                    generations=[],
                    critiques=[],
                    validations=[],
                    processing_time=1.0,
                )
                await storage.save(result)

    @pytest.mark.asyncio
    async def test_file_storage_permission_error(self):
        """Test file storage with permission errors."""
        # This test might not work on all systems, so we'll mock it
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(storage_dir=temp_dir)

            result = SifakaResult(
                original_text="test",
                final_text="test",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )

            # Mock permission error during save
            with patch(
                "aiofiles.open", side_effect=PermissionError("Permission denied")
            ):
                with pytest.raises(PermissionError):
                    await storage.save(result)

    @pytest.mark.asyncio
    async def test_storage_corrupted_file(self):
        """Test file storage with corrupted data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(storage_dir=temp_dir)

            # Create a corrupted file
            corrupted_file = os.path.join(temp_dir, "corrupted.json")
            with open(corrupted_file, "w") as f:
                f.write("Not valid JSON content")

            # Try to load it
            result = await storage.load("corrupted")
            assert result is None  # Should handle corruption gracefully

    @pytest.mark.asyncio
    async def test_memory_storage_memory_limit(self):
        """Test memory storage behavior at memory limits."""
        storage = MemoryStorage(max_results=3)

        # Create and save 5 results
        results = []
        for i in range(5):
            result = SifakaResult(
                original_text=f"test {i}",
                final_text=f"test {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            result_id = await storage.save(result)
            results.append(result_id)

        # Should only keep the last 3
        all_results = await storage.list_results()
        assert len(all_results) == 3

        # First 2 should be evicted
        assert await storage.load(results[0]) is None
        assert await storage.load(results[1]) is None

        # Last 3 should still exist
        assert await storage.load(results[2]) is not None
        assert await storage.load(results[3]) is not None
        assert await storage.load(results[4]) is not None


class TestErrorEdgeCases:
    """Test edge cases in error handling."""

    @pytest.mark.asyncio
    async def test_timeout_edge_cases(self):
        """Test timeout handling edge cases."""
        # Very short timeout
        with pytest.raises(TimeoutError):
            await improve(
                "Test text",
                timeout_seconds=0.001,
                critics=["reflexion"],  # 1ms timeout
            )

    @pytest.mark.asyncio
    async def test_multiple_error_conditions(self):
        """Test behavior when multiple error conditions occur."""
        # Create a scenario where timeout could be hit
        with pytest.raises(TimeoutError):
            await improve(
                "Test text",
                timeout_seconds=0.001,  # Very short timeout
                critics=["reflexion"],
            )


class TestDataIntegrityEdgeCases:
    """Test edge cases related to data integrity."""

    def test_result_id_uniqueness(self):
        """Test that result IDs are always unique."""
        ids = set()

        for _ in range(1000):
            result = SifakaResult(
                original_text="test",
                final_text="test",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )

            assert result.id not in ids, "Duplicate ID generated"
            ids.add(result.id)

    def test_timestamp_consistency(self):
        """Test timestamp consistency and ordering."""
        result1 = SifakaResult(
            original_text="test1",
            final_text="test1",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        # Small delay to ensure different timestamps
        import time

        time.sleep(0.001)

        result2 = SifakaResult(
            original_text="test2",
            final_text="test2",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        assert result2.created_at >= result1.created_at

    def test_memory_bounded_collections_edge_cases(self):
        """Test memory bounded collections at their limits."""
        result = SifakaResult(
            original_text="test",
            final_text="test",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        # Add items up to the limit
        for i in range(15):  # More than the limit of 10
            generation = Generation(
                iteration=1,
                text=f"Generation {i}",
                model="gpt-4o-mini",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            )
            result.generations.append(generation)

        # Should be bounded to 10
        assert len(result.generations) == 10

        # Should contain the most recent items
        assert "Generation 14" in list(result.generations)[-1].text

    def test_json_serialization_edge_cases(self):
        """Test JSON serialization of complex objects."""
        # Create result with all possible fields populated
        result = SifakaResult(
            original_text="test with unicode 🎉 and special chars \n\t",
            final_text="improved text",
            iteration=5,
            generations=[
                Generation(
                    iteration=1,
                    text="Gen 1",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=200,
                    total_tokens=300,
                )
            ],
            critiques=[
                CritiqueResult(
                    critic="reflexion",
                    feedback="Good feedback with unicode 🎯",
                    suggestions=["Suggestion 1", "Suggestion 2"],
                    needs_improvement=True,
                    confidence=0.85,
                )
            ],
            validations=[
                ValidationResult(
                    validator="length",
                    passed=True,
                    score=1.0,
                    details="Length validation passed",
                )
            ],
            processing_time=2.5,
        )

        # Should be able to serialize to JSON
        json_str = json.dumps(result.model_dump(), default=str)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["original_text"] == result.original_text
