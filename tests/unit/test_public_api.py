"""Tests for Sifaka's public API surface."""

from datetime import datetime
from unittest.mock import patch

import pytest

from sifaka import improve_sync
from sifaka.core.models import (
    SifakaResult,
)


class TestPublicAPI:
    """Test the main public API functions."""

    @patch("sifaka.api.improve")
    def test_improve_sync_basic(self, mock_improve):
        """Test basic synchronous improve call."""
        # Setup mock
        mock_result = SifakaResult(
            original_text="Original text", final_text="Improved text", iteration=1
        )
        mock_improve.return_value = mock_result

        # Call improve_sync
        result = improve_sync("Original text")

        # Verify
        assert result.original_text == "Original text"
        assert result.final_text == "Improved text"
        assert result.iteration == 1
        mock_improve.assert_called_once()

    @patch("sifaka.api.improve")
    def test_improve_sync_with_options(self, mock_improve):
        """Test improve_sync with custom options."""
        from sifaka.core.types import CriticType

        mock_result = SifakaResult(
            original_text="Test", final_text="Better test", iteration=2
        )
        mock_improve.return_value = mock_result

        result = improve_sync("Test", critics=[CriticType.STYLE], max_iterations=3)

        assert result.final_text == "Better test"

        # Check that options were passed
        mock_improve.assert_called_once()

    @patch("sifaka.api.improve")
    def test_improve_sync_error_handling(self, mock_improve):
        """Test error handling in improve_sync."""
        # Test general exception
        mock_improve.side_effect = Exception("Something went wrong")

        with pytest.raises(Exception, match="Something went wrong"):
            improve_sync("Test")

    @patch("sifaka.api.improve")
    def test_improve_sync_empty_text(self, mock_improve):
        """Test improve_sync with empty text."""
        # Just verify the call happens - actual validation happens in improve()
        mock_result = SifakaResult(original_text="", final_text="", iteration=0)
        mock_improve.return_value = mock_result

        improve_sync("")
        mock_improve.assert_called_once()


class TestSifakaResultOperations:
    """Test operations on SifakaResult objects."""

    def test_result_creation(self):
        """Test creating a SifakaResult."""
        result = SifakaResult(original_text="Original", final_text="Final", iteration=3)

        assert result.original_text == "Original"
        assert result.final_text == "Final"
        assert result.iteration == 3
        assert len(result.id) > 0  # Auto-generated ID
        assert isinstance(result.created_at, datetime)
        assert isinstance(result.updated_at, datetime)

    def test_result_add_generation(self):
        """Test adding generations to result."""
        result = SifakaResult(original_text="Start", final_text="End", iteration=0)

        # Add a generation
        result.add_generation(
            text="First version",
            model="gpt-4",
            prompt="Improve this",
            tokens=100,
            processing_time=1.5,
        )

        assert len(result.generations) == 1
        gen = list(result.generations)[0]
        assert gen.text == "First version"
        assert gen.model == "gpt-4"
        assert gen.tokens_used == 100
        assert gen.processing_time == 1.5

    def test_result_add_critique(self):
        """Test adding critiques to result."""
        result = SifakaResult(original_text="Test", final_text="Test", iteration=1)

        # Add a critique
        result.add_critique(
            critic="style",
            feedback="Could be more formal",
            suggestions=["Use professional language", "Avoid contractions"],
            needs_improvement=True,
            confidence=0.8,
        )

        assert len(result.critiques) == 1
        critique = list(result.critiques)[0]
        assert critique.critic == "style"
        assert critique.feedback == "Could be more formal"
        assert len(critique.suggestions) == 2
        assert critique.confidence == 0.8
        assert critique.needs_improvement is True

    def test_result_add_validation(self):
        """Test adding validations to result."""
        result = SifakaResult(
            original_text="Test", final_text="Test improved", iteration=1
        )

        # Add validations
        result.add_validation(
            validator="length", passed=True, score=0.9, details="Good length"
        )

        result.add_validation(
            validator="grammar", passed=False, score=0.4, details="Grammar issues found"
        )

        assert len(result.validations) == 2
        assert result.all_passed is False  # One validation failed

    def test_result_current_text(self):
        """Test getting current text from result."""
        result = SifakaResult(original_text="Original", final_text="Final", iteration=2)

        # Without generations, should return original
        assert result.current_text == "Original"

        # Add generations
        result.add_generation("Version 1", "gpt-4")
        assert result.current_text == "Version 1"

        result.add_generation("Version 2", "gpt-4")
        assert result.current_text == "Version 2"

    def test_result_needs_improvement(self):
        """Test checking if result needs improvement."""
        result = SifakaResult(original_text="Test", final_text="Test", iteration=1)

        # No critiques = needs improvement
        assert result.needs_improvement is True

        # Add critique that needs improvement
        result.add_critique(
            critic="clarity",
            feedback="Unclear",
            suggestions=["Be more specific"],
            needs_improvement=True,
        )
        assert result.needs_improvement is True

        # Add critique that doesn't need improvement
        result.add_critique(
            critic="style",
            feedback="Perfect style",
            suggestions=[],
            needs_improvement=False,
        )

        # Still needs improvement (one critic says yes)
        assert result.needs_improvement is True

    def test_result_deque_memory_bounds(self):
        """Test that deques respect memory bounds."""
        result = SifakaResult(original_text="Test", final_text="Test", iteration=1)

        # Add more than maxlen generations (10)
        for i in range(15):
            result.add_generation(f"Gen {i}", "gpt-4")

        # Should only keep last 10
        assert len(result.generations) == 10
        gens = list(result.generations)
        assert gens[0].text == "Gen 5"  # First 5 dropped
        assert gens[-1].text == "Gen 14"  # Last one kept

    def test_result_model_dump(self):
        """Test converting result to dict."""
        result = SifakaResult(
            original_text="Original",
            final_text="Improved",
            iteration=2,
            processing_time=5.5,
        )

        result.add_generation("Version 1", "gpt-4", tokens=100)
        result.add_critique(
            critic="style", feedback="Good", suggestions=[], needs_improvement=False
        )

        # Convert to dict
        result_dict = result.model_dump()

        assert result_dict["original_text"] == "Original"
        assert result_dict["final_text"] == "Improved"
        assert result_dict["iteration"] == 2
        assert result_dict["processing_time"] == 5.5
        assert len(result_dict["generations"]) == 1
        assert len(result_dict["critiques"]) == 1
