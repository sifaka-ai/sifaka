"""Tests for the validation engine module."""

import pytest
from unittest.mock import Mock, AsyncMock
from collections import deque

from sifaka.core.engine.validation import ValidationRunner
from sifaka.core.models import (
    SifakaResult,
    ValidationResult,
    Generation,
    CritiqueResult,
)


class TestValidationRunner:
    """Test the ValidationRunner class."""

    @pytest.fixture
    def runner(self):
        """Create a validation runner instance."""
        return ValidationRunner()

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Current version")

    @pytest.mark.asyncio
    async def test_run_validators_empty_list(self, runner, sample_result):
        """Test running with no validators."""
        result = await runner.run_validators("Test text", sample_result, [])
        assert result is True
        assert len(sample_result.validations) == 0

    @pytest.mark.asyncio
    async def test_run_validators_all_pass(self, runner, sample_result):
        """Test when all validators pass."""
        # Create mock validators
        validator1 = Mock()
        validator1.validate = AsyncMock(
            return_value=ValidationResult(
                validator="LengthValidator",
                passed=True,
                score=1.0,
                details="Length is good",
            )
        )

        validator2 = Mock()
        validator2.validate = AsyncMock(
            return_value=ValidationResult(
                validator="FormatValidator",
                passed=True,
                score=0.9,
                details="Format is acceptable",
            )
        )

        validators = [validator1, validator2]
        result = await runner.run_validators("Test text", sample_result, validators)

        assert result is True
        assert len(sample_result.validations) == 2
        assert sample_result.validations[0].validator == "LengthValidator"
        assert sample_result.validations[0].passed is True
        assert sample_result.validations[1].validator == "FormatValidator"
        assert sample_result.validations[1].passed is True

    @pytest.mark.asyncio
    async def test_run_validators_some_fail(self, runner, sample_result):
        """Test when some validators fail."""
        # Create mock validators
        validator1 = Mock()
        validator1.validate = AsyncMock(
            return_value=ValidationResult(
                validator="LengthValidator",
                passed=False,
                score=0.3,
                details="Text too short",
            )
        )

        validator2 = Mock()
        validator2.validate = AsyncMock(
            return_value=ValidationResult(
                validator="FormatValidator",
                passed=True,
                score=0.9,
                details="Format is good",
            )
        )

        validator3 = Mock()
        validator3.validate = AsyncMock(
            return_value=ValidationResult(
                validator="ContentValidator",
                passed=False,
                score=0.0,
                details="Missing required content",
            )
        )

        validators = [validator1, validator2, validator3]
        result = await runner.run_validators("Test text", sample_result, validators)

        assert result is False
        assert len(sample_result.validations) == 3
        assert sample_result.validations[0].passed is False
        assert sample_result.validations[1].passed is True
        assert sample_result.validations[2].passed is False

    @pytest.mark.asyncio
    async def test_run_validators_with_exception(self, runner, sample_result):
        """Test handling validator exceptions."""
        # Create validators with one that raises exception
        validator1 = Mock()
        validator1.validate = AsyncMock(
            return_value=ValidationResult(
                validator="GoodValidator", passed=True, score=1.0, details="All good"
            )
        )

        validator2 = Mock()
        validator2.name = "BadValidator"
        validator2.validate = AsyncMock(side_effect=Exception("Validation failed"))

        validator3 = Mock()
        validator3.validate = AsyncMock(
            return_value=ValidationResult(
                validator="AnotherGoodValidator",
                passed=True,
                score=0.8,
                details="Also good",
            )
        )

        validators = [validator1, validator2, validator3]
        result = await runner.run_validators("Test text", sample_result, validators)

        assert result is False  # Should fail due to exception
        assert len(sample_result.validations) == 3
        assert sample_result.validations[0].passed is True
        assert sample_result.validations[1].passed is False
        assert sample_result.validations[1].validator == "BadValidator"
        assert (
            "Validation error: Validation failed"
            in sample_result.validations[1].details
        )
        assert sample_result.validations[1].score == 0.0
        assert sample_result.validations[2].passed is True

    @pytest.mark.asyncio
    async def test_run_validators_exception_no_name(self, runner, sample_result):
        """Test handling validator exception when validator has no name attribute."""
        validator = Mock(spec=[])  # No attributes
        validator.validate = AsyncMock(side_effect=Exception("No name error"))

        validators = [validator]
        result = await runner.run_validators("Test text", sample_result, validators)

        assert result is False
        assert len(sample_result.validations) == 1
        assert sample_result.validations[0].validator == "unknown"
        assert sample_result.validations[0].passed is False
        assert "Validation error: No name error" in sample_result.validations[0].details

    @pytest.mark.asyncio
    async def test_run_validators_calls_validate_correctly(self, runner, sample_result):
        """Test that validators are called with correct arguments."""
        validator = Mock()
        validator.validate = AsyncMock(
            return_value=ValidationResult(
                validator="TestValidator", passed=True, score=1.0, details="Good"
            )
        )

        text = "Text to validate"
        await runner.run_validators(text, sample_result, [validator])

        # Verify validator was called with correct arguments
        validator.validate.assert_called_once_with(text, sample_result)

    def test_check_memory_bounds_within_limits(self, runner):
        """Test memory bounds when within limits."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add some items
        for i in range(10):
            result.generations.append(
                Generation(text=f"Gen {i}", model="test", iteration=i)
            )
            result.critiques.append(
                CritiqueResult(
                    critic=f"Critic {i}",
                    feedback=f"Feedback {i}",
                    suggestions=[],
                    needs_improvement=True,
                )
            )
            result.validations.append(
                ValidationResult(
                    validator=f"Validator {i}",
                    passed=True,
                    score=1.0,
                    details=f"Details {i}",
                )
            )

        # Check bounds with high limit
        runner.check_memory_bounds(result, max_elements=100)

        # Should not trim anything
        assert len(result.generations) == 10
        assert len(result.critiques) == 10
        assert len(result.validations) == 10

    def test_check_memory_bounds_exceeds_generations(self, runner):
        """Test memory bounds when generations exceed limit."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add many generations
        for i in range(20):
            result.generations.append(
                Generation(text=f"Gen {i}", model="test", iteration=i)
            )

        # Check bounds with low limit
        runner.check_memory_bounds(result, max_elements=5)

        # Should keep only last 5
        assert len(result.generations) == 5
        assert result.generations[0].text == "Gen 15"
        assert result.generations[-1].text == "Gen 19"
        assert result.generations.maxlen == 5

    def test_check_memory_bounds_exceeds_critiques(self, runner):
        """Test memory bounds when critiques exceed limit."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add many critiques
        for i in range(20):
            result.critiques.append(
                CritiqueResult(
                    critic=f"Critic {i}",
                    feedback=f"Feedback {i}",
                    suggestions=[],
                    needs_improvement=True,
                )
            )

        # Check bounds with low limit
        runner.check_memory_bounds(result, max_elements=5)

        # Should keep only last 5
        assert len(result.critiques) == 5
        assert result.critiques[0].critic == "Critic 15"
        assert result.critiques[-1].critic == "Critic 19"
        assert result.critiques.maxlen == 5

    def test_check_memory_bounds_exceeds_validations(self, runner):
        """Test memory bounds when validations exceed limit."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add many validations
        for i in range(20):
            result.validations.append(
                ValidationResult(
                    validator=f"Validator {i}",
                    passed=True,
                    score=1.0,
                    details=f"Details {i}",
                )
            )

        # Check bounds with low limit
        runner.check_memory_bounds(result, max_elements=5)

        # Should keep only last 5
        assert len(result.validations) == 5
        assert result.validations[0].validator == "Validator 15"
        assert result.validations[-1].validator == "Validator 19"
        assert result.validations.maxlen == 5

    def test_check_memory_bounds_all_exceed(self, runner):
        """Test memory bounds when all collections exceed limit."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add many items to all collections
        for i in range(100):
            result.generations.append(
                Generation(text=f"Gen {i}", model="test", iteration=i)
            )
            result.critiques.append(
                CritiqueResult(
                    critic=f"Critic {i}",
                    feedback=f"Feedback {i}",
                    suggestions=[],
                    needs_improvement=True,
                )
            )
            result.validations.append(
                ValidationResult(
                    validator=f"Validator {i}",
                    passed=True,
                    score=1.0,
                    details=f"Details {i}",
                )
            )

        # Check bounds with limit of 10
        runner.check_memory_bounds(result, max_elements=10)

        # All should be trimmed to 10
        assert len(result.generations) == 10
        assert len(result.critiques) == 10
        assert len(result.validations) == 10

        # Check we kept the most recent items
        assert result.generations[0].text == "Gen 90"
        assert result.critiques[0].critic == "Critic 90"
        assert result.validations[0].validator == "Validator 90"

    def test_check_memory_bounds_empty_collections(self, runner):
        """Test memory bounds with empty collections."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Ensure collections are empty
        assert len(result.generations) == 0
        assert len(result.critiques) == 0
        assert len(result.validations) == 0

        # Check bounds
        runner.check_memory_bounds(result, max_elements=10)

        # Should still be empty
        assert len(result.generations) == 0
        assert len(result.critiques) == 0
        assert len(result.validations) == 0

    def test_check_memory_bounds_exact_limit(self, runner):
        """Test memory bounds when exactly at limit."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add exactly 5 items
        for i in range(5):
            result.generations.append(
                Generation(text=f"Gen {i}", model="test", iteration=i)
            )

        # Check bounds with limit of 5
        runner.check_memory_bounds(result, max_elements=5)

        # Should not change
        assert len(result.generations) == 5
        assert result.generations[0].text == "Gen 0"
        assert result.generations[-1].text == "Gen 4"

    def test_check_memory_bounds_preserves_deque_behavior(self, runner):
        """Test that memory bounds preserves deque behavior after trimming."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Add many items
        for i in range(20):
            result.generations.append(
                Generation(text=f"Gen {i}", model="test", iteration=i)
            )

        # Check bounds
        runner.check_memory_bounds(result, max_elements=5)

        # Verify it's still a deque with correct maxlen
        assert isinstance(result.generations, deque)
        assert result.generations.maxlen == 5

        # Add new item - should auto-remove oldest
        result.generations.append(
            Generation(text="Gen New", model="test", iteration=20)
        )
        assert len(result.generations) == 5
        assert result.generations[0].text == "Gen 16"  # Gen 15 was removed
        assert result.generations[-1].text == "Gen New"
