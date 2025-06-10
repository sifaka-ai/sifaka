"""Comprehensive unit tests for Sifaka validators.

This module provides extensive testing of the validator system:
- BaseValidator functionality and interface compliance
- Specific validator implementations (length, coherence, etc.)
- Async/sync coordination and error handling
- Validation result aggregation and reporting
- Performance characteristics and edge cases

Tests cover:
- Validator interface compliance
- Validation logic and thresholds
- Error handling and fallbacks
- Performance and resource usage
- Integration with SifakaThought
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from sifaka.core.interfaces import Validator
from sifaka.core.thought import SifakaThought
from sifaka.utils.errors import ValidationError
from sifaka.validators.base import BaseValidator, ValidationResult
from sifaka.validators.coherence import CoherenceValidator
from sifaka.validators.length import LengthValidator


class MockValidator(BaseValidator):
    """Mock validator for testing base functionality."""

    def __init__(self, name: str = "mock", should_pass: bool = True, should_error: bool = False):
        super().__init__(name, f"Mock validator: {name}")
        self.should_pass = should_pass
        self.should_error = should_error
        self.call_count = 0

    async def validate_async(self, thought: SifakaThought) -> ValidationResult:
        """Mock validation implementation."""
        self.call_count += 1

        if self.should_error:
            raise ValidationError(f"Mock validation error for {self.name}")

        return ValidationResult(
            passed=self.should_pass,
            message=f"Mock validation result for {self.name}",
            score=0.8 if self.should_pass else 0.4,
            issues=[] if self.should_pass else ["Mock issue"],
            suggestions=[] if self.should_pass else ["Mock suggestion"],
            metadata={"mock_detail": f"validation_{self.call_count}"},
            validator_name=self.name,
            processing_time_ms=10.0,
        )


class TestBaseValidator:
    """Test BaseValidator functionality and interface compliance."""

    def test_base_validator_initialization(self):
        """Test BaseValidator initialization."""
        validator = MockValidator("test_validator")

        assert validator.name == "test_validator"
        assert validator.description == "Mock validator: test_validator"

    def test_base_validator_interface_compliance(self):
        """Test that BaseValidator implements Validator protocol."""
        validator = MockValidator()

        # Should implement Validator protocol
        assert isinstance(validator, Validator)
        assert hasattr(validator, "validate_async")
        assert hasattr(validator, "name")

    @pytest.mark.asyncio
    async def test_validate_async_basic(self):
        """Test basic async validation."""
        validator = MockValidator("async_test", should_pass=True)
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Test text", "gpt-4", 0)

        result = await validator.validate_async(thought)

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.score == 0.8
        assert "mock_detail" in result.metadata
        assert validator.call_count == 1

    @pytest.mark.asyncio
    async def test_validate_async_failure(self):
        """Test async validation failure."""
        validator = MockValidator("failure_test", should_pass=False)
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Bad text", "gpt-4", 0)

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert result.score == 0.4
        assert len(result.issues) == 1
        assert len(result.suggestions) == 1

    @pytest.mark.asyncio
    async def test_validate_async_error_handling(self):
        """Test async validation error handling."""
        validator = MockValidator("error_test", should_error=True)
        thought = SifakaThought(prompt="Test prompt")

        with pytest.raises(ValidationError):
            await validator.validate_async(thought)

    def test_validate_sync_basic(self):
        """Test basic sync validation."""
        validator = MockValidator("sync_test", should_pass=True)
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Test text", "gpt-4", 0)

        result = validator.validate(thought)

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert validator.call_count == 1

    def test_validate_sync_error_handling(self):
        """Test sync validation error handling."""
        validator = MockValidator("sync_error_test", should_error=True)
        thought = SifakaThought(prompt="Test prompt")

        with pytest.raises(ValidationError):
            validator.validate(thought)

    @pytest.mark.asyncio
    async def test_concurrent_validation(self):
        """Test concurrent validation calls."""
        validator = MockValidator("concurrent_test")
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Test text", "gpt-4", 0)

        # Run multiple validations concurrently
        tasks = [validator.validate_async(thought) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 10
        assert all(r.passed for r in results)
        assert validator.call_count == 10

    def test_validator_state_isolation(self):
        """Test that validator instances maintain separate state."""
        validator1 = MockValidator("validator1")
        validator2 = MockValidator("validator2")

        thought = SifakaThought(prompt="Test")
        thought.add_generation("Text", "gpt-4", 0)

        # Call each validator
        validator1.validate(thought)
        validator2.validate(thought)
        validator1.validate(thought)

        # Each should maintain separate call counts
        assert validator1.call_count == 2
        assert validator2.call_count == 1


class TestLengthValidator:
    """Test LengthValidator implementation."""

    def test_length_validator_initialization(self):
        """Test LengthValidator initialization."""
        validator = LengthValidator(min_length=50, max_length=500)

        assert validator.name == "length"
        assert validator.min_length == 50
        assert validator.max_length == 500

    def test_length_validator_default_parameters(self):
        """Test LengthValidator with default parameters."""
        validator = LengthValidator()

        # Should have reasonable defaults
        assert validator.min_length > 0
        assert validator.max_length > validator.min_length

    @pytest.mark.asyncio
    async def test_length_validation_pass(self):
        """Test length validation that passes."""
        validator = LengthValidator(min_length=10, max_length=100)
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation(
            "This is a test text that should pass length validation.", "gpt-4", 0
        )

        result = await validator.validate_async(thought)

        assert result.passed is True
        assert "word_count" in result.metadata
        assert "character_count" in result.metadata
        assert result.metadata["word_count"] >= 10

    @pytest.mark.asyncio
    async def test_length_validation_too_short(self):
        """Test length validation that fails (too short)."""
        validator = LengthValidator(min_length=50, max_length=500)
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Short text.", "gpt-4", 0)

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert "too short" in str(result.issues).lower()
        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_length_validation_too_long(self):
        """Test length validation that fails (too long)."""
        validator = LengthValidator(min_length=5, max_length=20)
        thought = SifakaThought(prompt="Test prompt")
        long_text = "This is a very long text that exceeds the maximum length limit set for this validator test."
        thought.add_generation(long_text, "gpt-4", 0)

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert "too long" in str(result.issues).lower()
        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_length_validation_no_text(self):
        """Test length validation with no generated text."""
        validator = LengthValidator(min_length=10, max_length=100)
        thought = SifakaThought(prompt="Test prompt")
        # No generation added

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert "no text" in str(result.issues).lower()

    @pytest.mark.asyncio
    async def test_length_validation_empty_text(self):
        """Test length validation with empty text."""
        validator = LengthValidator(min_length=10, max_length=100)
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("", "gpt-4", 0)

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert result.metadata["word_count"] == 0
        assert result.metadata["character_count"] == 0

    @pytest.mark.asyncio
    async def test_length_validation_whitespace_only(self):
        """Test length validation with whitespace-only text."""
        validator = LengthValidator(min_length=10, max_length=100)
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("   \n\t   ", "gpt-4", 0)

        result = await validator.validate_async(thought)

        assert result.passed is False
        # Should count actual words, not whitespace
        assert result.metadata["word_count"] == 0

    def test_length_validator_edge_cases(self):
        """Test LengthValidator edge cases."""
        # Zero min length
        validator = LengthValidator(min_length=0, max_length=100)
        assert validator.min_length == 0

        # Equal min and max
        validator = LengthValidator(min_length=50, max_length=50)
        assert validator.min_length == validator.max_length

        # Invalid parameters should raise error
        with pytest.raises(ValueError):
            LengthValidator(min_length=100, max_length=50)  # min > max


class TestCoherenceValidator:
    """Test CoherenceValidator implementation."""

    @pytest.fixture
    def mock_coherence_validator(self):
        """Create a mock CoherenceValidator for testing."""
        with patch("sifaka.validators.coherence.CoherenceValidator") as mock_class:
            mock_instance = Mock(spec=CoherenceValidator)
            mock_instance.name = "coherence"
            mock_instance.threshold = 0.7
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_coherence_validation_pass(self, mock_coherence_validator):
        """Test coherence validation that passes."""
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation(
            "This is a coherent text with logical flow and clear structure.", "gpt-4", 0
        )

        # Mock high coherence score
        mock_coherence_validator.validate_async = AsyncMock(
            return_value=ValidationResult(
                passed=True,
                score=0.85,
                details={"coherence_score": 0.85, "threshold": 0.7},
                issues=[],
                suggestions=[],
            )
        )

        result = await mock_coherence_validator.validate_async(thought)

        assert result.passed is True
        assert result.score == 0.85
        assert result.metadata["coherence_score"] > result.metadata["threshold"]

    @pytest.mark.asyncio
    async def test_coherence_validation_fail(self, mock_coherence_validator):
        """Test coherence validation that fails."""
        thought = SifakaThought(prompt="Test prompt")
        thought.add_generation("Random words without logical connection or structure.", "gpt-4", 0)

        # Mock low coherence score
        mock_coherence_validator.validate_async = AsyncMock(
            return_value=ValidationResult(
                passed=False,
                score=0.45,
                details={"coherence_score": 0.45, "threshold": 0.7},
                issues=["Low coherence score"],
                suggestions=["Improve logical flow", "Add transitions"],
            )
        )

        result = await mock_coherence_validator.validate_async(thought)

        assert result.passed is False
        assert result.score == 0.45
        assert len(result.issues) > 0
        assert len(result.suggestions) > 0


class TestValidatorIntegration:
    """Test validator integration with SifakaThought."""

    def test_multiple_validators_integration(self):
        """Test integration of multiple validators with thought."""
        thought = SifakaThought(prompt="Integration test")
        thought.add_generation("This is a test text for validator integration.", "gpt-4", 0)

        # Create multiple validators
        length_validator = MockValidator("length", should_pass=True)
        coherence_validator = MockValidator("coherence", should_pass=False)
        factual_validator = MockValidator("factual", should_pass=True)

        validators = [length_validator, coherence_validator, factual_validator]

        # Run all validators
        for validator in validators:
            result = validator.validate(thought)
            thought.add_validation(validator.name, result.passed, result.metadata)

        # Verify all validations were recorded
        assert len(thought.validations) == 3

        # Check individual results
        length_val = next(v for v in thought.validations if v.validator_name == "length")
        assert length_val.passed is True

        coherence_val = next(v for v in thought.validations if v.validator_name == "coherence")
        assert coherence_val.passed is False

        factual_val = next(v for v in thought.validations if v.validator_name == "factual")
        assert factual_val.passed is True

        # Overall validation should fail (one validator failed)
        assert thought.validation_passed() is False

    @pytest.mark.asyncio
    async def test_validator_performance_characteristics(self):
        """Test validator performance with large inputs."""
        import time

        # Create large text
        large_text = "This is a test sentence. " * 1000  # ~5000 words
        thought = SifakaThought(prompt="Performance test")
        thought.add_generation(large_text, "gpt-4", 0)

        validator = LengthValidator(min_length=100, max_length=10000)

        # Measure validation time
        start_time = time.time()
        result = await validator.validate_async(thought)
        end_time = time.time()

        # Should complete quickly even with large text
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second

        # Should still produce correct result
        assert result.passed is True
        assert result.metadata["word_count"] > 4000

    @pytest.mark.asyncio
    async def test_validator_error_recovery(self):
        """Test validator error recovery and graceful degradation."""
        thought = SifakaThought(prompt="Error recovery test")
        thought.add_generation("Test text", "gpt-4", 0)

        # Create validator that sometimes fails
        error_validator = MockValidator("error_prone", should_error=True)
        success_validator = MockValidator("reliable", should_pass=True)

        # Test error handling
        with pytest.raises(ValidationError):
            await error_validator.validate_async(thought)

        # Reliable validator should still work
        result = await success_validator.validate_async(thought)
        assert result.passed is True

        # System should be able to continue after errors
        assert success_validator.call_count == 1
