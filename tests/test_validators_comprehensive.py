"""Comprehensive tests for all validators."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import json

from sifaka.validators import LengthValidator, ContentValidator
from sifaka.core.interfaces import Validator
from sifaka.core.models import ValidationResult, SifakaResult


class TestLengthValidator:
    """Test LengthValidator thoroughly."""

    def test_init_valid_params(self):
        """Test validator initialization with valid parameters."""
        validator = LengthValidator(min_length=10, max_length=100)
        assert validator.min_length == 10
        assert validator.max_length == 100
        assert validator.name == "length"

    def test_init_min_only(self):
        """Test validator with only min_length."""
        validator = LengthValidator(min_length=50)
        assert validator.min_length == 50
        assert validator.max_length is None

    def test_init_max_only(self):
        """Test validator with only max_length."""
        validator = LengthValidator(max_length=200)
        assert validator.min_length is None
        assert validator.max_length == 200

    def test_init_invalid_params(self):
        """Test validator initialization with invalid parameters."""
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            LengthValidator(min_length=-1)

        with pytest.raises(ValueError, match="max_length must be non-negative"):
            LengthValidator(max_length=-1)

        with pytest.raises(
            ValueError, match="min_length cannot be greater than max_length"
        ):
            LengthValidator(min_length=100, max_length=50)

    def test_init_no_params(self):
        """Test validator initialization with no parameters."""
        # This should work - both params are optional
        validator = LengthValidator()
        assert validator.min_length is None
        assert validator.max_length is None

    @pytest.mark.asyncio
    async def test_validate_within_range(self):
        """Test validation of text within range."""
        validator = LengthValidator(min_length=10, max_length=50)
        text = "This is a test text that should pass validation."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert isinstance(result, ValidationResult)
        assert result.validator == "length"
        assert result.passed is True
        assert result.score == 1.0
        assert f"Text length: {len(text)}" in result.details

    @pytest.mark.asyncio
    async def test_validate_too_short(self):
        """Test validation of text that's too short."""
        validator = LengthValidator(min_length=50, max_length=100)
        text = "Short text"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert result.score == 0.2  # 10/50 = 0.2
        assert "Text length: 10 characters (minimum required: 50)" in result.details

    @pytest.mark.asyncio
    async def test_validate_too_long(self):
        """Test validation of text that's too long."""
        validator = LengthValidator(min_length=10, max_length=20)
        text = "This is a very long text that exceeds the maximum length limit."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert result.score == 1.0  # Score only considers min_length, 63/10 > 1.0
        assert "Text length: 63 characters (maximum allowed: 20)" in result.details

    @pytest.mark.asyncio
    async def test_validate_min_only_pass(self):
        """Test validation with only min_length that passes."""
        validator = LengthValidator(min_length=10)
        text = "This text is long enough to pass the minimum requirement."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_validate_min_only_fail(self):
        """Test validation with only min_length that fails."""
        validator = LengthValidator(min_length=100)
        text = "Too short"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert result.score == 0.09  # 9/100 = 0.09

    @pytest.mark.asyncio
    async def test_validate_max_only_pass(self):
        """Test validation with only max_length that passes."""
        validator = LengthValidator(max_length=100)
        text = "Short text"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_validate_max_only_fail(self):
        """Test validation with only max_length that fails."""
        validator = LengthValidator(max_length=10)
        text = "This text is way too long to pass validation"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert (
            result.score == 1.0
        )  # Score only considers min_length, text exceeds max but that doesn't affect score

    @pytest.mark.asyncio
    async def test_validate_empty_text(self):
        """Test validation of empty text."""
        validator = LengthValidator(min_length=1, max_length=100)
        text = ""

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert "Text length: 0 characters (minimum required: 1)" in result.details

    @pytest.mark.asyncio
    async def test_validate_whitespace_text(self):
        """Test validation of whitespace-only text."""
        validator = LengthValidator(min_length=5, max_length=10)
        text = "   \n\t  "

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        # Length should include whitespace
        assert len(text) == 7
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_unicode_text(self):
        """Test validation of unicode text."""
        validator = LengthValidator(min_length=5, max_length=20)
        text = "Hello 世界! 🌍"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_validate_exact_boundaries(self):
        """Test validation at exact length boundaries."""
        validator = LengthValidator(min_length=10, max_length=10)
        text = "Exactly10!"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert len(text) == 10
        assert result.passed is True
        assert result.score == 1.0


class TestContentValidator:
    """Test ContentValidator thoroughly."""

    def test_init_valid_params(self):
        """Test validator initialization with valid parameters."""
        validator = ContentValidator(
            required_terms=["hello", "world"], forbidden_terms=["bad", "evil"]
        )
        assert validator.required_terms == ["hello", "world"]
        assert validator.forbidden_terms == ["bad", "evil"]
        assert validator.name == "content"

    def test_init_required_only(self):
        """Test validator with only required terms."""
        validator = ContentValidator(required_terms=["test"])
        assert validator.required_terms == ["test"]
        assert validator.forbidden_terms == []

    def test_init_forbidden_only(self):
        """Test validator with only forbidden terms."""
        validator = ContentValidator(forbidden_terms=["spam"])
        assert validator.required_terms == []
        assert validator.forbidden_terms == ["spam"]

    def test_init_empty_lists(self):
        """Test validator with empty lists."""
        validator = ContentValidator(required_terms=[], forbidden_terms=[])
        assert validator.required_terms == []
        assert validator.forbidden_terms == []

    def test_init_no_params(self):
        """Test validator initialization with no parameters."""
        # This should work - both params are optional
        validator = ContentValidator()
        assert validator.required_terms == []
        assert validator.forbidden_terms == []

    @pytest.mark.asyncio
    async def test_validate_all_required_present(self):
        """Test validation when all required terms are present."""
        validator = ContentValidator(required_terms=["python", "programming"])
        text = "I love python programming and software development."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "All content requirements met" in result.details

    @pytest.mark.asyncio
    async def test_validate_missing_required_terms(self):
        """Test validation when required terms are missing."""
        validator = ContentValidator(required_terms=["python", "java", "rust"])
        text = "I love python programming but not java."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert result.score < 1.0
        assert "Missing required terms: ['rust']" in result.details

    @pytest.mark.asyncio
    async def test_validate_no_forbidden_terms(self):
        """Test validation when no forbidden terms are present."""
        validator = ContentValidator(forbidden_terms=["spam", "scam"])
        text = "This is a legitimate message about programming."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "All content requirements met" in result.details

    @pytest.mark.asyncio
    async def test_validate_forbidden_terms_present(self):
        """Test validation when forbidden terms are present."""
        validator = ContentValidator(forbidden_terms=["bad", "terrible"])
        text = "This is a bad example of terrible code."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Contains forbidden terms: ['bad', 'terrible']" in result.details

    @pytest.mark.asyncio
    async def test_validate_mixed_requirements(self):
        """Test validation with both required and forbidden terms."""
        validator = ContentValidator(
            required_terms=["good", "excellent"], forbidden_terms=["bad", "awful"]
        )
        text = "This is a good and excellent example."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_validate_mixed_requirements_fail_required(self):
        """Test validation failing on required terms."""
        validator = ContentValidator(
            required_terms=["good", "excellent"], forbidden_terms=["bad", "awful"]
        )
        text = "This is a good example only."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert "Missing required terms: ['excellent']" in result.details

    @pytest.mark.asyncio
    async def test_validate_mixed_requirements_fail_forbidden(self):
        """Test validation failing on forbidden terms."""
        validator = ContentValidator(
            required_terms=["good", "excellent"], forbidden_terms=["bad", "awful"]
        )
        text = "This is a good and excellent but bad example."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert "Contains forbidden terms: ['bad']" in result.details

    @pytest.mark.asyncio
    async def test_validate_case_sensitivity(self):
        """Test case-insensitive term matching."""
        validator = ContentValidator(
            required_terms=["Python", "PROGRAMMING"], forbidden_terms=["Bad", "SPAM"]
        )
        text = "I love python programming and avoid spam and bad practices."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        # Should fail because both required and forbidden terms are found (case insensitive)
        assert result.passed is False
        assert "Contains forbidden terms:" in result.details

    @pytest.mark.asyncio
    async def test_validate_partial_word_matching(self):
        """Test that partial word matches are detected."""
        validator = ContentValidator(required_terms=["test"])
        text = "This is testing the validator."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        # "testing" contains "test"
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_empty_text(self):
        """Test validation of empty text."""
        validator = ContentValidator(required_terms=["hello"])
        text = ""

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert "Missing required terms: ['hello']" in result.details

    @pytest.mark.asyncio
    async def test_validate_whitespace_text(self):
        """Test validation of whitespace-only text."""
        validator = ContentValidator(required_terms=["hello"])
        text = "   \n\t  "

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_validate_unicode_terms(self):
        """Test validation with unicode terms."""
        validator = ContentValidator(
            required_terms=["世界", "🌍"], forbidden_terms=["😈"]
        )
        text = "Hello 世界! Welcome to the 🌍."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_unicode_forbidden(self):
        """Test validation with unicode forbidden terms."""
        validator = ContentValidator(forbidden_terms=["😈", "💀"])
        text = "This text has evil 😈 emoji."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        assert result.passed is False
        assert "Contains forbidden terms: ['😈']" in result.details

    @pytest.mark.asyncio
    async def test_score_calculation_partial_match(self):
        """Test score calculation with partial matches."""
        validator = ContentValidator(required_terms=["one", "two", "three"])
        text = "This text contains one and two but not the third."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        # Should have 2/3 = 0.67 score (rounded)
        assert result.passed is False
        assert 0.6 < result.score < 0.7

    @pytest.mark.asyncio
    async def test_multiple_occurrences(self):
        """Test that multiple occurrences of terms are handled correctly."""
        validator = ContentValidator(required_terms=["test"], forbidden_terms=["bad"])
        text = "This test is a test of test functionality, not bad, bad, bad stuff."

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)

        # Should find required term but also forbidden term
        assert result.passed is False
        assert "Contains forbidden terms: ['bad']" in result.details


class TestCustomValidator:
    """Test custom validator implementation."""

    def test_custom_validator_interface(self):
        """Test that custom validators can be implemented."""

        class CustomValidator(Validator):
            def __init__(self, threshold: float = 0.5):
                self.threshold = threshold

            @property
            def name(self) -> str:
                return "custom"

            async def validate(
                self, text: str, result: SifakaResult
            ) -> ValidationResult:
                # Simple custom logic: pass if text length > threshold * 100
                score = min(len(text) / 100.0, 1.0)
                passed = score >= self.threshold

                return ValidationResult(
                    validator=self.name,
                    passed=passed,
                    score=score,
                    details=f"Custom validation: {score:.2f} >= {self.threshold}",
                )

        validator = CustomValidator(threshold=0.3)
        assert validator.name == "custom"
        assert validator.threshold == 0.3

    @pytest.mark.asyncio
    async def test_custom_validator_validation(self):
        """Test custom validator validation logic."""

        class CustomValidator(Validator):
            @property
            def name(self) -> str:
                return "custom"

            async def validate(
                self, text: str, result: SifakaResult
            ) -> ValidationResult:
                # Pass if text contains numbers
                has_numbers = any(c.isdigit() for c in text)
                return ValidationResult(
                    validator=self.name,
                    passed=has_numbers,
                    score=1.0 if has_numbers else 0.0,
                    details=f"Numbers found: {has_numbers}",
                )

        validator = CustomValidator()

        # Test with numbers
        text1 = "Test 123 text"
        dummy_result1 = SifakaResult(
            original_text=text1,
            final_text=text1,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text1, dummy_result1)
        assert result.passed is True
        assert result.score == 1.0

        # Test without numbers
        text2 = "Test text only"
        dummy_result2 = SifakaResult(
            original_text=text2,
            final_text=text2,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text2, dummy_result2)
        assert result.passed is False
        assert result.score == 0.0


class TestValidationResult:
    """Test ValidationResult model."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult objects."""
        result = ValidationResult(
            validator="test", passed=True, score=0.85, details="Test validation passed"
        )

        assert result.validator == "test"
        assert result.passed is True
        assert result.score == 0.85
        assert result.details == "Test validation passed"
        assert result.timestamp is not None

    def test_validation_result_repr(self):
        """Test ValidationResult string representation."""
        result = ValidationResult(
            validator="length", passed=False, score=0.5, details="Failed validation"
        )

        repr_str = repr(result)
        assert "ValidationResult" in repr_str
        assert "validator='length'" in repr_str
        assert "passed=False" in repr_str


class TestValidatorEdgeCases:
    """Test edge cases for validators."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test validators with very long text."""
        validator = LengthValidator(min_length=1000, max_length=100000)
        long_text = "A" * 50000

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=long_text,
            final_text=long_text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(long_text, dummy_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test validators with special characters."""
        validator = ContentValidator(
            required_terms=["@#$%", "!@#"], forbidden_terms=["***", "???"]
        )
        text = "Special chars: @#$% and !@# but not *** or ???"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)
        assert result.passed is False  # Text contains forbidden terms
        assert "Contains forbidden terms: ['***', '???']" in result.details

    @pytest.mark.asyncio
    async def test_newlines_and_tabs(self):
        """Test validators with newlines and tabs."""
        validator = LengthValidator(min_length=10, max_length=50)
        text = "Line 1\nLine 2\tTabbed\r\nWindows newline"

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_html_content(self):
        """Test validators with HTML content."""
        validator = ContentValidator(
            required_terms=[
                "<div",
                "class",
            ],  # Look for <div without closing > to match any div tag
            forbidden_terms=["<script>"],
        )
        text = '<div class="container">Content</div>'

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_json_content(self):
        """Test validators with JSON content."""
        validator = ContentValidator(required_terms=["key", "value"])
        text = '{"key": "value", "number": 42}'

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_performance_large_term_lists(self):
        """Test validator performance with large term lists."""
        # Create large lists of terms
        required_terms = [f"term_{i}" for i in range(100)]
        forbidden_terms = [f"bad_{i}" for i in range(100)]

        validator = ContentValidator(
            required_terms=required_terms[:10],  # Only require first 10
            forbidden_terms=forbidden_terms,
        )

        # Create text with some required terms
        text = "This text contains " + " ".join(required_terms[:10])

        # Create a dummy result
        dummy_result = SifakaResult(
            original_text=text,
            final_text=text,
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result = await validator.validate(text, dummy_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_concurrent_validation(self):
        """Test concurrent validation calls."""
        import asyncio

        validator = LengthValidator(min_length=10, max_length=100)
        texts = [f"Test text number {i} for concurrent validation" for i in range(10)]

        # Create dummy results for each text
        dummy_results = []
        for text in texts:
            dummy_result = SifakaResult(
                original_text=text,
                final_text=text,
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            dummy_results.append(dummy_result)

        # Run validations concurrently
        results = await asyncio.gather(
            *[
                validator.validate(text, dummy_result)
                for text, dummy_result in zip(texts, dummy_results)
            ]
        )

        assert len(results) == 10
        assert all(result.passed for result in results)
