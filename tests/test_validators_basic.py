"""Tests for basic validators."""

import pytest

from sifaka.core.models import SifakaResult
from sifaka.validators.base import ValidatorConfig
from sifaka.validators.basic import (
    ContentValidator,
    FormatValidator,
    LengthValidator,
)


class TestLengthValidator:
    """Test the LengthValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_valid(self):
        """Test valid initialization."""
        validator = LengthValidator(min_length=10, max_length=100)
        assert validator.min_length == 10
        assert validator.max_length == 100

    def test_initialization_negative_min(self):
        """Test initialization with negative min_length."""
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            LengthValidator(min_length=-1)

    def test_initialization_negative_max(self):
        """Test initialization with negative max_length."""
        with pytest.raises(ValueError, match="max_length must be non-negative"):
            LengthValidator(max_length=-1)

    def test_initialization_min_greater_than_max(self):
        """Test initialization with min > max."""
        with pytest.raises(
            ValueError, match="min_length cannot be greater than max_length"
        ):
            LengthValidator(min_length=100, max_length=50)

    def test_name_property(self):
        """Test validator name."""
        validator = LengthValidator()
        assert validator.name == "length"

    @pytest.mark.asyncio
    async def test_no_constraints(self, sample_result):
        """Test with no length constraints."""
        validator = LengthValidator()
        result = await validator.validate("Any length text", sample_result)
        assert result.passed is True
        assert result.score == 1.0
        assert "Text length: 15 characters" in result.details

    @pytest.mark.asyncio
    async def test_min_length_passed(self, sample_result):
        """Test text meeting minimum length."""
        validator = LengthValidator(min_length=10)
        result = await validator.validate("This is a longer text", sample_result)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_min_length_failed(self, sample_result):
        """Test text below minimum length."""
        validator = LengthValidator(min_length=20)
        text = "Short text"  # 10 chars
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert result.score == 0.5  # 10/20
        assert "minimum required: 20" in result.details

    @pytest.mark.asyncio
    async def test_max_length_passed(self, sample_result):
        """Test text within maximum length."""
        validator = LengthValidator(max_length=50)
        result = await validator.validate("This is within the limit", sample_result)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_max_length_failed(self, sample_result):
        """Test text exceeding maximum length."""
        validator = LengthValidator(max_length=10)
        text = "This text is too long"  # 21 chars
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert result.score == pytest.approx(10 / 21)
        assert "maximum allowed: 10" in result.details

    @pytest.mark.asyncio
    async def test_both_constraints_passed(self, sample_result):
        """Test text meeting both min and max constraints."""
        validator = LengthValidator(min_length=10, max_length=30)
        text = "This is just right"  # 18 chars
        result = await validator.validate(text, sample_result)
        assert result.passed is True
        # Score should be high (centered in range)
        assert result.score > 0.9

    @pytest.mark.asyncio
    async def test_centering_score(self, sample_result):
        """Test centering score calculation."""
        validator = LengthValidator(min_length=10, max_length=30)

        # Text at exact center (20 chars)
        result = await validator.validate("x" * 20, sample_result)
        assert result.score == 1.0

        # Text near edges gets slightly lower score
        result = await validator.validate("x" * 11, sample_result)
        assert 0.9 < result.score < 1.0

        result = await validator.validate("x" * 29, sample_result)
        assert 0.9 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_type_error_non_string(self, sample_result):
        """Test with non-string input."""
        validator = LengthValidator()
        with pytest.raises(TypeError, match="Expected str, got int"):
            await validator._perform_validation(123, sample_result)

    @pytest.mark.asyncio
    async def test_empty_string(self, sample_result):
        """Test with empty string."""
        validator = LengthValidator(min_length=5)
        result = await validator.validate("", sample_result)
        assert result.passed is False
        assert result.score == 0.0
        assert "Text length: 0 characters" in result.details

    @pytest.mark.asyncio
    async def test_with_config(self, sample_result):
        """Test with validator config."""
        config = ValidatorConfig(enabled=True, strict=True)
        validator = LengthValidator(min_length=10, config=config)
        assert validator.config == config


class TestContentValidator:
    """Test the ContentValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization_default(self):
        """Test default initialization."""
        validator = ContentValidator()
        assert validator.required_terms == []
        assert validator.forbidden_terms == []
        assert validator.case_sensitive is False

    def test_initialization_with_terms(self):
        """Test initialization with terms."""
        validator = ContentValidator(
            required_terms=["hello", "world"],
            forbidden_terms=["bad", "word"],
            case_sensitive=True,
        )
        assert validator.required_terms == ["hello", "world"]
        assert validator.forbidden_terms == ["bad", "word"]
        assert validator.case_sensitive is True

    def test_initialization_invalid_required_terms(self):
        """Test initialization with non-string required terms."""
        with pytest.raises(TypeError, match="All required_terms must be strings"):
            ContentValidator(required_terms=["valid", 123])

    def test_initialization_invalid_forbidden_terms(self):
        """Test initialization with non-string forbidden terms."""
        with pytest.raises(TypeError, match="All forbidden_terms must be strings"):
            ContentValidator(forbidden_terms=["valid", None])

    def test_name_property(self):
        """Test validator name."""
        validator = ContentValidator()
        assert validator.name == "content"

    @pytest.mark.asyncio
    async def test_no_terms(self, sample_result):
        """Test with no terms configured."""
        validator = ContentValidator()
        result = await validator.validate("Any text content", sample_result)
        assert result.passed is True
        assert result.score == 1.0
        assert "All content requirements met" in result.details

    @pytest.mark.asyncio
    async def test_required_terms_present(self, sample_result):
        """Test with all required terms present."""
        validator = ContentValidator(required_terms=["hello", "world"])
        result = await validator.validate("Hello world from Python", sample_result)
        assert result.passed is True
        assert result.score == 1.0
        assert "All content requirements met" in result.details

    @pytest.mark.asyncio
    async def test_required_terms_missing(self, sample_result):
        """Test with missing required terms."""
        validator = ContentValidator(required_terms=["hello", "world", "python"])
        result = await validator.validate("Hello world", sample_result)
        assert result.passed is False
        assert result.score == pytest.approx(2 / 3)
        assert "Missing required terms: ['python']" in result.details

    @pytest.mark.asyncio
    async def test_forbidden_terms_absent(self, sample_result):
        """Test with no forbidden terms found."""
        validator = ContentValidator(forbidden_terms=["bad", "evil"])
        result = await validator.validate("This is good content", sample_result)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_forbidden_terms_present(self, sample_result):
        """Test with forbidden terms present."""
        validator = ContentValidator(forbidden_terms=["bad", "evil"])
        result = await validator.validate("This has bad and evil words", sample_result)
        assert result.passed is False
        assert result.score == 0.0
        assert "Contains forbidden terms: ['bad', 'evil']" in result.details

    @pytest.mark.asyncio
    async def test_case_sensitive_matching(self, sample_result):
        """Test case-sensitive matching."""
        validator = ContentValidator(
            required_terms=["Hello", "World"], case_sensitive=True
        )

        # Exact case match
        result = await validator.validate("Hello World", sample_result)
        assert result.passed is True

        # Wrong case
        result = await validator.validate("hello world", sample_result)
        assert result.passed is False
        assert "Missing required terms: ['Hello', 'World']" in result.details

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, sample_result):
        """Test case-insensitive matching (default)."""
        validator = ContentValidator(
            required_terms=["hello", "world"], forbidden_terms=["BAD"]
        )

        # Different case should still match
        result = await validator.validate("HELLO WORLD", sample_result)
        assert result.passed is True

        # Forbidden term in different case
        result = await validator.validate("This is bad", sample_result)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_mixed_requirements(self, sample_result):
        """Test with both required and forbidden terms."""
        validator = ContentValidator(
            required_terms=["python", "code"], forbidden_terms=["error", "bug"]
        )

        # All requirements met
        result = await validator.validate("Python code is great", sample_result)
        assert result.passed is True
        assert result.score == 1.0

        # Missing required and has forbidden
        result = await validator.validate("Code has an error", sample_result)
        assert result.passed is False
        assert result.score == 0.5  # 2/4 checks passed

    @pytest.mark.asyncio
    async def test_type_error_non_string(self, sample_result):
        """Test with non-string input."""
        validator = ContentValidator()
        with pytest.raises(TypeError, match="Expected str, got list"):
            await validator._perform_validation([], sample_result)

    @pytest.mark.asyncio
    async def test_empty_string(self, sample_result):
        """Test with empty string."""
        validator = ContentValidator(required_terms=["content"])
        result = await validator.validate("", sample_result)
        assert result.passed is False
        assert "Missing required terms: ['content']" in result.details


class TestFormatValidator:
    """Test the FormatValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Original text", final_text="Final text")

    def test_initialization(self):
        """Test initialization."""
        validator = FormatValidator(
            required_sections=["Introduction", "Conclusion"],
            min_paragraphs=2,
            max_paragraphs=10,
        )
        assert validator.required_sections == ["Introduction", "Conclusion"]
        assert validator.min_paragraphs == 2
        assert validator.max_paragraphs == 10

    def test_name_property(self):
        """Test validator name."""
        validator = FormatValidator()
        assert validator.name == "format"

    @pytest.mark.asyncio
    async def test_no_requirements(self, sample_result):
        """Test with no format requirements."""
        validator = FormatValidator()
        result = await validator.validate("Simple text", sample_result)
        assert result.passed is True
        assert result.score == 1.0
        assert "Paragraphs: 1" in result.details

    @pytest.mark.asyncio
    async def test_paragraph_counting_double_newline(self, sample_result):
        """Test paragraph counting with double newlines."""
        validator = FormatValidator()
        text = """First paragraph.

Second paragraph.

Third paragraph."""
        result = await validator.validate(text, sample_result)
        assert "Paragraphs: 3" in result.details

    @pytest.mark.asyncio
    async def test_paragraph_counting_single_newline(self, sample_result):
        """Test paragraph counting with single newlines."""
        validator = FormatValidator()
        text = """First paragraph.
Second paragraph.
Third paragraph."""
        result = await validator.validate(text, sample_result)
        assert "Paragraphs: 3" in result.details

    @pytest.mark.asyncio
    async def test_min_paragraphs_passed(self, sample_result):
        """Test meeting minimum paragraph requirement."""
        validator = FormatValidator(min_paragraphs=2)
        text = "First para.\n\nSecond para.\n\nThird para."
        result = await validator.validate(text, sample_result)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_min_paragraphs_failed(self, sample_result):
        """Test failing minimum paragraph requirement."""
        validator = FormatValidator(min_paragraphs=3)
        text = "First para.\n\nSecond para."
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "Need at least 3 paragraphs" in result.details

    @pytest.mark.asyncio
    async def test_max_paragraphs_passed(self, sample_result):
        """Test within maximum paragraph limit."""
        validator = FormatValidator(max_paragraphs=3)
        text = "First.\n\nSecond."
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_max_paragraphs_failed(self, sample_result):
        """Test exceeding maximum paragraph limit."""
        validator = FormatValidator(max_paragraphs=2)
        text = "First.\n\nSecond.\n\nThird."
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "Too many paragraphs (max: 2)" in result.details

    @pytest.mark.asyncio
    async def test_required_sections_present(self, sample_result):
        """Test with all required sections present."""
        validator = FormatValidator(required_sections=["Introduction", "Conclusion"])
        text = """Introduction: This is the intro.

Main content here.

Conclusion: This wraps it up."""
        result = await validator.validate(text, sample_result)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_required_sections_missing(self, sample_result):
        """Test with missing required sections."""
        validator = FormatValidator(
            required_sections=["Introduction", "Methods", "Results"]
        )
        text = """Introduction: Here we begin.

Some content without other sections."""
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "Missing sections: ['Methods', 'Results']" in result.details

    @pytest.mark.asyncio
    async def test_case_insensitive_sections(self, sample_result):
        """Test case-insensitive section matching."""
        validator = FormatValidator(required_sections=["INTRODUCTION"])
        text = "introduction: here it is"
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_score_calculation(self, sample_result):
        """Test score calculation with multiple requirements."""
        validator = FormatValidator(
            required_sections=["Intro", "Body", "Conclusion"],
            min_paragraphs=2,
            max_paragraphs=5,
        )

        # All requirements met (5 total checks)
        text = """Intro: Start here.

Body: Main content.

Conclusion: The end."""
        result = await validator.validate(text, sample_result)
        assert result.score == 1.0

        # 3/5 requirements met
        text = "Just intro here with body"
        result = await validator.validate(text, sample_result)
        assert result.score == 0.6  # Missing conclusion, wrong paragraph count

    @pytest.mark.asyncio
    async def test_empty_paragraphs_ignored(self, sample_result):
        """Test that empty paragraphs are ignored."""
        validator = FormatValidator()
        text = """First paragraph.


Second paragraph."""  # Extra blank line
        result = await validator.validate(text, sample_result)
        assert "Paragraphs: 2" in result.details

    @pytest.mark.asyncio
    async def test_whitespace_paragraphs_ignored(self, sample_result):
        """Test that whitespace-only paragraphs are ignored."""
        validator = FormatValidator()
        text = """First paragraph.


Second paragraph."""
        result = await validator.validate(text, sample_result)
        assert "Paragraphs: 2" in result.details
