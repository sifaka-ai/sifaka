"""Tests for pattern validator."""

import pytest
from sifaka.validators.pattern import (
    PatternValidator,
    create_code_validator,
    create_citation_validator,
    create_structured_validator,
)
from sifaka.core.models import SifakaResult


class TestPatternValidator:
    """Test the PatternValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(
            original_text="Original text",
            final_text="Final text"
        )

    def test_initialization_empty(self):
        """Test initialization with no patterns."""
        validator = PatternValidator()
        assert len(validator.required_patterns) == 0
        assert len(validator.forbidden_patterns) == 0
        assert len(validator.pattern_counts) == 0

    def test_initialization_with_patterns(self):
        """Test initialization with patterns."""
        validator = PatternValidator(
            required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"},
            forbidden_patterns={"profanity": r"\b(bad|words)\b"},
            pattern_counts={"email": (1, 5)}
        )
        assert len(validator.required_patterns) == 1
        assert len(validator.forbidden_patterns) == 1
        assert len(validator.pattern_counts) == 1

    def test_name_property(self):
        """Test validator name."""
        validator = PatternValidator()
        assert validator.name == "pattern_validator"

    @pytest.mark.asyncio
    async def test_no_patterns_configured(self, sample_result):
        """Test validation with no patterns configured."""
        validator = PatternValidator()
        text = "Any text should pass"
        result = await validator.validate(text, sample_result)
        assert result.passed is True
        assert result.score == 1.0
        assert "No patterns configured" in result.details

    @pytest.mark.asyncio
    async def test_required_pattern_found(self, sample_result):
        """Test with required pattern that is found."""
        validator = PatternValidator(
            required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"}
        )
        text = "Contact us at test@example.com for more info."
        result = await validator.validate(text, sample_result)
        assert result.passed is True
        assert result.score == 1.0
        assert "1 pattern(s) validated successfully" in result.details

    @pytest.mark.asyncio
    async def test_required_pattern_not_found(self, sample_result):
        """Test with required pattern that is not found."""
        validator = PatternValidator(
            required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"}
        )
        text = "Contact us for more info."
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert result.score == 0.0
        assert "Required pattern 'email' not found" in result.details

    @pytest.mark.asyncio
    async def test_forbidden_pattern_not_found(self, sample_result):
        """Test with forbidden pattern that is not found."""
        validator = PatternValidator(
            forbidden_patterns={"profanity": r"\b(bad|words)\b"}
        )
        text = "This is clean text."
        result = await validator.validate(text, sample_result)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_forbidden_pattern_found(self, sample_result):
        """Test with forbidden pattern that is found."""
        validator = PatternValidator(
            forbidden_patterns={"profanity": r"\b(bad|words)\b"}
        )
        text = "This text contains bad language."
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert result.score == 0.0
        assert "Forbidden pattern 'profanity' found: 'bad'" in result.details

    @pytest.mark.asyncio
    async def test_pattern_count_minimum_met(self, sample_result):
        """Test pattern count with minimum requirement met."""
        validator = PatternValidator(
            required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"},
            pattern_counts={"email": (2, None)}
        )
        text = "Contact test@example.com or admin@example.com"
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_pattern_count_minimum_not_met(self, sample_result):
        """Test pattern count with minimum requirement not met."""
        validator = PatternValidator(
            required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"},
            pattern_counts={"email": (2, None)}
        )
        text = "Contact test@example.com"
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "Pattern 'email' must occur at least 2 times, found 1" in result.details

    @pytest.mark.asyncio
    async def test_pattern_count_maximum_exceeded(self, sample_result):
        """Test pattern count with maximum limit exceeded."""
        validator = PatternValidator(
            required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"},
            pattern_counts={"email": (1, 2)}
        )
        text = "Contact test@example.com, admin@example.com, and support@example.com"
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "Pattern 'email' must occur at most 2 times, found 3" in result.details

    @pytest.mark.asyncio
    async def test_pattern_count_range_met(self, sample_result):
        """Test pattern count within specified range."""
        validator = PatternValidator(
            required_patterns={"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"},
            pattern_counts={"email": (1, 3)}
        )
        text = "Contact test@example.com or admin@example.com"
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_multiple_patterns_all_valid(self, sample_result):
        """Test multiple patterns all valid."""
        validator = PatternValidator(
            required_patterns={
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "url": r"https?://[\w.-]+(?:\.[\w\.-]+)+"
            },
            forbidden_patterns={
                "script": r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>"
            }
        )
        text = "Visit https://example.com or email test@example.com"
        result = await validator.validate(text, sample_result)
        assert result.passed is True
        assert "3 pattern(s) validated successfully" in result.details

    @pytest.mark.asyncio
    async def test_multiple_patterns_some_invalid(self, sample_result):
        """Test multiple patterns with some invalid."""
        validator = PatternValidator(
            required_patterns={
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
            }
        )
        text = "Email test@example.com but no phone"
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "Required pattern 'phone' not found" in result.details

    @pytest.mark.asyncio
    async def test_multiline_pattern(self, sample_result):
        """Test pattern matching across multiple lines."""
        validator = PatternValidator(
            required_patterns={
                "code_block": r"```[\w]*\n[\s\S]+?\n```"
            }
        )
        text = """Here is some code:
```python
def hello():
    print("Hello")
```
"""
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_forbidden_pattern_truncation(self, sample_result):
        """Test long forbidden pattern match is truncated."""
        validator = PatternValidator(
            forbidden_patterns={
                "long_text": r"This is a very long text.*pattern"
            }
        )
        text = "This is a very long text that goes on and on and on with more text pattern"
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "..." in result.details  # Should truncate long match

    @pytest.mark.asyncio
    async def test_multiple_issues_truncation(self, sample_result):
        """Test that only first 3 issues are shown."""
        validator = PatternValidator(
            required_patterns={
                "p1": r"pattern1",
                "p2": r"pattern2",
                "p3": r"pattern3",
                "p4": r"pattern4",
            }
        )
        text = "No patterns here"
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        # Should only show first 3 issues
        assert result.details.count(";") == 2  # 3 issues separated by 2 semicolons

    @pytest.mark.asyncio
    async def test_case_sensitive_patterns(self, sample_result):
        """Test case-sensitive pattern matching."""
        validator = PatternValidator(
            required_patterns={"uppercase": r"[A-Z]+"}
        )
        
        # Should find uppercase
        text1 = "This has UPPERCASE letters"
        result1 = await validator.validate(text1, sample_result)
        assert result1.passed is True
        
        # Should not find uppercase
        text2 = "this has no uppercase letters"
        result2 = await validator.validate(text2, sample_result)
        assert result2.passed is False


class TestValidatorFactories:
    """Test the convenience factory functions."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(
            original_text="Original text",
            final_text="Final text"
        )

    def test_create_code_validator(self):
        """Test code validator factory."""
        validator = create_code_validator()
        assert "code_block" in validator.required_patterns
        assert "code_block" in validator.pattern_counts
        assert validator.pattern_counts["code_block"] == (1, None)

    @pytest.mark.asyncio
    async def test_code_validator_valid(self, sample_result):
        """Test code validator with valid code block."""
        validator = create_code_validator()
        text = """Here is some code:
```python
def hello():
    print("Hello, World!")
```
"""
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_code_validator_invalid(self, sample_result):
        """Test code validator without code block."""
        validator = create_code_validator()
        text = "Here is some text without code blocks"
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "at least 1 times, found 0" in result.details

    @pytest.mark.asyncio
    async def test_code_validator_multiple_blocks(self, sample_result):
        """Test code validator with multiple code blocks."""
        validator = create_code_validator()
        text = """First block:
```python
print("one")
```

Second block:
```js
console.log("two")
```
"""
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    def test_create_citation_validator(self):
        """Test citation validator factory."""
        validator = create_citation_validator()
        assert "citation" in validator.required_patterns
        assert "citation" in validator.pattern_counts

    @pytest.mark.asyncio
    async def test_citation_validator_numeric(self, sample_result):
        """Test citation validator with numeric citations."""
        validator = create_citation_validator()
        text = "This is supported by research [1] and further studies [2]."
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_citation_validator_author_year(self, sample_result):
        """Test citation validator with author-year citations."""
        validator = create_citation_validator()
        text = "As shown by (Smith, 2023) and (Jones 2022)."
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_citation_validator_no_citations(self, sample_result):
        """Test citation validator without citations."""
        validator = create_citation_validator()
        text = "This text makes claims without citations."
        result = await validator.validate(text, sample_result)
        assert result.passed is False

    def test_create_structured_validator(self):
        """Test structured document validator factory."""
        validator = create_structured_validator()
        assert "heading" in validator.required_patterns
        assert "list_item" in validator.required_patterns
        assert validator.pattern_counts["heading"] == (1, None)
        assert validator.pattern_counts["list_item"] == (2, None)

    @pytest.mark.asyncio
    async def test_structured_validator_markdown_headings(self, sample_result):
        """Test structured validator with markdown headings."""
        validator = create_structured_validator()
        text = """# Main Heading

Some content here.

## Subheading

- First item
- Second item
- Third item
"""
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_structured_validator_underline_headings(self, sample_result):
        """Test structured validator with underline headings."""
        validator = create_structured_validator()
        text = """Main Heading
============

Some content here.

Subheading
----------

1. First item
2. Second item
"""
        result = await validator.validate(text, sample_result)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_structured_validator_insufficient_list_items(self, sample_result):
        """Test structured validator with too few list items."""
        validator = create_structured_validator()
        text = """# Heading

- Only one item
"""
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "list_item" in result.details
        assert "at least 2 times" in result.details

    @pytest.mark.asyncio
    async def test_structured_validator_no_heading(self, sample_result):
        """Test structured validator without headings."""
        validator = create_structured_validator()
        text = """Just plain text with some lists:
- Item 1
- Item 2
- Item 3
"""
        result = await validator.validate(text, sample_result)
        assert result.passed is False
        assert "heading" in result.details