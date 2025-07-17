"""Tests for the pattern validator module."""

import pytest

from sifaka.core.models import SifakaResult
from sifaka.validators.pattern import (
    PatternValidator,
    create_citation_validator,
    create_code_validator,
    create_structured_validator,
)


class TestPatternValidator:
    """Test the PatternValidator class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Test", final_text="Test")

    def test_init_empty(self):
        """Test initialization with no patterns."""
        validator = PatternValidator()

        assert validator.required_patterns == {}
        assert validator.forbidden_patterns == {}
        assert validator.pattern_counts == {}

    def test_init_with_patterns(self):
        """Test initialization with patterns."""
        validator = PatternValidator(
            required_patterns={
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            },
            forbidden_patterns={"profanity": r"\b(bad|word)\b"},
            pattern_counts={"email": (1, 3)},
        )

        assert len(validator.required_patterns) == 1
        assert len(validator.forbidden_patterns) == 1
        assert validator.pattern_counts == {"email": (1, 3)}

    def test_name_property(self):
        """Test name property."""
        validator = PatternValidator()
        assert validator.name == "pattern_validator"

    @pytest.mark.asyncio
    async def test_validate_no_patterns(self, sample_result):
        """Test validation with no patterns configured."""
        validator = PatternValidator()
        text = "Any text here."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details == "No patterns configured"

    @pytest.mark.asyncio
    async def test_validate_required_pattern_found(self, sample_result):
        """Test validation when required pattern is found."""
        validator = PatternValidator(
            required_patterns={
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            }
        )
        text = "Contact us at test@example.com for more info."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "1 pattern(s) validated successfully" in result.details

    @pytest.mark.asyncio
    async def test_validate_required_pattern_not_found(self, sample_result):
        """Test validation when required pattern is not found."""
        validator = PatternValidator(
            required_patterns={
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            }
        )
        text = "Contact us for more info."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Required pattern 'email' not found" in result.details

    @pytest.mark.asyncio
    async def test_validate_forbidden_pattern_found(self, sample_result):
        """Test validation when forbidden pattern is found."""
        validator = PatternValidator(
            forbidden_patterns={"profanity": r"\b(bad|ugly)\b"}
        )
        text = "This is a bad example."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Forbidden pattern 'profanity' found: 'bad'" in result.details

    @pytest.mark.asyncio
    async def test_validate_forbidden_pattern_not_found(self, sample_result):
        """Test validation when forbidden pattern is not found."""
        validator = PatternValidator(
            forbidden_patterns={"profanity": r"\b(bad|ugly)\b"}
        )
        text = "This is a good example."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "1 pattern(s) validated successfully" in result.details

    @pytest.mark.asyncio
    async def test_validate_pattern_count_exact(self, sample_result):
        """Test validation with exact pattern count."""
        validator = PatternValidator(
            required_patterns={"number": r"\d+"},
            pattern_counts={"number": (2, 2)},
        )
        text = "There are 2 apples and 3 oranges."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_validate_pattern_count_too_few(self, sample_result):
        """Test validation with too few pattern matches."""
        validator = PatternValidator(
            required_patterns={"number": r"\d+"},
            pattern_counts={"number": (3, None)},
        )
        text = "There are 2 apples."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Pattern 'number' must occur at least 3 times, found 1" in result.details

    @pytest.mark.asyncio
    async def test_validate_pattern_count_too_many(self, sample_result):
        """Test validation with too many pattern matches."""
        validator = PatternValidator(
            required_patterns={"number": r"\d+"},
            pattern_counts={"number": (1, 2)},
        )
        text = "Numbers: 1, 2, 3, 4, 5."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "Pattern 'number' must occur at most 2 times, found 5" in result.details

    @pytest.mark.asyncio
    async def test_validate_multiple_patterns(self, sample_result):
        """Test validation with multiple patterns."""
        validator = PatternValidator(
            required_patterns={
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "url": r"https?://[^\s]+",
            },
            forbidden_patterns={
                "profanity": r"\b(bad|ugly)\b",
            },
        )
        text = "Visit https://example.com or email test@example.com"

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0
        assert "3 pattern(s) validated successfully" in result.details

    @pytest.mark.asyncio
    async def test_validate_multiline_patterns(self, sample_result):
        """Test validation with multiline patterns."""
        validator = PatternValidator(
            required_patterns={
                "section": r"^## .+$",
            }
        )
        text = """# Main Title
## Section One
Some content here.
## Section Two
More content."""

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_validate_long_match_truncation(self, sample_result):
        """Test that long matches are truncated in error messages."""
        validator = PatternValidator(
            forbidden_patterns={
                "long_text": r"[A-Za-z]{60,}",
            }
        )
        text = (
            "This contains a "
            + "verylongwordthatgoesonforsixtycharactersormoretotest" * 2
        )

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert result.score == 0.0
        assert "..." in result.details  # Should truncate the long match

    @pytest.mark.asyncio
    async def test_validate_multiple_issues_limit(self, sample_result):
        """Test that only first 3 issues are shown."""
        validator = PatternValidator(
            required_patterns={
                "pat1": r"MISSING1",
                "pat2": r"MISSING2",
                "pat3": r"MISSING3",
                "pat4": r"MISSING4",
            }
        )
        text = "This text has none of the required patterns."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        # Should only show first 3 issues
        issue_count = result.details.count("not found")
        assert issue_count == 3

    @pytest.mark.asyncio
    async def test_validate_complex_regex(self, sample_result):
        """Test validation with complex regex patterns."""
        validator = PatternValidator(
            required_patterns={
                "ipv4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                "date": r"\b\d{4}-\d{2}-\d{2}\b",
            }
        )
        text = "Server at 192.168.1.1 was updated on 2023-12-25."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_validate_case_sensitivity(self, sample_result):
        """Test that patterns are case sensitive by default."""
        validator = PatternValidator(required_patterns={"title": r"^TITLE:"})
        text = "title: This won't match"

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert "Required pattern 'title' not found" in result.details

    @pytest.mark.asyncio
    async def test_validate_empty_text(self, sample_result):
        """Test validation with empty text."""
        validator = PatternValidator(required_patterns={"something": r".+"})
        text = ""

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert "Required pattern 'something' not found" in result.details


class TestFactoryFunctions:
    """Test the convenience factory functions."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(original_text="Test", final_text="Test")

    def test_create_code_validator(self):
        """Test code validator factory."""
        validator = create_code_validator()

        assert "code_block" in validator.required_patterns
        assert validator.pattern_counts["code_block"] == (1, None)

    @pytest.mark.asyncio
    async def test_code_validator_with_code_block(self, sample_result):
        """Test code validator with valid code block."""
        validator = create_code_validator()
        text = """Here's some code:
```python
def hello():
    print("Hello, world!")
```
That's it!"""

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_code_validator_without_code_block(self, sample_result):
        """Test code validator without code block."""
        validator = create_code_validator()
        text = "Here's some text but no code block."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert "Pattern 'code_block' must occur at least 1 times" in result.details

    @pytest.mark.asyncio
    async def test_code_validator_with_language(self, sample_result):
        """Test code validator with language specified."""
        validator = create_code_validator()
        text = """```javascript
console.log("Hello");
```"""

        result = await validator.validate(text, sample_result)

        assert result.passed is True

    def test_create_citation_validator(self):
        """Test citation validator factory."""
        validator = create_citation_validator()

        assert "citation" in validator.required_patterns
        assert validator.pattern_counts["citation"] == (1, None)

    @pytest.mark.asyncio
    async def test_citation_validator_numeric_style(self, sample_result):
        """Test citation validator with numeric citations."""
        validator = create_citation_validator()
        text = "This is supported by research [1] and studies [2,3]."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_citation_validator_author_year_style(self, sample_result):
        """Test citation validator with author-year citations."""
        validator = create_citation_validator()
        text = "According to (Smith, 2023) and (Jones 2022), this is true."

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_citation_validator_no_citations(self, sample_result):
        """Test citation validator without citations."""
        validator = create_citation_validator()
        text = "This text makes claims but has no citations."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert "Pattern 'citation' must occur at least 1 times" in result.details

    def test_create_structured_validator(self):
        """Test structured document validator factory."""
        validator = create_structured_validator()

        assert "heading" in validator.required_patterns
        assert "list_item" in validator.required_patterns
        assert validator.pattern_counts["heading"] == (1, None)
        assert validator.pattern_counts["list_item"] == (2, None)

    @pytest.mark.asyncio
    async def test_structured_validator_markdown_style(self, sample_result):
        """Test structured validator with markdown formatting."""
        validator = create_structured_validator()
        text = """# Main Title

## Section 1
- First item
- Second item
- Third item

## Section 2
1. Numbered item
2. Another item"""

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_structured_validator_underline_style(self, sample_result):
        """Test structured validator with underline headings."""
        validator = create_structured_validator()
        text = """Main Title
==========

* First bullet
* Second bullet

Subsection
----------
- Another item
- And one more"""

        result = await validator.validate(text, sample_result)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_structured_validator_insufficient_structure(self, sample_result):
        """Test structured validator with insufficient structure."""
        validator = create_structured_validator()
        text = """# Only One Heading
And just one bullet point:
- Single item"""

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        assert (
            "Pattern 'list_item' must occur at least 2 times, found 1" in result.details
        )

    @pytest.mark.asyncio
    async def test_structured_validator_no_structure(self, sample_result):
        """Test structured validator with no structure."""
        validator = create_structured_validator()
        text = "Just plain text with no formatting whatsoever."

        result = await validator.validate(text, sample_result)

        assert result.passed is False
        # Should have multiple issues
        assert "heading" in result.details or "list_item" in result.details

    @pytest.mark.asyncio
    async def test_factory_validators_edge_cases(self, sample_result):
        """Test factory validators with edge cases."""
        # Code validator with minimal code block
        code_validator = create_code_validator()
        result = await code_validator.validate("```\npass\n```", sample_result)
        assert result.passed is True

        # Citation validator with mixed styles
        citation_validator = create_citation_validator()
        result = await citation_validator.validate(
            "See [1], (Smith, 2023), and [42]", sample_result
        )
        assert result.passed is True

        # Structured validator with nested lists
        structured_validator = create_structured_validator()
        text = """## Heading
        - Item 1
          - Nested item
        - Item 2"""
        result = await structured_validator.validate(text, sample_result)
        assert result.passed is True
