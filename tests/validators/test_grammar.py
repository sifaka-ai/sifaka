"""
Tests for the GrammarValidator.
"""

import pytest
from unittest.mock import MagicMock, patch
from sifaka.validators import GrammarValidator
from sifaka.types import ValidationResult


@pytest.fixture
def mock_language_tool():
    """Create a mock LanguageTool instance for testing."""
    with patch("language_tool_python.LanguageTool") as mock_cls:
        # Create the mock instance
        mock_tool = MagicMock()
        mock_cls.return_value = mock_tool

        # Define sample matches for testing
        sample_match1 = MagicMock()
        sample_match1.ruleIssueType = "GRAMMAR"
        sample_match1.message = "Grammar error"
        sample_match1.offset = 10
        sample_match1.errorLength = 5
        sample_match1.context = "This is context with error here."
        sample_match1.offsetInContext = 15
        sample_match1.replacements = ["correction"]
        sample_match1.ruleId = "GRAMMAR_RULE_1"

        sample_match2 = MagicMock()
        sample_match2.ruleIssueType = "SPELLING"
        sample_match2.message = "Spelling error"
        sample_match2.offset = 20
        sample_match2.errorLength = 4
        sample_match2.context = "This has a speling mistake."
        sample_match2.offsetInContext = 10
        sample_match2.replacements = ["spelling"]
        sample_match2.ruleId = "SPELLING_RULE_1"

        sample_match3 = MagicMock()
        sample_match3.ruleIssueType = "PUNCTUATION"
        sample_match3.message = "Missing comma"
        sample_match3.offset = 5
        sample_match3.errorLength = 1
        sample_match3.context = "Hello world this needs a comma."
        sample_match3.offsetInContext = 5
        sample_match3.replacements = ["world,"]
        sample_match3.ruleId = "PUNCTUATION_RULE_1"

        # Return different sets of matches depending on the input text
        def check_side_effect(text):
            if text == "Perfect text.":
                return []
            elif text == "Text with grammar error.":
                return [sample_match1]
            elif text == "Text with spelling and punctuation errors.":
                return [sample_match2, sample_match3]
            else:
                return [sample_match1, sample_match2, sample_match3]

        mock_tool.check.side_effect = check_side_effect
        yield mock_tool


class TestGrammarValidator:
    """Test cases for GrammarValidator."""

    def test_initialization(self):
        """Test GrammarValidator initializes with correct parameters."""
        validator = GrammarValidator(
            max_errors=5,
            error_categories=["GRAMMAR", "SPELLING"],
            ignore_categories=["PUNCTUATION"],
            language="en-US",
            fail_on_any_error=True,
        )

        assert validator.max_errors == 5
        assert validator.error_categories == ["GRAMMAR", "SPELLING"]
        assert validator.ignore_categories == ["PUNCTUATION"]
        assert validator.language == "en-US"
        assert validator.fail_on_any_error is True
        assert validator._tool is None  # Tool should be lazy-loaded

    def test_empty_text(self):
        """Test handling of empty text."""
        validator = GrammarValidator()
        result = validator.validate("")

        assert result.passed is True
        assert "Empty text" in result.message
        assert result.score == 1.0
        assert not result.issues
        assert not result.suggestions

    def test_perfect_text(self, mock_language_tool):
        """Test validation of text with no grammar errors."""
        validator = GrammarValidator()
        # Mock tool is already loaded by fixture
        validator._tool = mock_language_tool

        result = validator.validate("Perfect text.")

        assert result.passed is True
        assert "No grammar errors" in result.message
        assert result.score == 1.0
        assert len(result.issues) == 0
        assert len(result.suggestions) == 0
        assert result.metadata["error_count"] == 0

    def test_text_with_errors(self, mock_language_tool):
        """Test validation of text with grammar errors."""
        validator = GrammarValidator()
        validator._tool = mock_language_tool

        result = validator.validate("Text with grammar error.")

        assert result.passed is False  # Default behavior fails on GRAMMAR errors
        assert "Grammar validation failed" in result.message
        assert result.score < 1.0
        assert len(result.issues) == 1
        assert len(result.suggestions) == 1
        assert "GRAMMAR" in result.issues[0]
        assert "correction" in result.suggestions[0]
        assert result.metadata["error_count"] == 1

    def test_max_errors_threshold(self, mock_language_tool):
        """Test validation with max_errors threshold."""
        # Allow up to 2 errors
        validator = GrammarValidator(max_errors=2)
        validator._tool = mock_language_tool

        # Text with one error should pass
        result1 = validator.validate("Text with grammar error.")
        assert result1.passed is True
        assert "Grammar validation passed" in result1.message

        # Text with three errors should fail
        result2 = validator.validate(
            "Text with multiple errors."
        )  # This will return 3 errors from our mock
        assert result2.passed is False
        assert "Grammar validation failed" in result2.message
        assert result2.metadata["error_count"] == 3

    def test_error_categories_filtering(self, mock_language_tool):
        """Test filtering errors by category."""
        # Only care about spelling errors
        validator = GrammarValidator(error_categories=["SPELLING"], fail_on_any_error=True)
        validator._tool = mock_language_tool

        # Text with grammar error but no spelling should pass
        result = validator.validate("Text with grammar error.")
        assert result.passed is True
        assert result.metadata["error_count"] == 0

        # Text with spelling error should fail
        result = validator.validate("Text with spelling and punctuation errors.")
        assert result.passed is False
        assert "SPELLING" in result.issues[0]
        assert result.metadata["error_count"] == 1  # Only 1 because we filtered out punctuation

    def test_ignore_categories(self, mock_language_tool):
        """Test ignoring specific error categories."""
        # Ignore spelling errors
        validator = GrammarValidator(ignore_categories=["SPELLING"], fail_on_any_error=True)
        validator._tool = mock_language_tool

        # Text with only spelling error should pass
        result = validator.validate("Text with spelling and punctuation errors.")
        assert result.metadata["error_count"] == 1  # Only punctuation error counted
        assert "PUNCTUATION" in result.issues[0]
        assert "SPELLING" not in str(result.issues)

    def test_fail_on_any_error(self, mock_language_tool):
        """Test fail_on_any_error option."""
        # Don't fail on any error
        validator1 = GrammarValidator(fail_on_any_error=False)
        validator1._tool = mock_language_tool

        # Text with only punctuation error should pass by default
        result1 = validator1.validate("Text with spelling and punctuation errors.")
        assert result1.passed is False  # Still fails because it has SPELLING error (critical)

        # Set up validator to fail on any error
        validator2 = GrammarValidator(
            error_categories=["PUNCTUATION"], fail_on_any_error=True  # Only care about punctuation
        )
        validator2._tool = mock_language_tool

        # Text with punctuation error should fail
        result2 = validator2.validate("Text with spelling and punctuation errors.")
        assert result2.passed is False
        assert "PUNCTUATION" in result2.issues[0]

    def test_import_error_handling(self):
        """Test handling of missing language_tool_python package."""
        with patch.dict("sys.modules", {"language_tool_python": None}):
            validator = GrammarValidator()
            with pytest.raises(ImportError) as excinfo:
                # Access tool property to trigger import
                _ = validator.tool
            assert "language_tool_python is required" in str(excinfo.value)
