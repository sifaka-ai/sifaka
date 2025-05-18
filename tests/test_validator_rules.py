"""
Tests for the validator rules.

This module contains tests for the functional validators in the Sifaka framework.
"""

import pytest
import re
from typing import Any, Dict, List, Optional

from sifaka.validators.rules import length, content, format, json_schema
from sifaka.results import ValidationResult


class TestLengthValidator:
    """Tests for the length validator function."""

    def test_no_constraints(self) -> None:
        """Test that at least one constraint is required."""
        with pytest.raises(ValueError):
            length()

    def test_min_words(self) -> None:
        """Test the minimum word count constraint."""
        validator = length(min_words=5)

        # Test with text that meets the constraint
        result = validator("This is a test with five words.")
        assert result.passed is True
        assert "within constraints" in result.message
        assert result._details["word_count"] >= 5  # At least 5 words

        # Test with text that doesn't meet the constraint
        result = validator("Too few words.")
        assert result.passed is False
        assert "too short" in result.message
        assert "words" in result.message
        assert result._details["word_count"] == 3

    def test_max_words(self) -> None:
        """Test the maximum word count constraint."""
        validator = length(max_words=5)

        # Test with text that meets the constraint
        result = validator("Five words is the maximum.")
        assert result.passed is True
        assert "within constraints" in result.message
        assert result._details["word_count"] == 5

        # Test with text that doesn't meet the constraint
        result = validator("This text has more than five words and should fail validation.")
        assert result.passed is False
        assert "too long" in result.message
        assert "words" in result.message
        assert result._details["word_count"] > 5

    def test_min_and_max_words(self) -> None:
        """Test both minimum and maximum word count constraints."""
        validator = length(min_words=3, max_words=5)

        # Test with text that meets both constraints
        result = validator("Four words is good.")
        assert result.passed is True
        assert "within constraints" in result.message
        assert result._details["word_count"] == 4

        # Test with text that doesn't meet the minimum constraint
        result = validator("Too few.")
        assert result.passed is False
        assert "too short" in result.message
        assert result._details["word_count"] < 3

        # Test with text that doesn't meet the maximum constraint
        result = validator("This text has more than five words and should fail validation.")
        assert result.passed is False
        assert "too long" in result.message
        assert result._details["word_count"] > 5

    def test_min_chars(self) -> None:
        """Test the minimum character count constraint."""
        validator = length(min_chars=20)

        # Test with text that meets the constraint
        result = validator("This text has more than twenty characters.")
        assert result.passed is True
        assert "within constraints" in result.message
        assert result._details["char_count"] > 20

        # Test with text that doesn't meet the constraint
        result = validator("Too short.")
        assert result.passed is False
        assert "too short" in result.message
        assert "characters" in result.message
        assert result._details["char_count"] < 20

    def test_max_chars(self) -> None:
        """Test the maximum character count constraint."""
        validator = length(max_chars=20)

        # Test with text that meets the constraint
        result = validator("Short enough.")
        assert result.passed is True
        assert "within constraints" in result.message
        assert result._details["char_count"] <= 20

        # Test with text that doesn't meet the constraint
        result = validator("This text has more than twenty characters and should fail validation.")
        assert result.passed is False
        assert "too long" in result.message
        assert "characters" in result.message
        assert result._details["char_count"] > 20

    def test_multiple_constraints(self) -> None:
        """Test multiple constraints together."""
        validator = length(min_words=3, max_words=10, min_chars=15, max_chars=100)

        # Test with text that meets all constraints
        result = validator("This text meets all the constraints.")
        assert result.passed is True
        assert "within constraints" in result.message

        # Test with text that fails one constraint
        result = validator("Short.")
        assert result.passed is False
        assert "too short" in result.message


class TestContentValidator:
    """Tests for the content validator function."""

    def test_no_constraints(self) -> None:
        """Test that at least one constraint is required."""
        with pytest.raises(ValueError):
            content()

    def test_required_terms(self) -> None:
        """Test the required terms constraint."""
        validator = content(required_terms=["python", "code"])

        # Test with text that contains all required terms
        result = validator("This is Python code.")
        assert result.passed is True
        assert "meets all requirements" in result.message

        # Test with text that doesn't contain all required terms
        result = validator("This is Python.")
        assert result.passed is False
        assert "missing required terms" in result.message
        assert "code" in result.message

    def test_forbidden_terms(self) -> None:
        """Test the forbidden terms constraint."""
        validator = content(forbidden_terms=["bug", "error"])

        # Test with text that doesn't contain any forbidden terms
        result = validator("This code works perfectly.")
        assert result.passed is True
        assert "meets all requirements" in result.message

        # Test with text that contains a forbidden term
        result = validator("This code has a bug.")
        assert result.passed is False
        assert "contains forbidden terms" in result.message
        assert "bug" in result.message

    def test_required_and_forbidden_terms(self) -> None:
        """Test both required and forbidden terms constraints."""
        validator = content(required_terms=["python", "code"], forbidden_terms=["bug", "error"])

        # Test with text that meets both constraints
        result = validator("This is Python code that works perfectly.")
        assert result.passed is True
        assert "meets all requirements" in result.message

        # Test with text that doesn't meet the required terms constraint
        result = validator("This is Python that works perfectly.")
        assert result.passed is False
        assert "missing required terms" in result.message

        # Test with text that doesn't meet the forbidden terms constraint
        result = validator("This is Python code that has a bug.")
        assert result.passed is False
        assert "contains forbidden terms" in result.message

    def test_case_sensitivity(self) -> None:
        """Test case sensitivity."""
        # Case-insensitive (default)
        validator = content(required_terms=["python"], forbidden_terms=["bug"])

        result = validator("This is PYTHON code.")
        assert result.passed is True

        result = validator("This code has a BUG.")
        assert result.passed is False

        # Case-sensitive
        validator = content(required_terms=["Python"], forbidden_terms=["bug"], case_sensitive=True)

        result = validator("This is Python code.")
        assert result.passed is True

        result = validator("This is PYTHON code.")
        assert result.passed is False

        # With case_sensitive=True, "BUG" is not the same as "bug"
        # But we also need to include the required term "Python"
        result = validator("This is Python code and has a BUG.")
        assert result.passed is True  # Should pass because it has "Python" and "BUG" is not "bug"

        result = validator("This code has a bug.")
        assert result.passed is False


class TestFormatValidator:
    """Tests for the format validator function."""

    def test_with_string_pattern(self) -> None:
        """Test with a string pattern."""
        validator = format(r"^\d{3}-\d{2}-\d{4}$")

        # Test with text that matches the pattern
        result = validator("123-45-6789")
        assert result.passed is True
        assert "matches the required format" in result.message

        # Test with text that doesn't match the pattern
        result = validator("12-345-6789")
        assert result.passed is False
        assert "does not match the required format" in result.message

    def test_with_compiled_pattern(self) -> None:
        """Test with a compiled pattern."""
        pattern = re.compile(r"^\d{3}-\d{2}-\d{4}$")
        validator = format(pattern)

        # Test with text that matches the pattern
        result = validator("123-45-6789")
        assert result.passed is True

        # Test with text that doesn't match the pattern
        result = validator("12-345-6789")
        assert result.passed is False

    def test_with_description(self) -> None:
        """Test with a description."""
        validator = format(r"^\d{3}-\d{2}-\d{4}$", description="SSN (e.g., 123-45-6789)")

        # Test with text that matches the pattern
        result = validator("123-45-6789")
        assert result.passed is True
        assert "SSN (e.g., 123-45-6789)" in result.message

        # Test with text that doesn't match the pattern
        result = validator("12-345-6789")
        assert result.passed is False
        assert "SSN (e.g., 123-45-6789)" in result.message


@pytest.mark.skipif(
    pytest.importorskip("jsonschema", reason="jsonschema not installed") is None,
    reason="jsonschema not installed",
)
class TestJsonSchemaValidator:
    """Tests for the json_schema validator function."""

    def test_with_valid_json(self) -> None:
        """Test with valid JSON that matches the schema."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        validator = json_schema(schema)

        # Test with valid JSON that matches the schema
        result = validator('{"name": "John"}')
        assert result.passed is True
        assert "valid and matches the schema" in result.message

        # Test with valid JSON that doesn't match the schema
        result = validator('{"name": 123}')
        assert result.passed is False
        assert "does not match the schema" in result.message

    def test_with_invalid_json(self) -> None:
        """Test with invalid JSON."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        validator = json_schema(schema)

        # Test with invalid JSON
        result = validator('{"name": "John"')
        assert result.passed is False
        # The error message might vary depending on the implementation
        assert result.passed is False

    def test_extract_json(self) -> None:
        """Test extracting JSON from text."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        validator = json_schema(schema, extract_json=True)

        # Test with JSON embedded in text
        result = validator('Here is some JSON: {"name": "John"} in the text.')
        assert result.passed is True
        assert "valid and matches the schema" in result.message

        # Test with no extractable JSON
        result = validator("Here is some text with no JSON.")
        assert result.passed is False
        assert "Could not extract JSON" in result.message

    def test_no_extract_json(self) -> None:
        """Test not extracting JSON from text."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        validator = json_schema(schema, extract_json=False)

        # Test with JSON embedded in text
        result = validator('Here is some JSON: {"name": "John"} in the text.')
        assert result.passed is False
        assert "not valid JSON" in result.message
