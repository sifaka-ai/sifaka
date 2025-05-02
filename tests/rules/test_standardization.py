"""
Tests for standardized rule and validator patterns.
"""

from unittest.mock import patch

from sifaka.classifiers.base import ClassificationResult
from sifaka.rules.content.prohibited import create_prohibited_content_rule
from sifaka.rules.formatting.style import create_style_rule, CapitalizationStyle


class MockClassifier:
    """Mock classifier for testing."""

    def __init__(self, **kwargs):
        """Initialize with any kwargs."""
        self.config = kwargs.get("config")

    def classify(self, text):
        """Return clean for clean text, profane for text with prohibited terms."""
        if any(term in text.lower() for term in ["inappropriate", "offensive"]):
            return ClassificationResult(
                label="profane", confidence=0.9, metadata={"terms_found": ["inappropriate"]}
            )
        return ClassificationResult(label="clean", confidence=0.9, metadata={})


@patch("sifaka.rules.content.prohibited.ProfanityClassifier", MockClassifier)
def test_prohibited_content_rule_delegation():
    """Test that ProhibitedContentRule properly delegates to its validator."""
    # Create a rule using the factory function
    rule = create_prohibited_content_rule(
        terms=["inappropriate", "offensive"],
        threshold=0.5,
    )

    # Test with clean text
    result = rule.validate("This is a clean test.")
    assert result.passed is True
    assert "No prohibited content detected" in result.message
    assert "rule_id" in result.metadata

    # Test with prohibited content
    result = rule.validate("This is inappropriate content.")
    assert result.passed is False
    assert "Prohibited content detected" in result.message
    assert "rule_id" in result.metadata


def test_style_rule_delegation():
    """Test that StyleRule properly delegates to its validator."""
    # Create a rule using the factory function
    rule = create_style_rule(
        name="sentence_case_rule",
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True,
    )

    # Test with valid text
    result = rule.validate("This is a properly formatted sentence.")
    assert result.passed is True
    assert "Style validation successful" in result.message
    assert "rule_id" in result.metadata
    assert result.metadata["rule_id"] == "sentence_case_rule"

    # Test with invalid text (lowercase)
    result = rule.validate("this is not properly formatted")
    assert result.passed is False
    assert "Text should be in sentence case" in result.message
    assert "rule_id" in result.metadata
    assert result.metadata["rule_id"] == "sentence_case_rule"


@patch("sifaka.rules.content.prohibited.ProfanityClassifier", MockClassifier)
def test_empty_text_handling():
    """Test that empty text is handled consistently across rules."""
    # Create rules
    prohibited_rule = create_prohibited_content_rule()
    style_rule = create_style_rule(capitalization=CapitalizationStyle.SENTENCE_CASE)

    # Test with empty string
    prohibited_result = prohibited_rule.validate("")
    style_result = style_rule.validate("")

    # Both should pass with the same message
    assert prohibited_result.passed is True
    assert style_result.passed is True
    assert prohibited_result.message == "Empty text validation skipped"
    assert style_result.message == "Empty text validation skipped"
    assert prohibited_result.metadata.get("reason") == "empty_input"
    assert style_result.metadata.get("reason") == "empty_input"

    # Test with whitespace-only string
    prohibited_result = prohibited_rule.validate("   \n   ")
    style_result = style_rule.validate("   \n   ")

    # Both should pass with the same message
    assert prohibited_result.passed is True
    assert style_result.passed is True
    assert prohibited_result.message == "Empty text validation skipped"
    assert style_result.message == "Empty text validation skipped"
    assert prohibited_result.metadata.get("reason") == "empty_input"
    assert style_result.metadata.get("reason") == "empty_input"
