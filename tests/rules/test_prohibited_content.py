"""Tests for the prohibited content rule."""

import pytest
from typing import Dict, Any
from sifaka.rules.prohibited_content import ProhibitedContentRule
from sifaka.rules.base import RuleResult


@pytest.fixture
def rule():
    """Create a test rule instance."""
    return ProhibitedContentRule(
        name="test_prohibited_content",
        description="Test prohibited content rule",
        config={
            "prohibited_terms": ["bad", "worse", "worst"],
            "case_sensitive": False,
        },
    )


def test_initialization_validation():
    """Test initialization with invalid parameters."""
    # Test empty prohibited terms
    with pytest.raises(ValueError, match="prohibited_terms list cannot be empty"):
        ProhibitedContentRule(
            name="test",
            description="test",
            config={"prohibited_terms": [], "case_sensitive": False},
        )


def test_prohibited_rule_validation(rule):
    """Test validation of content for prohibited terms."""
    # Test content without prohibited terms
    result = rule.validate("This is a safe and acceptable message.")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert result.metadata["found_terms"] == []

    # Test content with prohibited terms
    result = rule.validate("This contains a bad word.")
    assert not result.passed
    assert "bad" in result.metadata["found_terms"]

    # Test case sensitivity
    rule_case_sensitive = ProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["Bad"], "case_sensitive": True},
    )
    result = rule_case_sensitive.validate("This contains a bad word.")
    assert result.passed  # Should pass because case doesn't match

    result = rule_case_sensitive.validate("This contains a Bad word.")
    assert not result.passed  # Should fail because case matches


def test_error_handling(rule):
    """Test error handling for invalid inputs."""
    # Test None input
    with pytest.raises(ValueError, match="Output must be a string"):
        rule.validate(None)

    # Test integer input
    with pytest.raises(ValueError, match="Output must be a string"):
        rule.validate(123)

    # Test list input
    with pytest.raises(ValueError, match="Output must be a string"):
        rule.validate([])

    # Test dict input
    with pytest.raises(ValueError, match="Output must be a string"):
        rule.validate({})


def test_metadata(rule):
    """Test metadata in validation results."""
    # Test with no prohibited terms
    result = rule.validate("Clean text")
    assert result.metadata == {"found_terms": []}

    # Test with one prohibited term
    result = rule.validate("This is bad")
    assert result.metadata == {"found_terms": ["bad"]}

    # Test with multiple prohibited terms
    result = rule.validate("This is bad and worse")
    assert set(result.metadata["found_terms"]) == {"bad", "worse"}
