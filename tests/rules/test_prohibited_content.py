"""Tests for the ProhibitedContentRule."""

import pytest
from typing import Dict, Any

from sifaka.rules.prohibited_content import ProhibitedContentRule
from sifaka.rules.base import RuleResult


class TestProhibitedContentRule(ProhibitedContentRule):
    """Test implementation of ProhibitedContentRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        found_terms = []
        for term in self.prohibited_terms:
            if self.case_sensitive:
                if term in output:
                    found_terms.append(term)
            else:
                if term.lower() in output.lower():
                    found_terms.append(term)

        passed = len(found_terms) == 0
        message = (
            "No prohibited terms found"
            if passed
            else f"Found prohibited terms: {', '.join(found_terms)}"
        )

        return RuleResult(passed=passed, message=message, metadata={"found_terms": found_terms})


@pytest.fixture
def rule():
    """Create a TestProhibitedContentRule instance."""
    return TestProhibitedContentRule(
        name="test_prohibited_content",
        description="Test prohibited content rule",
        config={"prohibited_terms": ["bad", "worse", "worst"], "case_sensitive": False},
    )


def test_initialization():
    """Test rule initialization with different parameters."""
    # Test default initialization
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["bad", "worse", "worst"], "case_sensitive": False},
    )
    assert rule.name == "test"
    assert rule.prohibited_terms == ["bad", "worse", "worst"]
    assert not rule.case_sensitive

    # Test case sensitive initialization
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["Bad", "Worse", "Worst"], "case_sensitive": True},
    )
    assert rule.case_sensitive


def test_initialization_validation():
    """Test initialization with invalid parameters."""
    # Test empty prohibited terms
    with pytest.raises(ValueError, match="Prohibited terms list cannot be empty"):
        TestProhibitedContentRule(
            name="test",
            description="test",
            config={"prohibited_terms": [], "case_sensitive": False},
        )

    # Test invalid prohibited terms type
    with pytest.raises(ValueError, match="Prohibited terms must be a list of strings"):
        TestProhibitedContentRule(
            name="test",
            description="test",
            config={"prohibited_terms": "not a list", "case_sensitive": False},
        )


def test_prohibited_rule_validation(rule):
    """Test validation of content for prohibited terms."""
    # Test content without prohibited terms
    result = rule.validate("This is a safe and acceptable message.")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert result.metadata["found_terms"] is None

    # Test content with prohibited terms
    result = rule.validate("This content contains bad and worse terms.")
    assert not result.passed
    assert "prohibited content detected" in result.message.lower()
    assert len(result.metadata["found_terms"]) == 2
    assert "bad" in result.metadata["found_terms"]
    assert "worse" in result.metadata["found_terms"]

    # Test case sensitivity
    rule.case_sensitive = True
    result = rule.validate("This content contains BAD and WORSE terms.")
    assert result.passed  # Should pass because terms don't match case
    assert result.metadata["found_terms"] is None

    rule.case_sensitive = False
    result = rule.validate("This content contains BAD and WORSE terms.")
    assert not result.passed  # Should fail because case is ignored
    assert len(result.metadata["found_terms"]) == 2


def test_case_sensitivity():
    """Test case sensitive and insensitive matching."""
    # Case insensitive rule
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["bad"], "case_sensitive": False},
    )
    assert rule.validate("This is BAD.").passed is False
    assert rule.validate("This is Bad.").passed is False
    assert rule.validate("This is bad.").passed is False

    # Case sensitive rule
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["Bad"], "case_sensitive": True},
    )
    assert rule.validate("This is BAD.").passed is True
    assert rule.validate("This is Bad.").passed is False
    assert rule.validate("This is bad.").passed is True


def test_partial_word_matching():
    """Test matching behavior for partial words."""
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["bad"], "case_sensitive": False},
    )

    # Test partial matches
    result = rule.validate("badly")
    assert not result.passed

    result = rule.validate("embadden")
    assert not result.passed


def test_edge_cases(rule):
    """Test handling of edge cases."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
    }

    for text in edge_cases.values():
        result = rule.validate(text)
        assert isinstance(result, RuleResult)
        assert result.passed  # None of these should contain prohibited terms


def test_error_handling(rule):
    """Test error handling for invalid inputs."""
    # Test None input
    with pytest.raises(ValueError, match="Output cannot be None"):
        rule.validate(None)

    # Test non-string input
    with pytest.raises(ValueError, match="Output must be a string"):
        rule.validate(123)


def test_metadata(rule):
    """Test metadata in validation results."""
    # Test clean content
    result = rule.validate("This is good.")
    assert "found_terms" in result.metadata
    assert isinstance(result.metadata["found_terms"], list)
    assert len(result.metadata["found_terms"]) == 0

    # Test content with prohibited terms
    result = rule.validate("This is bad and worse.")
    assert "found_terms" in result.metadata
    assert isinstance(result.metadata["found_terms"], list)
    assert len(result.metadata["found_terms"]) == 2
    assert "bad" in result.metadata["found_terms"]
    assert "worse" in result.metadata["found_terms"]


def test_multiple_occurrences():
    """Test handling of multiple occurrences of the same term."""
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["bad"], "case_sensitive": False},
    )

    result = rule.validate("This is bad, really bad, very bad.")
    assert not result.passed
    assert len(result.metadata["found_terms"]) == 1  # Should only report unique terms
