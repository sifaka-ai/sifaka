"""Tests for the legal rules."""

import pytest
import re
from typing import Dict, Any, List

from sifaka.rules.legal import LegalCitationRule
from sifaka.rules.base import RuleResult


class TestLegalCitationRule(LegalCitationRule):
    """Test implementation of LegalCitationRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        # Find all citations in the output
        found_citations = []
        for pattern in self.compiled_patterns:
            found_citations.extend(pattern.findall(output))

        if not found_citations:
            return RuleResult(
                passed=True,
                message="No legal citations found in the output.",
                metadata={"valid_citations": [], "invalid_citations": []},
            )

        # For this test implementation, we'll validate basic citation formats
        invalid_citations = []
        for citation in found_citations:
            # Basic format validation
            if not any(pattern.match(citation) for pattern in self.compiled_patterns):
                invalid_citations.append(citation)

        if invalid_citations:
            return RuleResult(
                passed=False,
                message=f"Found {len(invalid_citations)} invalid legal citations",
                metadata={
                    "invalid_citations": invalid_citations,
                    "valid_citations": [c for c in found_citations if c not in invalid_citations],
                },
            )

        return RuleResult(
            passed=True,
            message=f"All {len(found_citations)} legal citations are valid",
            metadata={"valid_citations": found_citations, "invalid_citations": []},
        )


@pytest.fixture
def rule():
    """Create a TestLegalCitationRule instance."""
    return TestLegalCitationRule(
        name="test_legal_citation",
        description="Test legal citation rule",
        config={
            "citation_patterns": [
                r"\d+\s+U\.S\.\s+\d+",  # US Reports citations
                r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
                r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter citations
                r"\d+\s+F\.\s*Supp\.\s+\d+",  # Federal Supplement
                r"\d+\s+[A-Za-z]+(?:\.[A-Za-z]+)*\.[23]d\s+\d+",  # State Reporter citations
                r"\d+\s+U\.S\.C\.\s+§\s+\d+",  # Federal statute citations
            ],
        },
    )


def test_initialization():
    """Test rule initialization with different parameters."""
    # Test default initialization
    rule = TestLegalCitationRule(name="test", description="test")
    assert rule.name == "test"
    assert isinstance(rule.citation_patterns, list)
    assert isinstance(rule.compiled_patterns[0], re.Pattern)

    # Test custom initialization
    custom_patterns = [r"\d+ Custom \d+", r"[A-Z]+ v\. [A-Z]+"]
    rule = TestLegalCitationRule(
        name="test", description="test", config={"citation_patterns": custom_patterns}
    )
    assert rule.citation_patterns == custom_patterns
    assert all(isinstance(pattern, re.Pattern) for pattern in rule.compiled_patterns)


def test_supreme_court_citations(rule):
    """Test validation of Supreme Court citations."""
    # Test valid Supreme Court citations
    valid_citations = [
        "410 U.S. 113",  # Roe v. Wade
        "347 U.S. 483",  # Brown v. Board of Education
        "576 U.S. 644",  # Obergefell v. Hodges
        "138 S.Ct. 2448",  # Trump v. Hawaii
    ]

    for citation in valid_citations:
        result = rule.validate(f"The case {citation} established...")
        assert result.passed
        assert citation in result.metadata["valid_citations"]

    # Test multiple citations in one text
    text = "The cases 410 U.S. 113 and 347 U.S. 483 are landmark decisions."
    result = rule.validate(text)
    assert result.passed
    assert len(result.metadata["valid_citations"]) == 2


def test_federal_reporter_citations(rule):
    """Test validation of Federal Reporter citations."""
    # Test valid Federal Reporter citations
    valid_citations = ["789 F.2d 123", "456 F.3d 789", "123 F.2d 456"]

    for citation in valid_citations:
        result = rule.validate(f"In {citation}, the court held...")
        assert result.passed
        assert citation in result.metadata["valid_citations"]


def test_state_case_citations(rule):
    """Test validation of state case citations."""
    # Test valid state case citations
    valid_citations = ["123 Cal.2d 456", "789 N.Y.2d 123", "456 Tex.3d 789"]

    for citation in valid_citations:
        result = rule.validate(f"According to {citation}...")
        assert result.passed
        assert citation in result.metadata["valid_citations"]


def test_statute_citations(rule):
    """Test validation of statute citations."""
    # Test valid statute citations
    valid_citations = ["42 U.S.C. § 1983", "18 U.S.C. § 242", "28 U.S.C. § 1332"]

    for citation in valid_citations:
        result = rule.validate(f"Under {citation}...")
        assert result.passed
        assert citation in result.metadata["valid_citations"]


def test_invalid_citations(rule):
    """Test validation of invalid citations."""
    invalid_citations = [
        "123 Invalid 456",  # Invalid format
        "ABC U.S. XYZ",  # Non-numeric components
        "123 U.S.",  # Incomplete citation
        "F.2d 123",  # Missing volume number
        "123 456",  # Missing reporter
    ]

    for citation in invalid_citations:
        result = rule.validate(f"According to {citation}...")
        assert result.passed  # Should pass because invalid citations aren't matched
        assert not result.metadata.get("valid_citations", [])  # No citations should be found


def test_mixed_citations(rule):
    """Test validation of text with both valid and invalid citations."""
    text = """
    The Supreme Court in 410 U.S. 113 established the right to privacy.
    Some people incorrectly cite it as 410 US 113 or 410 U.S 113.
    Later, in 347 U.S. 483, the Court addressed segregation.
    """

    result = rule.validate(text)
    assert result.passed
    valid_citations = result.metadata["valid_citations"]
    assert len(valid_citations) == 2
    assert "410 U.S. 113" in valid_citations
    assert "347 U.S. 483" in valid_citations


def test_edge_cases(rule):
    """Test handling of edge cases."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "numbers_only": "123 456 789",
        "partial_citation": "U.S. § Code",
    }

    for text in edge_cases.values():
        result = rule.validate(text)
        assert isinstance(result, RuleResult)
        assert result.passed  # Should pass when no citations are found
        assert "valid_citations" in result.metadata
        assert not result.metadata["valid_citations"]  # Should be empty list


def test_error_handling(rule):
    """Test error handling for invalid inputs."""
    # Test None input
    with pytest.raises(ValueError):
        rule.validate(None)

    # Test non-string input
    with pytest.raises(ValueError):
        rule.validate(123)


def test_metadata(rule):
    """Test metadata in validation results."""
    # Test with valid citation
    result = rule.validate("According to 410 U.S. 113...")
    assert "valid_citations" in result.metadata
    assert isinstance(result.metadata["valid_citations"], list)
    assert "410 U.S. 113" in result.metadata["valid_citations"]

    # Test with no citations
    result = rule.validate("This text has no citations.")
    assert "valid_citations" in result.metadata
    assert not result.metadata["valid_citations"]  # Should be empty list


def test_consistent_results(rule):
    """Test consistency of citation validation."""
    test_texts = {
        "no_citation": "This text has no legal citations.",
        "supreme_court": "According to 410 U.S. 113...",
        "federal_reporter": "In 789 F.2d 123, the court held...",
        "state_case": "As stated in 123 Cal.2d 456...",
        "statute": "Under 42 U.S.C. § 1983...",
    }

    for text in test_texts.values():
        # Run validation multiple times
        results = [rule.validate(text) for _ in range(3)]

        # All results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result.passed == first_result.passed
            assert result.message == first_result.message
            assert result.metadata == first_result.metadata


def test_citation_patterns():
    """Test that all default citation patterns are working."""
    rule = TestLegalCitationRule(
        name="test",
        description="test",
        config={
            "citation_patterns": [
                r"\d+\s+U\.S\.\s+\d+",  # US Reports citations
                r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
                r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter citations
                r"\d+\s+F\.\s*Supp\.\s+\d+",  # Federal Supplement
                r"\d+\s+[A-Za-z]+(?:\.[A-Za-z]+)*\.[23]d\s+\d+",  # State Reporter citations
                r"\d+\s+U\.S\.C\.\s+§\s+\d+",  # Federal statute citations
            ],
        },
    )

    # Test each citation pattern
    test_cases = [
        ("Supreme Court", "410 U.S. 113"),
        ("Supreme Court Reporter", "138 S.Ct. 2448"),
        ("Federal Reporter", "789 F.2d 123"),
        ("State Reporter", "123 Cal.2d 456"),
        ("Federal Statute", "42 U.S.C. § 1983"),
    ]

    for case_type, citation in test_cases:
        result = rule.validate(citation)
        assert result.passed, f"Failed to validate {case_type} citation: {citation}"
        assert citation in result.metadata["valid_citations"]
