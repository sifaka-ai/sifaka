"""Tests for the content rules."""

import pytest
from typing import Dict, Any, List

from sifaka.rules.content import ProhibitedContentRule, ToneConsistencyRule
from sifaka.rules.base import RuleResult


class TestProhibitedContentRule(ProhibitedContentRule):
    """Test implementation of ProhibitedContentRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        check_output = output if self.case_sensitive else output.lower()
        found_terms = []

        for term in self.prohibited_terms:
            search_term = term if self.case_sensitive else term.lower()
            if search_term in check_output:
                found_terms.append(term)

        if found_terms:
            return RuleResult(
                passed=False,
                message=f"Output contains prohibited terms: {', '.join(found_terms)}",
                metadata={"found_terms": found_terms},
            )

        return RuleResult(passed=True, message="No prohibited terms found in the output")


class TestToneConsistencyRule(ToneConsistencyRule):
    """Test implementation of ToneConsistencyRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        if self.expected_tone.lower() not in self.tone_indicators:
            return RuleResult(
                passed=False,
                message=f"Unknown tone: {self.expected_tone}",
                metadata={"available_tones": list(self.tone_indicators.keys())},
            )

        output_lower = output.lower()

        # Check for positive indicators
        positive_indicators = []
        for term in self.tone_indicators[self.expected_tone.lower()]["positive"]:
            if term in output_lower:
                positive_indicators.append(term)

        # Check for negative indicators
        negative_indicators = []
        for term in self.tone_indicators[self.expected_tone.lower()]["negative"]:
            if term in output_lower:
                negative_indicators.append(term)

        # Simple scoring (in a real implementation, this would be more sophisticated)
        if len(negative_indicators) > len(positive_indicators):
            return RuleResult(
                passed=False,
                message=f"Output does not maintain {self.expected_tone} tone",
                metadata={
                    "positive_indicators": positive_indicators,
                    "negative_indicators": negative_indicators,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Output maintains {self.expected_tone} tone",
            metadata={
                "positive_indicators": positive_indicators,
                "negative_indicators": negative_indicators,
            },
        )


@pytest.fixture
def prohibited_rule():
    """Create a TestProhibitedContentRule instance."""
    return TestProhibitedContentRule(
        name="test_prohibited",
        description="Test prohibited content rule",
        prohibited_terms=["bad", "inappropriate", "offensive"],
        case_sensitive=False,
    )


@pytest.fixture
def tone_rule():
    """Create a TestToneConsistencyRule instance."""
    return TestToneConsistencyRule(
        name="test_tone", description="Test tone consistency rule", expected_tone="formal"
    )


def test_prohibited_rule_initialization():
    """Test ProhibitedContentRule initialization."""
    custom_terms = ["term1", "term2", "term3"]
    rule = TestProhibitedContentRule(
        name="test", description="test", prohibited_terms=custom_terms, case_sensitive=True
    )
    assert rule.name == "test"
    assert rule.prohibited_terms == custom_terms
    assert rule.case_sensitive is True


def test_prohibited_rule_validation(prohibited_rule):
    """Test prohibited content rule validation."""
    # Test text without prohibited terms
    safe_text = "This is a safe and appropriate text."
    result = prohibited_rule.validate(safe_text)
    assert result.passed
    assert "No prohibited terms" in result.message

    # Test text with prohibited terms
    unsafe_text = "This text contains bad and inappropriate content."
    result = prohibited_rule.validate(unsafe_text)
    assert not result.passed
    assert len(result.metadata["found_terms"]) == 2
    assert "bad" in result.metadata["found_terms"]
    assert "inappropriate" in result.metadata["found_terms"]

    # Test case sensitivity
    case_sensitive_rule = TestProhibitedContentRule(
        name="test",
        description="test",
        prohibited_terms=["Bad", "INAPPROPRIATE"],
        case_sensitive=True,
    )
    mixed_case_text = "This text contains bad and INAPPROPRIATE content."
    result = case_sensitive_rule.validate(mixed_case_text)
    assert result.passed  # Should pass because case doesn't match


def test_tone_rule_initialization():
    """Test ToneConsistencyRule initialization."""
    custom_indicators = {"test_tone": {"positive": ["good", "great"], "negative": ["bad", "poor"]}}
    rule = TestToneConsistencyRule(
        name="test",
        description="test",
        expected_tone="test_tone",
        tone_indicators=custom_indicators,
    )
    assert rule.name == "test"
    assert rule.expected_tone == "test_tone"
    assert rule.tone_indicators == custom_indicators


def test_tone_rule_validation(tone_rule):
    """Test tone consistency rule validation."""
    # Test formal tone
    formal_text = "Therefore, we must proceed. Furthermore, the analysis shows."
    result = tone_rule.validate(formal_text)
    assert result.passed
    assert len(result.metadata["positive_indicators"]) > 0
    assert len(result.metadata["negative_indicators"]) == 0

    # Test informal tone in formal context
    informal_text = "Yeah, that's cool! BTW, we're gonna do this awesome thing!"
    result = tone_rule.validate(informal_text)
    assert not result.passed
    assert len(result.metadata["negative_indicators"]) > len(result.metadata["positive_indicators"])

    # Test mixed tone
    mixed_text = "Therefore, we must proceed. BTW, it's gonna be great!"
    result = tone_rule.validate(mixed_text)
    assert "positive_indicators" in result.metadata
    assert "negative_indicators" in result.metadata

    # Test unknown tone
    unknown_tone_rule = TestToneConsistencyRule(
        name="test", description="test", expected_tone="unknown"
    )
    result = unknown_tone_rule.validate("Some text")
    assert not result.passed
    assert "Unknown tone" in result.message


def test_edge_cases():
    """Test edge cases for all rules."""
    rules = [
        TestProhibitedContentRule(name="prohibited", description="test", prohibited_terms=["bad"]),
        TestToneConsistencyRule(name="tone", description="test", expected_tone="formal"),
    ]

    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "numbers_only": "123 456 789",
    }

    for rule in rules:
        for case_name, text in edge_cases.items():
            result = rule.validate(text)
            assert isinstance(result, RuleResult)
            assert isinstance(result.passed, bool)
            assert isinstance(result.message, str)
            assert isinstance(result.metadata, dict)


def test_error_handling():
    """Test error handling for all rules."""
    rules = [
        TestProhibitedContentRule(name="prohibited", description="test", prohibited_terms=["bad"]),
        TestToneConsistencyRule(name="tone", description="test", expected_tone="formal"),
    ]

    invalid_inputs = [None, 123, [], {}]

    for rule in rules:
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                rule.validate(invalid_input)


def test_consistent_results():
    """Test consistency of validation results."""
    rules = [
        TestProhibitedContentRule(
            name="prohibited", description="test", prohibited_terms=["bad", "inappropriate"]
        ),
        TestToneConsistencyRule(name="tone", description="test", expected_tone="formal"),
    ]

    test_text = """
    Therefore, we must proceed with the analysis.
    Furthermore, this demonstrates good practice.
    """

    for rule in rules:
        # Run validation multiple times
        results = [rule.validate(test_text) for _ in range(3)]

        # All results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result.passed == first_result.passed
            assert result.message == first_result.message
            assert result.metadata == first_result.metadata
