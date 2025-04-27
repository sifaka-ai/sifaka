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

        return RuleResult(
            passed=len(found_terms) == 0,
            message=(
                "No prohibited terms found"
                if not found_terms
                else f"Found prohibited terms: {', '.join(found_terms)}"
            ),
            metadata={
                "found_terms": found_terms,
                "case_sensitive": self.case_sensitive,
            },
        )


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

        # Calculate consistency score
        total_positive = len(self.tone_indicators[self.expected_tone.lower()]["positive"])
        total_negative = len(self.tone_indicators[self.expected_tone.lower()]["negative"])

        positive_ratio = len(positive_indicators) / total_positive if total_positive > 0 else 0
        negative_ratio = len(negative_indicators) / total_negative if total_negative > 0 else 0
        consistency_score = positive_ratio - negative_ratio

        passed = len(negative_indicators) <= len(positive_indicators)

        return RuleResult(
            passed=passed,
            message=f"Output {'maintains' if passed else 'does not maintain'} {self.expected_tone} tone",
            metadata={
                "positive_indicators": positive_indicators,
                "negative_indicators": negative_indicators,
                "consistency_score": consistency_score,
            },
        )


@pytest.fixture
def prohibited_rule():
    """Create a ProhibitedContentRule instance."""
    return TestProhibitedContentRule(
        name="test_prohibited",
        description="Test prohibited content rule",
        config={"prohibited_terms": ["bad", "inappropriate"]},
    )


@pytest.fixture
def tone_rule():
    """Create a ToneConsistencyRule instance."""
    indicators = {
        "formal": {
            "positive": [
                "therefore",
                "furthermore",
                "consequently",
                "analysis",
                "proceed",
            ],
            "negative": [
                "hey",
                "cool",
                "awesome",
                "wow",
                "yeah",
            ],
        }
    }
    return TestToneConsistencyRule(
        name="test_tone",
        description="Test tone consistency rule",
        config={
            "tone_indicators": indicators,
            "expected_tone": "formal",
            "threshold": 0.7,
        },
    )


def test_prohibited_content_rule_initialization():
    """Test ProhibitedContentRule initialization."""
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["bad", "inappropriate"]},
    )
    assert rule.name == "test"
    assert rule.prohibited_terms == ["bad", "inappropriate"]


def test_prohibited_content_validation(prohibited_rule):
    """Test prohibited content rule validation."""
    # Test clean text
    clean_text = "This is a good and appropriate text."
    result = prohibited_rule.validate(clean_text)
    assert result.passed
    assert "found_terms" in result.metadata
    assert len(result.metadata["found_terms"]) == 0

    # Test text with prohibited terms
    bad_text = "This is a bad and inappropriate text."
    result = prohibited_rule.validate(bad_text)
    assert not result.passed
    assert "found_terms" in result.metadata
    assert len(result.metadata["found_terms"]) == 2
    assert "bad" in result.metadata["found_terms"]
    assert "inappropriate" in result.metadata["found_terms"]

    # Test case sensitivity
    mixed_case_text = "This is BAD and InAppropriate text."
    result = prohibited_rule.validate(mixed_case_text)
    assert not result.passed
    assert len(result.metadata["found_terms"]) == 2


def test_tone_rule_initialization():
    """Test ToneConsistencyRule initialization."""
    custom_indicators = {"test_tone": {"positive": ["good", "great"], "negative": ["bad", "poor"]}}
    rule = TestToneConsistencyRule(
        name="test",
        description="test",
        config={
            "tone_indicators": custom_indicators,
            "expected_tone": "test_tone",
            "threshold": 0.7,
        },
    )
    assert rule.name == "test"
    assert rule.expected_tone == "test_tone"
    assert rule.tone_indicators == custom_indicators
    assert rule.threshold == 0.7


def test_tone_rule_validation(tone_rule):
    """Test tone consistency rule validation."""
    # Test formal tone
    formal_text = "Therefore, we must proceed. Furthermore, the analysis shows."
    result = tone_rule.validate(formal_text)
    assert result.passed
    assert "consistency_score" in result.metadata

    # Test informal tone
    informal_text = "Hey! What's up? This is cool!"
    result = tone_rule.validate(informal_text)
    assert not result.passed
    assert "consistency_score" in result.metadata


def test_edge_cases():
    """Test edge cases for all rules."""
    rules = [
        TestProhibitedContentRule(
            name="prohibited",
            description="test",
            config={"prohibited_terms": ["bad"]},
        ),
        TestToneConsistencyRule(
            name="tone",
            description="test",
            config={
                "tone_indicators": {"formal": {"positive": ["therefore"], "negative": ["hey"]}},
                "expected_tone": "formal",
            },
        ),
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


def test_error_handling():
    """Test error handling for all rules."""
    rules = [
        TestProhibitedContentRule(
            name="prohibited",
            description="test",
            config={"prohibited_terms": ["bad"]},
        ),
        TestToneConsistencyRule(
            name="tone",
            description="test",
            config={
                "tone_indicators": {"formal": {"positive": ["therefore"], "negative": ["hey"]}},
                "expected_tone": "formal",
            },
        ),
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
            name="prohibited",
            description="test",
            config={"prohibited_terms": ["bad", "inappropriate"]},
        ),
        TestToneConsistencyRule(
            name="tone",
            description="test",
            config={
                "tone_indicators": {"formal": {"positive": ["therefore"], "negative": ["hey"]}},
                "expected_tone": "formal",
            },
        ),
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


def test_prohibited_content_edge_cases(prohibited_rule):
    """Test edge cases for prohibited content rule."""
    # Test empty string
    result = prohibited_rule.validate("")
    assert result.passed
    assert len(result.metadata["found_terms"]) == 0

    # Test whitespace only
    result = prohibited_rule.validate("   \n\t   ")
    assert result.passed
    assert len(result.metadata["found_terms"]) == 0

    # Test special characters
    result = prohibited_rule.validate("!@#$%^&*()")
    assert result.passed
    assert len(result.metadata["found_terms"]) == 0

    # Test numbers
    result = prohibited_rule.validate("12345")
    assert result.passed
    assert len(result.metadata["found_terms"]) == 0


def test_prohibited_content_error_handling():
    """Test error handling for prohibited content rule."""
    # Test None input
    rule = TestProhibitedContentRule(
        name="test",
        description="test",
        config={"prohibited_terms": ["bad", "inappropriate"]},
    )
    with pytest.raises(ValueError, match="Output cannot be None"):
        rule.validate(None)

    # Test invalid config
    with pytest.raises(ValueError):
        TestProhibitedContentRule(
            name="test",
            description="test",
            config={"prohibited_terms": None},
        )

    with pytest.raises(ValueError):
        TestProhibitedContentRule(
            name="test",
            description="test",
            config={"prohibited_terms": []},
        )


def test_prohibited_content_consistent_results(prohibited_rule):
    """Test consistent validation results."""
    text = "This is a bad and inappropriate text."

    # Run validation multiple times
    results = [prohibited_rule.validate(text) for _ in range(5)]

    # Check all results are the same
    first_result = results[0]
    for result in results[1:]:
        assert result.passed == first_result.passed
        assert result.metadata["found_terms"] == first_result.metadata["found_terms"]
