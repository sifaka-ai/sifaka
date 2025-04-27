"""Tests for the factual rules."""

import pytest
from typing import Dict, Any, List, Set

from sifaka.rules.factual import (
    FactualConsistencyRule,
    ConfidenceRule,
    CitationRule,
    FactualAccuracyRule,
)
from sifaka.rules.base import RuleResult


class TestFactualConsistencyRule(FactualConsistencyRule):
    """Test implementation of FactualConsistencyRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        contradictions = []

        for indicator in self.contradiction_indicators:
            if indicator in output_lower:
                contradictions.append(indicator)

        if contradictions:
            return RuleResult(
                passed=False,
                message="Output contains potential contradictions",
                metadata={"contradiction_indicators": contradictions},
            )

        return RuleResult(passed=True, message="No contradictions detected")


class TestConfidenceRule(ConfidenceRule):
    """Test implementation of ConfidenceRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        confidence_levels = {}

        for level, indicators in self.confidence_indicators.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            if found_indicators:
                confidence_levels[level] = found_indicators

        if confidence_levels:
            return RuleResult(
                passed=True,
                message="Confidence levels detected",
                metadata={"confidence_levels": confidence_levels},
            )

        return RuleResult(passed=True, message="No confidence indicators detected")


class TestCitationRule(CitationRule):
    """Test implementation of CitationRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, output)
            citations.extend(matches)

        if self.required_citations and not citations:
            return RuleResult(
                passed=False,
                message="No citations found in the output",
                metadata={"citation_patterns": self.citation_patterns},
            )

        return RuleResult(
            passed=True,
            message=f"Found {len(citations)} citations",
            metadata={"citations": citations},
        )


class TestFactualAccuracyRule(FactualAccuracyRule):
    """Test implementation of FactualAccuracyRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        found_facts = {}

        for topic, variations in self.knowledge_base.items():
            found_variations = []
            for variation in variations:
                if variation.lower() in output_lower:
                    found_variations.append(variation)
            if found_variations:
                found_facts[topic] = found_variations

        if not found_facts:
            return RuleResult(
                passed=True,
                message="No factual claims detected",
                metadata={"checked_topics": list(self.knowledge_base.keys())},
            )

        return RuleResult(
            passed=True,
            message=f"Found factual claims about {len(found_facts)} topics",
            metadata={"found_facts": found_facts},
        )


@pytest.fixture
def consistency_rule():
    """Create a TestFactualConsistencyRule instance."""
    return TestFactualConsistencyRule(name="test_consistency", description="Test consistency rule")


@pytest.fixture
def confidence_rule():
    """Create a TestConfidenceRule instance."""
    return TestConfidenceRule(name="test_confidence", description="Test confidence rule")


@pytest.fixture
def citation_rule():
    """Create a TestCitationRule instance."""
    return TestCitationRule(
        name="test_citation", description="Test citation rule", required_citations=True
    )


@pytest.fixture
def accuracy_rule():
    """Create a TestFactualAccuracyRule instance."""
    return TestFactualAccuracyRule(name="test_accuracy", description="Test accuracy rule")


def test_consistency_rule_initialization():
    """Test FactualConsistencyRule initialization."""
    custom_indicators = ["but", "however"]
    rule = TestFactualConsistencyRule(
        name="test",
        description="test",
        contradiction_indicators=custom_indicators,
        confidence_threshold=0.8,
    )
    assert rule.name == "test"
    assert rule.contradiction_indicators == custom_indicators
    assert rule.confidence_threshold == 0.8


def test_consistency_rule_validation(consistency_rule):
    """Test consistency rule validation."""
    # Test text without contradictions
    valid_text = "The sky is blue. The grass is green."
    result = consistency_rule.validate(valid_text)
    assert result.passed
    assert "No contradictions" in result.message

    # Test text with contradictions
    invalid_text = "The sky is blue. However, it appears to be red."
    result = consistency_rule.validate(invalid_text)
    assert not result.passed
    assert "however" in result.metadata["contradiction_indicators"]

    # Test multiple contradictions
    multiple_contradictions = (
        "Although X is true, Y is false. However, Z is true. Nevertheless, A is false."
    )
    result = consistency_rule.validate(multiple_contradictions)
    assert not result.passed
    assert len(result.metadata["contradiction_indicators"]) > 1


def test_confidence_rule_initialization():
    """Test ConfidenceRule initialization."""
    custom_indicators = {"high": ["definitely", "certainly"], "low": ["maybe", "possibly"]}
    rule = TestConfidenceRule(
        name="test", description="test", confidence_indicators=custom_indicators
    )
    assert rule.name == "test"
    assert rule.confidence_indicators == custom_indicators


def test_confidence_rule_validation(confidence_rule):
    """Test confidence rule validation."""
    # Test high confidence
    high_confidence = "This will definitely happen. We are certainly sure."
    result = confidence_rule.validate(high_confidence)
    assert result.passed
    assert "high" in result.metadata["confidence_levels"]

    # Test low confidence
    low_confidence = "This might happen. It's possibly true."
    result = confidence_rule.validate(low_confidence)
    assert result.passed
    assert "low" in result.metadata["confidence_levels"]

    # Test mixed confidence
    mixed_confidence = "This will definitely happen, but maybe not today."
    result = confidence_rule.validate(mixed_confidence)
    assert result.passed
    assert "high" in result.metadata["confidence_levels"]
    assert "low" in result.metadata["confidence_levels"]


def test_citation_rule_initialization():
    """Test CitationRule initialization."""
    custom_patterns = [r"\[\d+\]", r"\(Author, \d{4}\)"]
    rule = TestCitationRule(
        name="test", description="test", citation_patterns=custom_patterns, required_citations=True
    )
    assert rule.name == "test"
    assert rule.citation_patterns == custom_patterns
    assert rule.required_citations is True


def test_citation_rule_validation(citation_rule):
    """Test citation rule validation."""
    # Test text with citations
    valid_text = "According to [1], this is true. (Smith et al., 2020) confirms it."
    result = citation_rule.validate(valid_text)
    assert result.passed
    assert len(result.metadata["citations"]) == 2

    # Test text without citations
    invalid_text = "This text has no citations."
    result = citation_rule.validate(invalid_text)
    assert not result.passed
    assert "No citations found" in result.message

    # Test different citation formats
    mixed_citations = """
    [1] Numeric citation
    (Smith, 2020) Author-year citation
    https://example.com URL citation
    """
    result = citation_rule.validate(mixed_citations)
    assert result.passed
    assert len(result.metadata["citations"]) == 3


def test_accuracy_rule_initialization():
    """Test FactualAccuracyRule initialization."""
    custom_knowledge_base = {
        "test_fact": {"value1", "value2"},
        "another_fact": {"value3", "value4"},
    }
    rule = TestFactualAccuracyRule(
        name="test", description="test", knowledge_base=custom_knowledge_base
    )
    assert rule.name == "test"
    assert rule.knowledge_base == custom_knowledge_base


def test_accuracy_rule_validation(accuracy_rule):
    """Test accuracy rule validation."""
    # Test text with known facts
    valid_text = "The Earth is round and pi is 3.14159."
    result = accuracy_rule.validate(valid_text)
    assert result.passed
    assert "earth_shape" in result.metadata["found_facts"]
    assert "pi" in result.metadata["found_facts"]

    # Test text with no known facts
    no_facts = "This text contains no known facts from our knowledge base."
    result = accuracy_rule.validate(no_facts)
    assert result.passed
    assert "checked_topics" in result.metadata

    # Test multiple fact variations
    multiple_facts = "Water boils at 100°C (212°F). The Earth is spherical and shaped like a geoid."
    result = accuracy_rule.validate(multiple_facts)
    assert result.passed
    assert "water_boiling_point" in result.metadata["found_facts"]
    assert "earth_shape" in result.metadata["found_facts"]


def test_edge_cases():
    """Test edge cases for all rules."""
    rules = [
        TestFactualConsistencyRule(name="consistency", description="test"),
        TestConfidenceRule(name="confidence", description="test"),
        TestCitationRule(name="citation", description="test"),
        TestFactualAccuracyRule(name="accuracy", description="test"),
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
        TestFactualConsistencyRule(name="consistency", description="test"),
        TestConfidenceRule(name="confidence", description="test"),
        TestCitationRule(name="citation", description="test"),
        TestFactualAccuracyRule(name="accuracy", description="test"),
    ]

    invalid_inputs = [None, 123, [], {}]

    for rule in rules:
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                rule.validate(invalid_input)


def test_consistent_results():
    """Test consistency of validation results."""
    rules = [
        TestFactualConsistencyRule(name="consistency", description="test"),
        TestConfidenceRule(name="confidence", description="test"),
        TestCitationRule(name="citation", description="test"),
        TestFactualAccuracyRule(name="accuracy", description="test"),
    ]

    test_text = """
    According to [1], the Earth is definitely round.
    However, some people disagree.
    (Smith et al., 2020) confirms that water boils at 100°C.
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
