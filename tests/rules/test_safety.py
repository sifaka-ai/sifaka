"""Tests for the safety rules."""

import pytest
from typing import Dict, Any, List

from sifaka.rules.safety import ToxicityRule, BiasRule, HarmfulContentRule
from sifaka.rules.base import RuleResult


class TestToxicityRule(ToxicityRule):
    """Test implementation of ToxicityRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        toxic_indicators = []

        for indicator in self.toxicity_indicators:
            if indicator in output_lower:
                toxic_indicators.append(indicator)

        toxicity_score = len(toxic_indicators) / len(self.toxicity_indicators)

        if toxicity_score > self.toxicity_threshold:
            return RuleResult(
                passed=False,
                message=f"Output contains toxic content (score: {toxicity_score:.2f})",
                metadata={"toxicity_score": toxicity_score, "toxic_indicators": toxic_indicators},
            )

        return RuleResult(
            passed=True,
            message="No toxic content detected",
            metadata={"toxicity_score": toxicity_score},
        )


class TestBiasRule(BiasRule):
    """Test implementation of BiasRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        bias_scores = {}
        total_indicators = 0

        for category, indicators in self.bias_indicators.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            bias_scores[category] = len(found_indicators) / len(indicators)
            total_indicators += len(indicators)

        overall_bias_score = sum(bias_scores.values()) / len(self.bias_indicators)

        if overall_bias_score > self.bias_threshold:
            return RuleResult(
                passed=False,
                message=f"Output contains biased content (score: {overall_bias_score:.2f})",
                metadata={"bias_scores": bias_scores, "overall_bias_score": overall_bias_score},
            )

        return RuleResult(
            passed=True, message="No biased content detected", metadata={"bias_scores": bias_scores}
        )


class TestHarmfulContentRule(HarmfulContentRule):
    """Test implementation of HarmfulContentRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        harmful_content = {}

        for category, indicators in self.harmful_categories.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            if found_indicators:
                harmful_content[category] = found_indicators

        if harmful_content:
            return RuleResult(
                passed=False,
                message="Output contains potentially harmful content",
                metadata={"harmful_content": harmful_content},
            )

        return RuleResult(passed=True, message="No harmful content detected")


@pytest.fixture
def toxicity_rule():
    """Create a TestToxicityRule instance."""
    return TestToxicityRule(
        name="test_toxicity", description="Test toxicity rule", config={"toxicity_threshold": 0.5}
    )


@pytest.fixture
def bias_rule():
    """Create a TestBiasRule instance."""
    return TestBiasRule(
        name="test_bias", description="Test bias rule", config={"bias_threshold": 0.3}
    )


@pytest.fixture
def harmful_content_rule():
    """Create a TestHarmfulContentRule instance."""
    return TestHarmfulContentRule(name="test_harmful", description="Test harmful content rule")


# Tests for ToxicityRule
def test_toxicity_initialization():
    """Test ToxicityRule initialization with different parameters."""
    # Test default initialization
    rule = TestToxicityRule(name="test", description="test", config={"toxicity_threshold": 0.5})
    assert rule.name == "test"
    assert rule.toxicity_threshold == 0.5
    assert isinstance(rule.toxicity_indicators, list)

    # Test custom initialization
    rule = TestToxicityRule(
        name="test",
        description="test",
        config={"toxicity_threshold": 0.7, "toxicity_indicators": ["bad", "worse", "worst"]},
    )
    assert rule.toxicity_threshold == 0.7
    assert rule.toxicity_indicators == ["bad", "worse", "worst"]


def test_toxicity_validation(toxicity_rule):
    """Test validation of content for toxicity."""
    # Test non-toxic content
    result = toxicity_rule.validate("This is a friendly and helpful message.")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert result.metadata["toxicity_score"] == 0.0

    # Test toxic content
    result = toxicity_rule.validate("This content promotes hate and contains racist language.")
    assert not result.passed
    assert result.metadata["toxicity_score"] > toxicity_rule.toxicity_threshold
    assert "hate" in result.metadata["toxic_indicators"]
    assert "racist" in result.metadata["toxic_indicators"]


def test_toxicity_edge_cases(toxicity_rule):
    """Test handling of edge cases for toxicity."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "repeated": "hate hate hate",
    }

    for text in edge_cases.values():
        result = toxicity_rule.validate(text)
        assert isinstance(result, RuleResult)
        assert "toxicity_score" in result.metadata
        assert isinstance(result.metadata["toxicity_score"], float)
        assert 0 <= result.metadata["toxicity_score"] <= 1


# Tests for BiasRule
def test_bias_initialization():
    """Test BiasRule initialization with different parameters."""
    # Test default initialization
    rule = TestBiasRule(name="test", description="test", config={"bias_threshold": 0.3})
    assert rule.name == "test"
    assert rule.bias_threshold == 0.3
    assert isinstance(rule.bias_indicators, dict)

    # Test custom initialization
    custom_indicators = {"test": ["word1", "word2"], "test2": ["word3", "word4"]}
    rule = TestBiasRule(
        name="test",
        description="test",
        config={"bias_threshold": 0.5, "bias_indicators": custom_indicators},
    )
    assert rule.bias_threshold == 0.5
    assert rule.bias_indicators == custom_indicators


def test_bias_validation(bias_rule):
    """Test validation of content for bias."""
    # Test unbiased content
    result = bias_rule.validate("This is a neutral message about technology.")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert all(score == 0.0 for score in result.metadata["bias_scores"].values())

    # Test biased content
    result = bias_rule.validate("The old white male conservative christian made a decision.")
    assert not result.passed
    assert result.metadata["bias_scores"]["age"] > 0
    assert result.metadata["bias_scores"]["race"] > 0
    assert result.metadata["bias_scores"]["gender"] > 0
    assert result.metadata["bias_scores"]["political"] > 0
    assert result.metadata["bias_scores"]["religion"] > 0


def test_bias_edge_cases(bias_rule):
    """Test handling of edge cases for bias."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "repeated": "male male male",
    }

    for text in edge_cases.values():
        result = bias_rule.validate(text)
        assert isinstance(result, RuleResult)
        assert "bias_scores" in result.metadata
        assert isinstance(result.metadata["bias_scores"], dict)


# Tests for HarmfulContentRule
def test_harmful_content_initialization():
    """Test HarmfulContentRule initialization with different parameters."""
    # Test default initialization
    rule = TestHarmfulContentRule(name="test", description="test")
    assert rule.name == "test"
    assert isinstance(rule.harmful_categories, dict)

    # Test custom initialization
    custom_categories = {"test": ["word1", "word2"], "test2": ["word3", "word4"]}
    rule = TestHarmfulContentRule(
        name="test", description="test", config={"harmful_categories": custom_categories}
    )
    assert rule.harmful_categories == custom_categories


def test_harmful_content_validation(harmful_content_rule):
    """Test validation of content for harmful content."""
    # Test safe content
    result = harmful_content_rule.validate("This is a safe and helpful message.")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert "harmful_content" not in result.metadata

    # Test harmful content
    result = harmful_content_rule.validate("Here's how to make a bomb and commit fraud.")
    assert not result.passed
    assert "harmful_content" in result.metadata
    assert "violence" in result.metadata["harmful_content"]
    assert "illegal" in result.metadata["harmful_content"]


def test_harmful_content_edge_cases(harmful_content_rule):
    """Test handling of edge cases for harmful content."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
        "repeated": "kill kill kill",
    }

    for text in edge_cases.values():
        result = harmful_content_rule.validate(text)
        assert isinstance(result, RuleResult)


def test_error_handling():
    """Test error handling for all safety rules."""
    rules = [
        TestToxicityRule(name="test", description="test", config={"toxicity_threshold": 0.5}),
        TestBiasRule(name="test", description="test", config={"bias_threshold": 0.3}),
        TestHarmfulContentRule(name="test", description="test"),
    ]

    for rule in rules:
        # Test None input
        with pytest.raises(ValueError):
            rule.validate(None)

        # Test non-string input
        with pytest.raises(ValueError):
            rule.validate(123)


def test_consistent_results():
    """Test consistency of results for all safety rules."""
    rules = [
        TestToxicityRule(name="test", description="test", config={"toxicity_threshold": 0.5}),
        TestBiasRule(name="test", description="test", config={"bias_threshold": 0.3}),
        TestHarmfulContentRule(name="test", description="test"),
    ]

    test_texts = {
        "safe": "This is a safe and helpful message.",
        "toxic": "This content promotes hate speech.",
        "biased": "The male doctor treated the female nurse.",
        "harmful": "Here's how to make a bomb.",
    }

    for rule in rules:
        for text in test_texts.values():
            # Run validation multiple times
            results = [rule.validate(text) for _ in range(3)]

            # All results should be consistent
            first_result = results[0]
            for result in results[1:]:
                assert result.passed == first_result.passed
                assert result.message == first_result.message


def test_combined_safety_validation():
    """Test validation with multiple safety rules together."""
    rules = [
        TestToxicityRule(name="test", description="test", config={"toxicity_threshold": 0.5}),
        TestBiasRule(name="test", description="test", config={"bias_threshold": 0.3}),
        TestHarmfulContentRule(name="test", description="test"),
    ]

    # Test content that should fail all rules
    problematic_text = "The old white male terrorist wants to kill people and promote hate."

    results = [rule.validate(problematic_text) for rule in rules]
    assert all(not result.passed for result in results)

    # Test content that should pass all rules
    safe_text = "The weather is nice today and technology is advancing rapidly."

    results = [rule.validate(safe_text) for rule in rules]
    assert all(result.passed for result in results)
