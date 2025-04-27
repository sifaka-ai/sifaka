"""Tests for the safety rules."""

import pytest
from typing import Dict, Any, List

from sifaka.rules import ToxicityRule, BiasRule, HarmfulContentRule
from sifaka.rules.base import RuleResult


class TestToxicityRule(ToxicityRule):
    """Test implementation of ToxicityRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        toxic_indicators = []

        for indicator in self.indicators:
            if indicator in output_lower:
                toxic_indicators.append(indicator)

        toxicity_score = len(toxic_indicators) / len(self.indicators) if self.indicators else 0.0

        if toxicity_score > self.threshold:
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

        for category, indicators in self.categories.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            bias_scores[category] = len(found_indicators) / len(indicators)
            total_indicators += len(indicators)

        overall_bias_score = sum(bias_scores.values()) / len(self.categories)

        if overall_bias_score > self.threshold:
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

        for category, indicators in self.categories.items():
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

        return RuleResult(
            passed=True,
            message="No harmful content detected",
            metadata={"harmful_content": {}},  # Always include harmful_content in metadata
        )


@pytest.fixture
def toxicity_rule():
    """Create a TestToxicityRule instance."""
    return TestToxicityRule(
        name="toxicity_test",
        description="Test toxicity rule",
        config={
            "threshold": 0.5,
            "indicators": ["toxic", "hate", "offensive"],
        },
    )


@pytest.fixture
def bias_rule():
    """Create a TestBiasRule instance."""
    return TestBiasRule(
        name="bias_test",
        description="Test bias rule",
        config={
            "threshold": 0.5,
            "categories": {
                "gender": ["male", "female"],
                "race": ["white", "black"],
            },
        },
    )


@pytest.fixture
def harmful_content_rule():
    """Create a TestHarmfulContentRule instance."""
    return TestHarmfulContentRule(
        name="harmful_test",
        description="Test harmful content rule",
        config={
            "categories": {
                "violence": ["kill", "hurt"],
                "self_harm": ["suicide", "self-harm"],
            },
        },
    )


def test_toxicity_rule_initialization(toxicity_rule):
    assert toxicity_rule.name == "toxicity_test"
    assert toxicity_rule.description == "Test toxicity rule"
    assert toxicity_rule.threshold == 0.5
    assert "toxic" in toxicity_rule.indicators


def test_toxicity_rule_validation(toxicity_rule):
    # Test non-toxic content
    result = toxicity_rule._validate_impl("This is a normal sentence.")
    assert result.passed
    assert result.metadata["toxicity_score"] == 0.0

    # Test toxic content
    result = toxicity_rule._validate_impl("This is toxic and offensive content.")
    assert not result.passed
    assert result.metadata["toxicity_score"] > 0.0
    assert len(result.metadata["toxic_indicators"]) > 0


def test_bias_rule_initialization(bias_rule):
    assert bias_rule.name == "bias_test"
    assert bias_rule.description == "Test bias rule"
    assert bias_rule.threshold == 0.5
    assert "gender" in bias_rule.categories
    assert "race" in bias_rule.categories


def test_bias_rule_validation(bias_rule):
    # Test unbiased content
    result = bias_rule._validate_impl("This is a neutral sentence.")
    assert result.passed
    assert all(score == 0.0 for score in result.metadata["bias_scores"].values())

    # Test biased content
    result = bias_rule._validate_impl("The male and female participants were white and black.")
    assert not result.passed
    assert any(score > 0.0 for score in result.metadata["bias_scores"].values())
    assert result.metadata["overall_bias_score"] > 0.0


def test_harmful_content_rule_initialization(harmful_content_rule):
    assert harmful_content_rule.name == "harmful_test"
    assert harmful_content_rule.description == "Test harmful content rule"
    assert "violence" in harmful_content_rule.categories
    assert "self_harm" in harmful_content_rule.categories


def test_harmful_content_rule_validation(harmful_content_rule):
    # Test safe content
    result = harmful_content_rule._validate_impl("This is a safe sentence.")
    assert result.passed
    assert "harmful_content" in result.metadata
    assert not result.metadata["harmful_content"]

    # Test harmful content
    result = harmful_content_rule._validate_impl("They want to kill and hurt others.")
    assert not result.passed
    assert "violence" in result.metadata["harmful_content"]
    assert len(result.metadata["harmful_content"]["violence"]) > 0


def test_none_input_handling(toxicity_rule, bias_rule, harmful_content_rule):
    with pytest.raises(ValueError):
        toxicity_rule.validate(None)
    with pytest.raises(ValueError):
        bias_rule.validate(None)
    with pytest.raises(ValueError):
        harmful_content_rule.validate(None)


def test_invalid_input_handling(toxicity_rule, bias_rule, harmful_content_rule):
    with pytest.raises(ValueError):
        toxicity_rule.validate(123)
    with pytest.raises(ValueError):
        bias_rule.validate(["not", "a", "string"])
    with pytest.raises(ValueError):
        harmful_content_rule.validate({"key": "value"})


def test_error_handling():
    """Test error handling for all safety rules."""
    rules = [
        TestToxicityRule(
            name="test", description="test", config={"threshold": 0.5, "indicators": ["toxic"]}
        ),
        TestBiasRule(
            name="test",
            description="test",
            config={"threshold": 0.3, "categories": {"test": ["test"]}},
        ),
        TestHarmfulContentRule(
            name="test", description="test", config={"categories": {"test": ["test"]}}
        ),
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
        TestToxicityRule(
            name="test", description="test", config={"threshold": 0.5, "indicators": ["hate"]}
        ),
        TestBiasRule(
            name="test",
            description="test",
            config={"threshold": 0.3, "categories": {"gender": ["male"]}},
        ),
        TestHarmfulContentRule(
            name="test", description="test", config={"categories": {"violence": ["kill"]}}
        ),
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
        TestToxicityRule(
            name="test",
            description="test",
            config={
                "threshold": 0.5,
                "indicators": ["hate", "terrorist", "kill"],
            },
        ),
        TestBiasRule(
            name="test",
            description="test",
            config={
                "threshold": 0.3,
                "categories": {
                    "gender": ["male", "female"],
                    "race": ["white", "black"],
                },
            },
        ),
        TestHarmfulContentRule(
            name="test",
            description="test",
            config={
                "categories": {
                    "violence": ["kill", "terrorist"],
                    "hate_speech": ["hate"],
                },
            },
        ),
    ]

    # Test content that should fail all rules
    problematic_text = "The old white male terrorist wants to kill people and promote hate."

    results = [rule.validate(problematic_text) for rule in rules]
    assert all(not result.passed for result in results)

    # Test content that should pass all rules
    safe_text = "The weather is nice today and technology is advancing rapidly."

    results = [rule.validate(safe_text) for rule in rules]
    assert all(result.passed for result in results)
