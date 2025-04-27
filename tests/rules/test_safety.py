"""Tests for safety rules."""

import pytest
from typing import Dict, List

from sifaka.rules.safety import (
    ToxicityConfig,
    BiasConfig,
    HarmfulContentConfig,
    ToxicityRule,
    BiasRule,
    HarmfulContentRule,
    create_toxicity_rule,
    create_bias_rule,
    create_harmful_content_rule,
)
from sifaka.rules.base import RuleResult


@pytest.fixture
def toxicity_config() -> ToxicityConfig:
    """Create a test toxicity configuration."""
    return ToxicityConfig(
        threshold=0.5,
        indicators=["toxic", "hate", "offensive"],
        cache_size=100,
        priority=1,
        cost=1.0,
    )


@pytest.fixture
def bias_config() -> BiasConfig:
    """Create a test bias configuration."""
    return BiasConfig(
        threshold=0.5,
        categories={
            "gender": ["male", "female"],
            "race": ["white", "black", "asian"],
        },
        cache_size=100,
        priority=1,
        cost=1.0,
    )


@pytest.fixture
def harmful_content_config() -> HarmfulContentConfig:
    """Create a test harmful content configuration."""
    return HarmfulContentConfig(
        categories={
            "violence": ["kill", "hurt", "attack"],
            "self_harm": ["suicide", "self-harm"],
        },
        cache_size=100,
        priority=1,
        cost=1.0,
    )


def test_toxicity_config_validation():
    """Test toxicity configuration validation."""
    # Valid configuration
    config = ToxicityConfig(threshold=0.5, indicators=["toxic"])
    assert config.threshold == 0.5
    assert config.indicators == ["toxic"]

    # Invalid threshold
    with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
        ToxicityConfig(threshold=1.5, indicators=["toxic"])

    # Empty indicators
    with pytest.raises(ValueError, match="Must provide at least one toxicity indicator"):
        ToxicityConfig(threshold=0.5, indicators=[])

    # Invalid cache size
    with pytest.raises(ValueError, match="Cache size must be non-negative"):
        ToxicityConfig(threshold=0.5, indicators=["toxic"], cache_size=-1)

    # Invalid priority
    with pytest.raises(ValueError, match="Priority must be non-negative"):
        ToxicityConfig(threshold=0.5, indicators=["toxic"], priority=-1)

    # Invalid cost
    with pytest.raises(ValueError, match="Cost must be non-negative"):
        ToxicityConfig(threshold=0.5, indicators=["toxic"], cost=-1)


def test_bias_config_validation():
    """Test bias configuration validation."""
    # Valid configuration
    config = BiasConfig(threshold=0.5, categories={"gender": ["male", "female"]})
    assert config.threshold == 0.5
    assert config.categories == {"gender": ["male", "female"]}

    # Invalid threshold
    with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
        BiasConfig(threshold=1.5, categories={"gender": ["male"]})

    # Empty categories
    with pytest.raises(ValueError, match="Must provide at least one bias category"):
        BiasConfig(threshold=0.5, categories={})

    # Empty category indicators
    with pytest.raises(ValueError, match="Category gender must have at least one indicator"):
        BiasConfig(threshold=0.5, categories={"gender": []})


def test_harmful_content_config_validation():
    """Test harmful content configuration validation."""
    # Valid configuration
    config = HarmfulContentConfig(categories={"violence": ["kill", "hurt"]})
    assert config.categories == {"violence": ["kill", "hurt"]}

    # Empty categories
    with pytest.raises(ValueError, match="Must provide at least one harmful content category"):
        HarmfulContentConfig(categories={})

    # Empty category indicators
    with pytest.raises(ValueError, match="Category violence must have at least one indicator"):
        HarmfulContentConfig(categories={"violence": []})


def test_toxicity_rule_validation(toxicity_config):
    """Test toxicity rule validation."""
    rule = create_toxicity_rule(
        name="test_toxicity",
        description="Test toxicity rule",
        threshold=toxicity_config.threshold,
        indicators=toxicity_config.indicators,
    )

    # Test non-toxic content
    result = rule._validate_impl("This is a safe text.")
    assert result.passed
    assert "No toxic content detected" in result.message
    assert result.metadata["toxicity_score"] == 0.0

    # Test toxic content
    result = rule._validate_impl("This is toxic and offensive content.")
    assert not result.passed
    assert "toxic content" in result.message
    assert result.metadata["toxicity_score"] > toxicity_config.threshold
    assert "toxic" in result.metadata["toxic_indicators"]
    assert "offensive" in result.metadata["toxic_indicators"]

    # Test invalid input type
    with pytest.raises(ValueError, match="Input must be a string"):
        rule._validate_impl(123)  # type: ignore


def test_bias_rule_validation(bias_config):
    """Test bias rule validation."""
    rule = create_bias_rule(
        name="test_bias",
        description="Test bias rule",
        threshold=bias_config.threshold,
        categories=bias_config.categories,
    )

    # Test unbiased content
    result = rule._validate_impl("This is a neutral text.")
    assert result.passed
    assert "No biased content detected" in result.message
    assert all(score == 0.0 for score in result.metadata["bias_scores"].values())

    # Test biased content
    result = rule._validate_impl("Only male and white people are good at this.")
    assert not result.passed
    assert "biased content" in result.message
    assert result.metadata["bias_scores"]["gender"] > 0
    assert result.metadata["bias_scores"]["race"] > 0

    # Test invalid input type
    with pytest.raises(ValueError, match="Input must be a string"):
        rule._validate_impl(123)  # type: ignore


def test_harmful_content_rule_validation(harmful_content_config):
    """Test harmful content rule validation."""
    rule = create_harmful_content_rule(
        name="test_harmful",
        description="Test harmful content rule",
        categories=harmful_content_config.categories,
    )

    # Test safe content
    result = rule._validate_impl("This is a safe text.")
    assert result.passed
    assert "No harmful content detected" in result.message
    assert not result.metadata["harmful_content"]

    # Test harmful content
    result = rule._validate_impl("I want to kill and hurt others.")
    assert not result.passed
    assert "harmful content" in result.message
    assert "violence" in result.metadata["harmful_content"]
    assert "kill" in result.metadata["harmful_content"]["violence"]
    assert "hurt" in result.metadata["harmful_content"]["violence"]

    # Test invalid input type
    with pytest.raises(ValueError, match="Input must be a string"):
        rule._validate_impl(123)  # type: ignore


def test_create_toxicity_rule():
    """Test toxicity rule factory function."""
    rule = create_toxicity_rule(
        name="test_toxicity",
        description="Test toxicity rule",
    )
    assert isinstance(rule, ToxicityRule)
    assert rule.name == "test_toxicity"
    assert rule.description == "Test toxicity rule"


def test_create_bias_rule():
    """Test bias rule factory function."""
    rule = create_bias_rule(
        name="test_bias",
        description="Test bias rule",
    )
    assert isinstance(rule, BiasRule)
    assert rule.name == "test_bias"
    assert rule.description == "Test bias rule"


def test_create_harmful_content_rule():
    """Test harmful content rule factory function."""
    rule = create_harmful_content_rule(
        name="test_harmful",
        description="Test harmful content rule",
    )
    assert isinstance(rule, HarmfulContentRule)
    assert rule.name == "test_harmful"
    assert rule.description == "Test harmful content rule"
