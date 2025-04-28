"""Tests for configuration consistency across rules and classifiers."""

import pytest

from sifaka.rules.base import RuleConfig
from sifaka.rules.content import ToneConfig
from sifaka.rules.content import DefaultToneValidator
from sifaka.rules.prohibited_content import (
    ProhibitedContentRule,
    DefaultProhibitedContentValidator,
    create_prohibited_content_rule,
)


def test_rule_config_params_consistency():
    """Test that RuleConfig consistently uses params."""
    # Test that params is used when provided
    config = RuleConfig(params={"key": "value"})
    assert config.params == {"key": "value"}
    assert config.metadata == {}

    # Test that metadata is copied to params when params is empty
    config = RuleConfig(metadata={"key": "value"})
    assert config.params == {"key": "value"}
    assert config.metadata == {"key": "value"}

    # Test that with_params creates a new config with updated params
    config = RuleConfig(params={"key1": "value1"})
    new_config = config.with_params(key2="value2")
    assert new_config.params == {"key1": "value1", "key2": "value2"}
    assert config.params == {"key1": "value1"}  # Original unchanged


def test_tone_config_params_copying():
    """Test that ToneConfig copies values to params."""
    # Create a ToneConfig with custom values
    config = ToneConfig(
        expected_tone="formal",
        threshold=0.8,
        priority=3,
        cost=2.0,
        tone_indicators={
            "formal": {
                "positive": ["therefore", "consequently"],
                "negative": ["yo", "hey"],
            }
        },
    )

    # Verify that values are copied to params
    assert "expected_tone" in config.params
    assert config.params["expected_tone"] == "formal"
    assert "threshold" in config.params
    assert config.params["threshold"] == 0.8
    assert "priority" in config.params
    assert config.params["priority"] == 3
    assert "cost" in config.params
    assert config.params["cost"] == 2.0
    assert "tone_indicators" in config.params
    assert "formal" in config.params["tone_indicators"]


def test_prohibited_content_rule_params_usage():
    """Test that ProhibitedContentRule uses params correctly."""
    # Create a RuleConfig with params for prohibited content
    config = RuleConfig(
        priority=3,
        cost=2.0,
        params={
            "prohibited_terms": ["bad", "worse"],
            "case_sensitive": True,
        },
    )

    # Create a rule with this config
    rule = ProhibitedContentRule(
        name="Test Rule",
        description="Test rule",
        config=config,
    )

    # Verify that the validator uses the params correctly
    validator = rule._validator
    assert isinstance(validator, DefaultProhibitedContentValidator)
    assert set(validator.config.prohibited_terms) == set(["bad", "worse"])
    assert validator.config.case_sensitive is True


def test_default_tone_validator_params_usage():
    """Test that DefaultToneValidator uses values from params."""
    # Create a ToneConfig with params
    config = ToneConfig(
        expected_tone="formal",
        threshold=0.8,
        tone_indicators={
            "formal": {
                "positive": ["therefore", "consequently"],
                "negative": ["yo", "hey"],
            }
        },
        params={
            "expected_tone": "informal",  # Different from attribute
            "threshold": 0.6,  # Different from attribute
            "tone_indicators": {
                "informal": {
                    "positive": ["cool", "awesome"],
                    "negative": ["therefore", "consequently"],
                }
            },
        },
    )

    # Create validator with this config
    validator = DefaultToneValidator(config)

    # Validate some text that would pass with params values but fail with attribute values
    text = "This is a cool and awesome text!"
    result = validator.validate(text)

    # Should use params values (informal) instead of attribute values (formal)
    assert result.passed
    assert "informal" in result.message
    assert result.metadata["expected_tone"] == "informal"
    assert result.metadata["threshold"] == 0.6


def test_create_prohibited_content_rule_with_params():
    """Test that create_prohibited_content_rule uses params correctly."""
    # Create a rule with params
    rule = create_prohibited_content_rule(
        name="Test Rule",
        description="Test rule",
        config={
            "prohibited_terms": ["terrible", "awful"],
            "case_sensitive": True,
        },
    )

    # Verify that the validator uses the params correctly
    validator = rule._validator
    assert isinstance(validator, DefaultProhibitedContentValidator)
    assert set(validator.config.prohibited_terms) == set(["terrible", "awful"])
    assert validator.config.case_sensitive is True

    # Test validation with the rule
    # Text without prohibited terms
    text = "This is a bad and worse text."
    result = rule.validate(text)
    assert result.passed
    assert "No prohibited terms found" in result.message
    assert not result.metadata["found_terms"]
    assert result.metadata["case_sensitive"] is True

    # Text with prohibited terms
    text = "This is a terrible and awful text."
    result = rule.validate(text)
    assert not result.passed
    assert "Found prohibited terms" in result.message
    assert set(result.metadata["found_terms"]) == set(["terrible", "awful"])
    assert result.metadata["case_sensitive"] is True
