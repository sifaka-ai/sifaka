"""
Tests for the LengthRule.
"""

import pytest
from sifaka.rules.formatting.length import LengthRule, create_length_rule
from sifaka.utils.config import RuleConfig


def test_length_rule_initialization():
    """Test that the LengthRule can be initialized."""
    from sifaka.rules.formatting.length import create_length_validator

    # Create a validator first
    validator = create_length_validator(
        min_chars=10,
        max_chars=100,
        min_words=2,
        max_words=20,
    )

    rule = LengthRule(
        validator=validator,
        name="test_length_rule",
        description="Test length rule",
        config=RuleConfig(
            params={
                "min_chars": 10,
                "max_chars": 100,
                "min_words": 2,
                "max_words": 20,
            }
        ),
    )

    assert rule is not None
    assert rule.name == "test_length_rule"
    assert rule.description == "Test length rule"
    # Get config from state
    config = rule._state_manager.get("config")
    assert config.min_chars == 10
    assert config.max_chars == 100
    assert config.min_words == 2
    assert config.max_words == 20


def test_length_rule_factory():
    """Test that the length rule factory works."""
    rule = create_length_rule(
        min_chars=10,
        max_chars=100,
        min_words=2,
        max_words=20,
        name="test_length_rule",
        description="Test length rule",
    )

    assert rule is not None
    assert rule.name == "test_length_rule"
    assert rule.description == "Test length rule"
    # Get config from state
    config = rule._state_manager.get("config")
    assert config.min_chars == 10
    assert config.max_chars == 100
    assert config.min_words == 2
    assert config.max_words == 20


def test_length_rule_validate_success():
    """Test that the LengthRule validates correctly when text meets criteria."""
    rule = create_length_rule(
        min_chars=5,
        max_chars=100,
        min_words=2,
        max_words=20,
    )

    result = rule.model_validate("This is a test text.")

    assert result is not None
    assert result.passed is True
    assert "validation successful" in result.message.lower()
    assert "char_count" in result.metadata
    assert "word_count" in result.metadata
    assert result.metadata["char_count"] == 20
    assert result.metadata["word_count"] == 5


def test_length_rule_validate_too_short_chars():
    """Test that the LengthRule fails when text is too short (chars)."""
    rule = create_length_rule(
        min_chars=10,
        max_chars=100,
    )

    result = rule.model_validate("Short.")

    assert result is not None
    assert result.passed is False
    assert "too short" in result.message.lower()
    assert "char_count" in result.metadata
    assert result.metadata["char_count"] == 6


def test_length_rule_validate_too_long_chars():
    """Test that the LengthRule fails when text is too long (chars)."""
    rule = create_length_rule(
        min_chars=5,
        max_chars=10,
    )

    result = rule.model_validate("This text is too long for the maximum character limit.")

    assert result is not None
    assert result.passed is False
    assert "too long" in result.message.lower()
    assert "char_count" in result.metadata
    assert result.metadata["char_count"] > 10


def test_length_rule_validate_too_few_words():
    """Test that the LengthRule fails when text has too few words."""
    rule = create_length_rule(
        min_words=3,
        max_words=10,
    )

    result = rule.model_validate("Too few.")

    assert result is not None
    assert result.passed is False
    assert "too few" in result.message.lower()
    assert "word_count" in result.metadata
    assert result.metadata["word_count"] == 2


def test_length_rule_validate_too_many_words():
    """Test that the LengthRule fails when text has too many words."""
    rule = create_length_rule(
        min_words=2,
        max_words=5,
    )

    result = rule.model_validate("This text has too many words for the maximum word limit.")

    assert result is not None
    assert result.passed is False
    assert "too many" in result.message.lower()
    assert "word_count" in result.metadata
    assert result.metadata["word_count"] > 5
