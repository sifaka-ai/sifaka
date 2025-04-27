"""Tests for prohibited content rules."""

import pytest
from typing import Dict, Any, List, Set, Protocol, runtime_checkable, Final
from dataclasses import dataclass, field

from sifaka.rules.prohibited_content import (
    ProhibitedContentRule,
    ProhibitedContentConfig,
    ProhibitedContentValidator,
    DefaultProhibitedContentValidator,
    create_prohibited_content_rule,
)
from sifaka.rules.base import RuleResult


@pytest.fixture
def prohibited_config() -> ProhibitedContentConfig:
    """Create a test prohibited content configuration."""
    return ProhibitedContentConfig(
        prohibited_terms=frozenset(["bad", "worse", "worst"]),
        case_sensitive=False,
        cache_size=10,
        priority=2,
        cost=1.5,
    )


@pytest.fixture
def prohibited_validator(prohibited_config: ProhibitedContentConfig) -> ProhibitedContentValidator:
    """Create a test prohibited content validator."""
    return DefaultProhibitedContentValidator(prohibited_config)


@pytest.fixture
def prohibited_rule(
    prohibited_validator: ProhibitedContentValidator,
) -> ProhibitedContentRule:
    """Create a test prohibited content rule."""
    return ProhibitedContentRule(
        name="Test Prohibited Content Rule",
        description="Test prohibited content validation",
        validator=prohibited_validator,
    )


def test_prohibited_config_validation():
    """Test prohibited content configuration validation."""
    # Test valid configuration
    config = ProhibitedContentConfig(prohibited_terms={"bad", "worse"})
    assert "bad" in config.prohibited_terms
    assert "worse" in config.prohibited_terms
    assert not config.case_sensitive

    # Test invalid configurations
    with pytest.raises(ValueError, match="prohibited_terms set cannot be empty"):
        ProhibitedContentConfig(prohibited_terms=set())

    with pytest.raises(ValueError, match="all prohibited terms must be strings"):
        ProhibitedContentConfig(prohibited_terms={123, "bad"})  # type: ignore

    with pytest.raises(ValueError, match="prohibited terms cannot be empty or whitespace-only"):
        ProhibitedContentConfig(prohibited_terms={"bad", "  ", ""})

    # Test automatic conversion to frozenset
    config = ProhibitedContentConfig(prohibited_terms=["bad", "worse"])
    assert isinstance(config.prohibited_terms, frozenset)


def test_prohibited_validation(prohibited_rule: ProhibitedContentRule):
    """Test prohibited content validation."""
    # Test content without prohibited terms
    text = "This is a safe and acceptable message."
    result = prohibited_rule.validate(text)
    assert result.passed
    assert not result.metadata["found_terms"]
    assert result.metadata["total_terms"] == 0

    # Test content with prohibited terms
    text = "This contains a bad word."
    result = prohibited_rule.validate(text)
    assert not result.passed
    assert "bad" in result.metadata["found_terms"]
    assert result.metadata["total_terms"] == 1

    # Test multiple prohibited terms
    text = "This is bad and worse."
    result = prohibited_rule.validate(text)
    assert not result.passed
    assert set(result.metadata["found_terms"]) == {"bad", "worse"}
    assert result.metadata["total_terms"] == 2


def test_case_sensitivity():
    """Test case sensitivity in validation."""
    # Test case-insensitive matching (default)
    config = ProhibitedContentConfig(prohibited_terms={"Bad", "WORSE"})
    rule = create_prohibited_content_rule(config=config)

    result = rule.validate("This contains bad and worse words.")
    assert not result.passed
    assert set(result.metadata["found_terms"]) == {"Bad", "WORSE"}
    assert not result.metadata["case_sensitive"]

    # Test case-sensitive matching
    config = ProhibitedContentConfig(prohibited_terms={"Bad", "WORSE"}, case_sensitive=True)
    rule = create_prohibited_content_rule(config=config)

    result = rule.validate("This contains bad and worse words.")
    assert result.passed  # Should pass because case doesn't match
    assert not result.metadata["found_terms"]
    assert result.metadata["case_sensitive"]

    result = rule.validate("This contains Bad and WORSE words.")
    assert not result.passed  # Should fail because case matches
    assert set(result.metadata["found_terms"]) == {"Bad", "WORSE"}
    assert result.metadata["case_sensitive"]


def test_edge_cases(prohibited_rule: ProhibitedContentRule):
    """Test edge cases and error handling."""
    # Test empty text
    result = prohibited_rule.validate("")
    assert result.passed
    assert not result.metadata["found_terms"]
    assert result.metadata["total_terms"] == 0

    # Test whitespace-only text
    result = prohibited_rule.validate("   \n\t   ")
    assert result.passed
    assert not result.metadata["found_terms"]
    assert result.metadata["total_terms"] == 0

    # Test special characters
    text = "!@#$%^&*()"
    result = prohibited_rule.validate(text)
    assert result.passed
    assert not result.metadata["found_terms"]

    # Test Unicode characters
    text = "Hello 世界"
    result = prohibited_rule.validate(text)
    assert result.passed
    assert not result.metadata["found_terms"]


def test_error_handling(prohibited_rule: ProhibitedContentRule):
    """Test error handling for invalid inputs."""
    # Test None input
    with pytest.raises(ValueError, match="Text must be a string"):
        prohibited_rule.validate(None)  # type: ignore

    # Test non-string input
    with pytest.raises(ValueError, match="Text must be a string"):
        prohibited_rule.validate(123)  # type: ignore

    # Test list input
    with pytest.raises(ValueError, match="Text must be a string"):
        prohibited_rule.validate(["not", "a", "string"])  # type: ignore


def test_factory_function():
    """Test factory function for creating prohibited content rules."""
    # Test rule creation with default config
    rule = create_prohibited_content_rule(
        name="Test Rule",
        description="Test validation",
    )
    assert rule.name == "Test Rule"
    assert isinstance(rule.validator, DefaultProhibitedContentValidator)
    assert "inappropriate" in rule.validator.config.prohibited_terms
    assert "offensive" in rule.validator.config.prohibited_terms

    # Test rule creation with custom config
    config = ProhibitedContentConfig(prohibited_terms={"custom", "terms"})
    rule = create_prohibited_content_rule(
        name="Custom Rule",
        description="Custom validation",
        config=config,
    )
    assert "custom" in rule.validator.config.prohibited_terms
    assert "terms" in rule.validator.config.prohibited_terms

    # Test rule creation with custom validator
    validator = DefaultProhibitedContentValidator(
        ProhibitedContentConfig(prohibited_terms={"test"})
    )
    rule = create_prohibited_content_rule(
        name="Validator Rule",
        description="Validator test",
        validator=validator,
    )
    assert rule.validator is validator


def test_consistent_results(prohibited_rule: ProhibitedContentRule):
    """Test that validation results are consistent."""
    text = "This text contains a bad word that should be consistently detected."

    # Multiple validations should yield the same result
    result1 = prohibited_rule.validate(text)
    result2 = prohibited_rule.validate(text)
    assert result1.passed == result2.passed
    assert result1.message == result2.message
    assert result1.metadata == result2.metadata
