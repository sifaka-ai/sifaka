"""Tests for length rules."""


import pytest

from sifaka.rules.base import RuleResult
from sifaka.rules.length import (
    DefaultLengthValidator,
    LengthConfig,
    LengthRule,
    LengthValidator,
    create_length_rule,
)

@pytest.fixture
def length_config() -> LengthConfig:
    """Create a test length configuration."""
    return LengthConfig(
        min_length=10,
        max_length=100,
        unit="characters",
        cache_size=10,
        priority=2,
        cost=1.5,
    )

@pytest.fixture
def length_validator(length_config: LengthConfig) -> LengthValidator:
    """Create a test length validator."""
    return DefaultLengthValidator(length_config)

@pytest.fixture
def length_rule(
    length_validator: LengthValidator,
) -> LengthRule:
    """Create a test length rule."""
    return LengthRule(
        name="Test Length Rule",
        description="Test length validation",
        validator=length_validator,
    )

def test_length_config_validation():
    """Test length configuration validation."""
    # Test valid configuration
    config = LengthConfig(min_length=10, max_length=100)
    assert config.min_length == 10
    assert config.max_length == 100

    # Test invalid configurations
    with pytest.raises(ValueError, match="min_length must be non-negative"):
        LengthConfig(min_length=-1)

    with pytest.raises(ValueError, match="max_length must be greater than min_length"):
        LengthConfig(min_length=100, max_length=10)

    with pytest.raises(ValueError, match="unit must be one of"):
        LengthConfig(unit="invalid")

    # Test exact length validation
    with pytest.raises(ValueError, match="exact_length must be non-negative"):
        LengthConfig(exact_length=-1)

    with pytest.raises(
        ValueError, match="exact_length cannot be used with min_length or max_length"
    ):
        LengthConfig(exact_length=50, min_length=10)

def test_length_validation(length_rule: LengthRule):
    """Test length validation."""
    # Test valid length
    text = "This is a valid length text for testing."
    result = length_rule.validate(text)
    assert result.passed
    assert result.metadata["length"] >= length_rule.validator.config.min_length

    # Test too short
    text = "Too short"
    result = length_rule.validate(text)
    assert not result.passed
    assert "below minimum" in result.message
    assert result.metadata["length"] < length_rule.validator.config.min_length

    # Test too long
    text = "x" * (length_rule.validator.config.max_length + 1)
    result = length_rule.validate(text)
    assert not result.passed
    assert "exceeds maximum" in result.message
    assert result.metadata["length"] > length_rule.validator.config.max_length

def test_word_length_validation():
    """Test validation with word length."""
    config = LengthConfig(
        min_length=3,
        max_length=10,
        unit="words",
    )
    rule = create_length_rule(
        name="Word Length Rule",
        description="Test word length validation",
        config=config,
    )

    # Test valid word count
    text = "One two three four"
    result = rule.validate(text)
    assert result.passed
    assert result.metadata["length"] == 4

    # Test too few words
    text = "Two words"
    result = rule.validate(text)
    assert not result.passed
    assert "below minimum" in result.message
    assert result.metadata["length"] == 2

    # Test too many words
    text = "One two three four five six seven eight nine ten eleven"
    result = rule.validate(text)
    assert not result.passed
    assert "exceeds maximum" in result.message
    assert result.metadata["length"] == 11

def test_exact_length_validation():
    """Test validation with exact length requirement."""
    config = LengthConfig(
        exact_length=10,
        min_length=50,  # These will be ignored due to exact_length
        max_length=5000,  # These will be ignored due to exact_length
    )
    rule = create_length_rule(
        name="Exact Length Rule",
        description="Test exact length validation",
        config=config,
    )

    # Test exact match
    text = "0123456789"
    result = rule.validate(text)
    assert result.passed
    assert result.metadata["length"] == 10

    # Test too short
    text = "012345678"
    result = rule.validate(text)
    assert not result.passed
    assert "does not match required count" in result.message

    # Test too long
    text = "0123456789A"
    result = rule.validate(text)
    assert not result.passed
    assert "does not match required count" in result.message

def test_edge_cases(length_rule: LengthRule):
    """Test edge cases and error handling."""
    # Test empty text
    result = length_rule.validate("")
    assert not result.passed
    assert "Empty or whitespace-only text" in result.message

    # Test whitespace-only text
    result = length_rule.validate("   \n\t   ")
    assert not result.passed
    assert "Empty or whitespace-only text" in result.message

    # Test special characters
    text = "!@#$%^&*()"
    result = length_rule.validate(text)
    assert isinstance(result, RuleResult)
    assert result.metadata["length"] == len(text)

    # Test Unicode characters
    text = "Hello 世界"
    result = length_rule.validate(text)
    assert isinstance(result, RuleResult)
    assert result.metadata["length"] == len(text)

def test_error_handling(length_rule: LengthRule):
    """Test error handling for invalid inputs."""
    # Test None input
    with pytest.raises(ValueError, match="Text must be a string"):
        length_rule.validate(None)  # type: ignore

    # Test non-string input
    with pytest.raises(ValueError, match="Text must be a string"):
        length_rule.validate(123)  # type: ignore

    # Test list input
    with pytest.raises(ValueError, match="Text must be a string"):
        length_rule.validate(["not", "a", "string"])  # type: ignore

def test_factory_function():
    """Test factory function for creating length rules."""
    # Test rule creation with default config
    rule = create_length_rule(
        name="Test Rule",
        description="Test validation",
    )
    assert rule.name == "Test Rule"
    assert isinstance(rule.validator, DefaultLengthValidator)

    # Test rule creation with custom config
    config = LengthConfig(min_length=20, max_length=200)
    rule = create_length_rule(
        name="Custom Rule",
        description="Custom validation",
        config=config,
    )
    assert rule.validator.config.min_length == 20
    assert rule.validator.config.max_length == 200

    # Test rule creation with custom validator
    validator = DefaultLengthValidator(LengthConfig())
    rule = create_length_rule(
        name="Validator Rule",
        description="Validator test",
        validator=validator,
    )
    assert rule.validator is validator

def test_consistent_results(length_rule: LengthRule):
    """Test that validation results are consistent."""
    text = "This is a test text that should be consistently validated."

    # Multiple validations should yield the same result
    result1 = length_rule.validate(text)
    result2 = length_rule.validate(text)
    assert result1.passed == result2.passed
    assert result1.message == result2.message
    assert result1.metadata == result2.metadata
