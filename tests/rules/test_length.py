"""
Tests for the LengthRule module of Sifaka.
"""

import pytest

from sifaka.rules.length import LengthRule, LengthConfig, DefaultLengthValidator, create_length_rule
from sifaka.rules.base import RuleConfig


class TestLengthConfig:
    """Test suite for LengthConfig class."""

    def test_length_config_default_values(self):
        """Test LengthConfig initialization with default values."""
        config = LengthConfig()
        assert config.min_length == 50
        assert config.max_length == 5000
        assert config.exact_length is None
        assert config.unit == "characters"
        assert config.cache_size == 10
        assert config.priority == 2
        assert config.cost == 1.5

    def test_length_config_custom_values(self):
        """Test LengthConfig initialization with custom values."""
        config = LengthConfig(
            min_length=10, max_length=1000, unit="words", cache_size=5, priority=1, cost=2.0
        )
        assert config.min_length == 10
        assert config.max_length == 1000
        assert config.unit == "words"
        assert config.cache_size == 5
        assert config.priority == 1
        assert config.cost == 2.0

    def test_length_config_exact_length(self):
        """Test LengthConfig with exact_length."""
        config = LengthConfig(exact_length=100)
        assert config.exact_length == 100
        assert config.min_length == 50  # Default values remain unchanged but not used
        assert config.max_length == 5000  # Default values remain unchanged but not used

    def test_length_config_invalid_unit(self):
        """Test LengthConfig with invalid unit raises ValueError."""
        with pytest.raises(ValueError):
            LengthConfig(unit="paragraphs")

    def test_length_config_negative_exact_length(self):
        """Test LengthConfig with negative exact_length raises ValueError."""
        with pytest.raises(ValueError):
            LengthConfig(exact_length=-10)

    def test_length_config_exact_length_with_min_max(self):
        """Test LengthConfig with exact_length and customized min/max raises ValueError."""
        with pytest.raises(ValueError):
            LengthConfig(exact_length=100, min_length=10)

    def test_length_config_negative_min_length(self):
        """Test LengthConfig with negative min_length raises ValueError."""
        with pytest.raises(ValueError):
            LengthConfig(min_length=-10)

    def test_length_config_negative_max_length(self):
        """Test LengthConfig with negative max_length raises ValueError."""
        with pytest.raises(ValueError):
            LengthConfig(max_length=-10)

    def test_length_config_min_greater_than_max(self):
        """Test LengthConfig with min_length > max_length raises ValueError."""
        with pytest.raises(ValueError):
            LengthConfig(min_length=100, max_length=50)


class TestDefaultLengthValidator:
    """Test suite for DefaultLengthValidator class."""

    def test_validator_character_count(self):
        """Test validator counts characters correctly."""
        config = LengthConfig(min_length=10, max_length=100, unit="characters")
        validator = DefaultLengthValidator(config)

        # Test text with 15 characters
        result = validator.validate("Hello, world!")
        assert result.passed is True
        assert result.metadata["length"] == 13

        # Test text with 5 characters (too short)
        result = validator.validate("Hello")
        assert result.passed is False
        assert result.metadata["length"] == 5
        assert "below minimum" in result.message

        # Test text with 150 characters (too long)
        long_text = "a" * 150
        result = validator.validate(long_text)
        assert result.passed is False
        assert result.metadata["length"] == 150
        assert "exceeds maximum" in result.message

    def test_validator_word_count(self):
        """Test validator counts words correctly."""
        config = LengthConfig(min_length=3, max_length=10, unit="words")
        validator = DefaultLengthValidator(config)

        # Test text with 5 words
        result = validator.validate("This is a test sentence")
        assert result.passed is True
        assert result.metadata["length"] == 5

        # Test text with 2 words (too short)
        result = validator.validate("Hello world")
        assert result.passed is False
        assert result.metadata["length"] == 2
        assert "below minimum" in result.message

        # Test text with 15 words (too long)
        long_text = " ".join(["word"] * 15)
        result = validator.validate(long_text)
        assert result.passed is False
        assert result.metadata["length"] == 15
        assert "exceeds maximum" in result.message

    def test_validator_exact_length(self):
        """Test validator with exact_length."""
        config = LengthConfig(exact_length=5, unit="words")
        validator = DefaultLengthValidator(config)

        # Test text with exactly 5 words
        result = validator.validate("This is a test sentence")
        assert result.passed is True
        assert result.metadata["length"] == 5

        # Test text with 4 words (not exact match)
        result = validator.validate("This is a test")
        assert result.passed is False
        assert result.metadata["length"] == 4
        assert "does not match required count" in result.message

    def test_validator_empty_text(self):
        """Test validator with empty text."""
        config = LengthConfig(min_length=1)
        validator = DefaultLengthValidator(config)

        result = validator.validate("")
        assert result.passed is False
        assert "Empty or whitespace-only text" in result.message

        result = validator.validate("   ")
        assert result.passed is False
        assert "Empty or whitespace-only text" in result.message

    def test_validator_non_string_input(self):
        """Test validator with non-string input raises ValueError."""
        config = LengthConfig()
        validator = DefaultLengthValidator(config)

        with pytest.raises(ValueError):
            validator.validate(123)


class TestLengthRule:
    """Test suite for LengthRule class."""

    def test_length_rule_default_initialization(self):
        """Test LengthRule initialization with default values."""
        rule = LengthRule()
        assert rule.name == "length_rule"
        assert "Checks if text length is within bounds" in rule.description

    def test_length_rule_custom_initialization(self):
        """Test LengthRule initialization with custom values."""
        rule_config = RuleConfig(params={"min_length": 10, "max_length": 100, "unit": "words"})
        rule = LengthRule(
            name="custom_length_rule", description="Custom length rule", config=rule_config
        )
        assert rule.name == "custom_length_rule"
        assert rule.description == "Custom length rule"

        # Validate a text
        result = rule.validate("This is a test with exactly seven words")
        assert result.passed is False  # 8 words, below min_length of 10
        assert result.metadata["length"] == 8

    def test_create_length_rule_factory(self):
        """Test create_length_rule factory function."""
        rule = create_length_rule(
            name="factory_rule",
            description="Factory created rule",
            config={"min_length": 5, "max_length": 20, "unit": "words"},
        )
        assert rule.name == "factory_rule"
        assert rule.description == "Factory created rule"

        # Validate a text
        result = rule.validate("This is a test with exactly seven words")
        assert result.passed is True
        assert result.metadata["length"] == 8
