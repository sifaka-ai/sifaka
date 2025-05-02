"""Test module for sifaka.rules.formatting.whitespace."""

import pytest
from pydantic import ValidationError

from sifaka.rules.formatting.whitespace import (
    WhitespaceConfig,
    WhitespaceValidator,
    DefaultWhitespaceValidator,
    WhitespaceRule,
    create_whitespace_rule,
)


class TestWhitespaceConfig:
    """Tests for WhitespaceConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WhitespaceConfig()
        assert config.allow_leading_whitespace is False
        assert config.allow_trailing_whitespace is False
        assert config.allow_multiple_spaces is False
        assert config.allow_tabs is False
        assert config.allow_newlines is True
        assert config.max_newlines is None
        assert config.normalize_whitespace is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = WhitespaceConfig(
            allow_leading_whitespace=True,
            allow_trailing_whitespace=True,
            allow_multiple_spaces=True,
            allow_tabs=True,
            allow_newlines=False,
            normalize_whitespace=True,
        )
        assert config.allow_leading_whitespace is True
        assert config.allow_trailing_whitespace is True
        assert config.allow_multiple_spaces is True
        assert config.allow_tabs is True
        assert config.allow_newlines is False
        assert config.max_newlines is None
        assert config.normalize_whitespace is True

    def test_max_newlines_validation(self):
        """Test validation of max_newlines with allow_newlines."""
        # Valid: allow_newlines=True, max_newlines=2
        config = WhitespaceConfig(allow_newlines=True, max_newlines=2)
        assert config.max_newlines == 2

        # Invalid: allow_newlines=False, max_newlines=2
        with pytest.raises(ValidationError):
            WhitespaceConfig(allow_newlines=False, max_newlines=2)

    def test_max_newlines_range(self):
        """Test validation of max_newlines range."""
        # Valid: max_newlines=0
        config = WhitespaceConfig(max_newlines=0)
        assert config.max_newlines == 0

        # Invalid: max_newlines=-1
        with pytest.raises(ValidationError):
            WhitespaceConfig(max_newlines=-1)


class TestDefaultWhitespaceValidator:
    """Tests for DefaultWhitespaceValidator."""

    def test_initialization(self):
        """Test initialization with config."""
        config = WhitespaceConfig()
        validator = DefaultWhitespaceValidator(config)
        assert validator.config == config

    def test_validate_no_errors(self):
        """Test validation with no errors."""
        config = WhitespaceConfig()
        validator = DefaultWhitespaceValidator(config)
        text = "This is a valid text."
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

    def test_validate_leading_whitespace(self):
        """Test validation of leading whitespace."""
        # Not allowed (default)
        config = WhitespaceConfig(allow_leading_whitespace=False)
        validator = DefaultWhitespaceValidator(config)
        text = " Leading whitespace"
        is_valid, errors = validator.validate(text)
        assert is_valid is False
        assert "leading whitespace" in errors[0].lower()

        # Allowed
        config = WhitespaceConfig(allow_leading_whitespace=True)
        validator = DefaultWhitespaceValidator(config)
        text = " Leading whitespace"
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

    def test_validate_trailing_whitespace(self):
        """Test validation of trailing whitespace."""
        # Not allowed (default)
        config = WhitespaceConfig(allow_trailing_whitespace=False)
        validator = DefaultWhitespaceValidator(config)
        text = "Trailing whitespace "
        is_valid, errors = validator.validate(text)
        assert is_valid is False
        assert "trailing whitespace" in errors[0].lower()

        # Allowed
        config = WhitespaceConfig(allow_trailing_whitespace=True)
        validator = DefaultWhitespaceValidator(config)
        text = "Trailing whitespace "
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

    def test_validate_multiple_spaces(self):
        """Test validation of multiple consecutive spaces."""
        # Not allowed (default)
        config = WhitespaceConfig(allow_multiple_spaces=False)
        validator = DefaultWhitespaceValidator(config)
        text = "Multiple  spaces"
        is_valid, errors = validator.validate(text)
        assert is_valid is False
        assert "multiple consecutive spaces" in errors[0].lower()

        # Allowed
        config = WhitespaceConfig(allow_multiple_spaces=True)
        validator = DefaultWhitespaceValidator(config)
        text = "Multiple  spaces"
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

    def test_validate_tabs(self):
        """Test validation of tab characters."""
        # Not allowed (default)
        config = WhitespaceConfig(allow_tabs=False)
        validator = DefaultWhitespaceValidator(config)
        text = "Contains\ttab"
        is_valid, errors = validator.validate(text)
        assert is_valid is False
        assert "tab" in errors[0].lower()

        # Allowed
        config = WhitespaceConfig(allow_tabs=True)
        validator = DefaultWhitespaceValidator(config)
        text = "Contains\ttab"
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

    def test_validate_newlines(self):
        """Test validation of newline characters."""
        # Allowed (default)
        config = WhitespaceConfig(allow_newlines=True)
        validator = DefaultWhitespaceValidator(config)
        text = "Contains\nnewline"
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

        # Not allowed
        config = WhitespaceConfig(allow_newlines=False)
        validator = DefaultWhitespaceValidator(config)
        text = "Contains\nnewline"
        is_valid, errors = validator.validate(text)
        assert is_valid is False
        assert "newline" in errors[0].lower()

    def test_validate_max_newlines(self):
        """Test validation of maximum consecutive newlines."""
        # Max 1 newline
        config = WhitespaceConfig(allow_newlines=True, max_newlines=1)
        validator = DefaultWhitespaceValidator(config)

        # Valid: 1 newline
        text = "One\nnewline"
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

        # Invalid: 2 newlines
        text = "Two\n\nnewlines"
        is_valid, errors = validator.validate(text)
        assert is_valid is False
        assert "consecutive newlines" in errors[0].lower()
        assert "maximum allowed is 1" in errors[0].lower()

    def test_normalize_whitespace(self):
        """Test normalization of whitespace."""
        config = WhitespaceConfig(
            allow_leading_whitespace=False,
            allow_trailing_whitespace=False,
            allow_multiple_spaces=False,
            allow_tabs=False,
            normalize_whitespace=True,
        )
        validator = DefaultWhitespaceValidator(config)

        # Text with various whitespace issues
        text = " \t Multiple  spaces and\ttabs with trailing space "

        # With normalization enabled, validation should pass
        is_valid, errors = validator.validate(text)
        assert is_valid is True
        assert errors == []

        # Test the internal normalization method directly
        normalized = validator._normalize_whitespace(text)
        assert not normalized.startswith(" ")
        assert not normalized.endswith(" ")
        assert "  " not in normalized
        assert "\t" not in normalized

    def test_multiple_validation_issues(self):
        """Test validation with multiple issues."""
        config = WhitespaceConfig()
        validator = DefaultWhitespaceValidator(config)
        text = " Leading and trailing \t with tab and  multiple spaces"
        is_valid, errors = validator.validate(text)
        assert is_valid is False
        assert len(errors) == 3  # Leading, multiple spaces, and tab issues


class TestWhitespaceRule:
    """Tests for WhitespaceRule."""

    def test_initialization(self):
        """Test initialization with validator."""
        config = WhitespaceConfig()
        validator = DefaultWhitespaceValidator(config)
        rule = WhitespaceRule(validator=validator)
        assert rule.validator == validator
        assert rule.id == "rule"  # Default ID

    def test_custom_rule_id(self):
        """Test initialization with custom ID."""
        config = WhitespaceConfig()
        validator = DefaultWhitespaceValidator(config)
        rule = WhitespaceRule(validator=validator, id="custom_id")
        assert rule.id == "custom_id"

    def test_validate(self):
        """Test validation through the rule."""
        config = WhitespaceConfig()
        validator = DefaultWhitespaceValidator(config)
        rule = WhitespaceRule(validator=validator)

        # Valid text
        result = rule.validate("Valid text")
        assert result.passed is True
        assert result.rule_id == "rule"
        assert result.errors == []

        # Invalid text
        result = rule.validate(" Invalid text with leading space")
        assert result.passed is False
        assert result.rule_id == "rule"
        assert len(result.errors) == 1
        assert "leading whitespace" in result.errors[0].lower()


class TestCreateWhitespaceRule:
    """Tests for create_whitespace_rule helper function."""

    def test_default_creation(self):
        """Test creating rule with default parameters."""
        rule = create_whitespace_rule()
        assert isinstance(rule, WhitespaceRule)
        assert rule.id == "whitespace"  # Default ID

        # Test with default validator config
        result = rule.validate("Valid text")
        assert result.passed is True

        result = rule.validate(" Invalid text")
        assert result.passed is False

    def test_custom_creation(self):
        """Test creating rule with custom parameters."""
        rule = create_whitespace_rule(
            allow_leading_whitespace=True,
            allow_trailing_whitespace=True,
            allow_multiple_spaces=True,
            allow_tabs=True,
            allow_newlines=False,
            max_newlines=None,
            normalize_whitespace=True,
            rule_id="custom_whitespace",
            name="Custom Whitespace Rule",
            description="A custom whitespace rule",
        )

        assert isinstance(rule, WhitespaceRule)
        assert rule.id == "custom_whitespace"
        assert rule.name == "Custom Whitespace Rule"
        assert rule.description == "A custom whitespace rule"

        # Test with custom validator config
        result = rule.validate(" Leading whitespace is allowed ")
        assert result.passed is True

        result = rule.validate("Newlines are\nnot allowed")
        assert result.passed is False