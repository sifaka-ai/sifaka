"""
Tests for the style validation module.

This module contains tests for the style validation functionality,
including tests for StyleValidator, FormattingValidator, StyleRule,
FormattingRule, and their factory functions.
"""

import pytest

from sifaka.rules.formatting.style import (
    CapitalizationStyle,
    StyleConfig,
    FormattingConfig,
    DefaultStyleValidator,
    DefaultFormattingValidator,
    StyleRule,
    FormattingRule,
    create_style_validator,
    create_style_rule,
    create_formatting_validator,
    create_formatting_rule,
)


class TestCapitalizationStyle:
    """Tests for the CapitalizationStyle enum."""

    def test_enum_values(self):
        """Test that the enum has the expected values."""
        assert CapitalizationStyle.SENTENCE_CASE.name == "SENTENCE_CASE"
        assert CapitalizationStyle.TITLE_CASE.name == "TITLE_CASE"
        assert CapitalizationStyle.LOWERCASE.name == "LOWERCASE"
        assert CapitalizationStyle.UPPERCASE.name == "UPPERCASE"
        assert CapitalizationStyle.CAPITALIZE_FIRST.name == "CAPITALIZE_FIRST"


class TestStyleConfig:
    """Tests for the StyleConfig class."""

    def test_default_values(self):
        """Test that the default values are as expected."""
        config = StyleConfig()
        assert config.capitalization is None
        assert config.require_end_punctuation is False
        assert config.allowed_end_chars is None
        assert config.disallowed_chars is None
        assert config.strip_whitespace is True

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = StyleConfig(
            capitalization=CapitalizationStyle.SENTENCE_CASE,
            require_end_punctuation=True,
            allowed_end_chars=[".", "!", "?"],
            disallowed_chars=["@", "#"],
            strip_whitespace=False,
        )
        assert config.capitalization == CapitalizationStyle.SENTENCE_CASE
        assert config.require_end_punctuation is True
        assert config.allowed_end_chars == [".", "!", "?"]
        assert config.disallowed_chars == ["@", "#"]
        assert config.strip_whitespace is False


class TestFormattingConfig:
    """Tests for the FormattingConfig class."""

    def test_default_values(self):
        """Test that the default values are as expected."""
        config = FormattingConfig()
        assert config.style_config is None
        assert config.strip_whitespace is True
        assert config.normalize_whitespace is False
        assert config.remove_extra_lines is False

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        config = FormattingConfig(
            style_config=style_config,
            strip_whitespace=False,
            normalize_whitespace=True,
            remove_extra_lines=True,
        )
        assert config.style_config == style_config
        assert config.strip_whitespace is False
        assert config.normalize_whitespace is True
        assert config.remove_extra_lines is True


class TestDefaultStyleValidator:
    """Tests for the DefaultStyleValidator class."""

    def test_empty_text(self):
        """Test that empty text is handled correctly."""
        validator = DefaultStyleValidator(StyleConfig())
        result = validator.validate("")
        assert result.passed is False
        assert "empty" in result.message.lower()

    def test_capitalization_sentence_case(self):
        """Test sentence case validation."""
        validator = DefaultStyleValidator(
            StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        )
        # Valid sentence case
        result = validator.validate("This is a test.")
        assert result.passed is True
        # Invalid sentence case
        result = validator.validate("this is a test.")
        assert result.passed is False
        assert "sentence case" in result.message.lower()

    def test_capitalization_lowercase(self):
        """Test lowercase validation."""
        validator = DefaultStyleValidator(StyleConfig(capitalization=CapitalizationStyle.LOWERCASE))
        # Valid lowercase
        result = validator.validate("this is a test.")
        assert result.passed is True
        # Invalid lowercase
        result = validator.validate("This is a test.")
        assert result.passed is False
        assert "lowercase" in result.message.lower()

    def test_end_punctuation(self):
        """Test end punctuation validation."""
        validator = DefaultStyleValidator(StyleConfig(require_end_punctuation=True))
        # Valid end punctuation
        result = validator.validate("This is a test.")
        assert result.passed is True
        # Invalid end punctuation
        result = validator.validate("This is a test")
        assert result.passed is False
        assert "punctuation" in result.message.lower()

    def test_allowed_end_chars(self):
        """Test allowed end characters validation."""
        validator = DefaultStyleValidator(
            StyleConfig(require_end_punctuation=True, allowed_end_chars=[".", "!"])
        )
        # Valid end character
        result = validator.validate("This is a test.")
        assert result.passed is True
        result = validator.validate("This is a test!")
        assert result.passed is True
        # Invalid end character
        result = validator.validate("This is a test?")
        assert result.passed is False
        assert "end with one of" in result.message.lower()

    def test_disallowed_chars(self):
        """Test disallowed characters validation."""
        validator = DefaultStyleValidator(StyleConfig(disallowed_chars=["@", "#"]))
        # Valid text without disallowed characters
        result = validator.validate("This is a test.")
        assert result.passed is True
        # Invalid text with disallowed characters
        result = validator.validate("This is a #test.")
        assert result.passed is False
        assert "disallowed character" in result.message.lower()


class TestDefaultFormattingValidator:
    """Tests for the DefaultFormattingValidator class."""

    def test_empty_text(self):
        """Test that empty text is handled correctly."""
        validator = DefaultFormattingValidator(FormattingConfig())
        result = validator.validate("")
        assert result.passed is False
        assert "empty" in result.message.lower()

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        validator = DefaultFormattingValidator(FormattingConfig(strip_whitespace=True))
        result = validator.validate("  This is a test.  ")
        assert result.passed is True
        assert result.metadata["original_length"] > result.metadata["formatted_length"]

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        validator = DefaultFormattingValidator(FormattingConfig(normalize_whitespace=True))
        result = validator.validate("This   is  a   test.")
        assert result.passed is True
        assert "normalize whitespace" in result.suggestions[0].lower()

    def test_remove_extra_lines(self):
        """Test extra line removal."""
        validator = DefaultFormattingValidator(FormattingConfig(remove_extra_lines=True))
        result = validator.validate("This is a test.\n\n\nAnother line.")
        assert result.passed is True
        assert "remove extra blank lines" in result.suggestions[0].lower()

    def test_with_style_config(self):
        """Test formatting with style config."""
        style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        validator = DefaultFormattingValidator(FormattingConfig(style_config=style_config))
        # Valid text
        result = validator.validate("This is a test.")
        assert result.passed is True
        # Invalid text
        result = validator.validate("this is a test.")
        assert result.passed is False
        assert "sentence case" in result.message.lower()


class TestStyleRule:
    """Tests for the StyleRule class."""

    def test_rule_creation(self):
        """Test rule creation with default validator."""
        rule = StyleRule(
            name="test_rule",
            description="Test rule",
            config=None,
        )
        assert rule.name == "test_rule"
        assert rule.description == "Test rule"
        assert rule._validator is not None

    def test_rule_validation(self):
        """Test rule validation."""
        rule = StyleRule(
            name="test_rule",
            config=None,
            validator=DefaultStyleValidator(
                StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
            ),
        )
        # Valid text
        result = rule.model_validate("This is a test.")
        assert result.passed is True
        assert result.metadata["rule_id"] == "test_rule"
        # Invalid text
        result = rule.model_validate("this is a test.")
        assert result.passed is False
        assert "sentence case" in result.message.lower()
        assert result.metadata["rule_id"] == "test_rule"


class TestFormattingRule:
    """Tests for the FormattingRule class."""

    def test_rule_creation(self):
        """Test rule creation with default validator."""
        rule = FormattingRule(
            name="test_rule",
            description="Test rule",
            config=None,
        )
        assert rule.name == "test_rule"
        assert rule.description == "Test rule"
        assert rule._validator is not None

    def test_rule_validation(self):
        """Test rule validation."""
        style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        formatting_config = FormattingConfig(
            style_config=style_config,
            normalize_whitespace=True,
        )
        rule = FormattingRule(
            name="test_rule",
            config=None,
            validator=DefaultFormattingValidator(formatting_config),
        )
        # Valid text
        result = rule.model_validate("This is a test.")
        assert result.passed is True
        assert result.metadata["rule_id"] == "test_rule"
        # Invalid text
        result = rule.model_validate("this   is  a   test.")
        assert result.passed is False
        assert "sentence case" in result.message.lower()
        assert result.metadata["rule_id"] == "test_rule"


class TestFactoryFunctions:
    """Tests for the factory functions."""

    def test_create_style_validator(self):
        """Test create_style_validator function."""
        validator = create_style_validator(
            capitalization=CapitalizationStyle.SENTENCE_CASE,
            require_end_punctuation=True,
        )
        assert isinstance(validator, DefaultStyleValidator)
        assert validator.config.capitalization == CapitalizationStyle.SENTENCE_CASE
        assert validator.config.require_end_punctuation is True

    def test_create_style_rule(self):
        """Test create_style_rule function."""
        rule = create_style_rule(
            name="test_rule",
            capitalization=CapitalizationStyle.SENTENCE_CASE,
            require_end_punctuation=True,
        )
        assert isinstance(rule, StyleRule)
        assert rule.name == "test_rule"
        assert isinstance(rule._validator, DefaultStyleValidator)

    def test_create_formatting_validator(self):
        """Test create_formatting_validator function."""
        style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        validator = create_formatting_validator(
            style_config=style_config,
            normalize_whitespace=True,
        )
        assert isinstance(validator, DefaultFormattingValidator)
        assert validator.config.style_config == style_config
        assert validator.config.normalize_whitespace is True

    def test_create_formatting_rule(self):
        """Test create_formatting_rule function."""
        style_config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
        rule = create_formatting_rule(
            name="test_rule",
            style_config=style_config,
            normalize_whitespace=True,
        )
        assert isinstance(rule, FormattingRule)
        assert rule.name == "test_rule"
        assert isinstance(rule._validator, DefaultFormattingValidator)
