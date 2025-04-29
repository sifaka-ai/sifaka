"""
Style validation rules for text.

This module provides validators and rules for checking text styling constraints
such as capitalization, punctuation, and other formatting standards.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

from sifaka.rules.base import Rule, RuleResult


class CapitalizationStyle(Enum):
    """Enumeration of text capitalization styles."""

    SENTENCE_CASE = auto()  # First letter capitalized, rest lowercase
    TITLE_CASE = auto()  # Major Words Capitalized
    LOWERCASE = auto()  # all lowercase
    UPPERCASE = auto()  # ALL UPPERCASE
    CAPITALIZE_FIRST = auto()  # Only first letter capitalized


@dataclass
class StyleConfig:
    """Configuration for text style validation.

    Attributes:
        capitalization: Required capitalization style
        require_end_punctuation: Whether text must end with punctuation
        allowed_end_chars: List of allowed ending characters
        disallowed_chars: List of characters not allowed in the text
        strip_whitespace: Whether to strip whitespace before validation
    """

    capitalization: Optional[CapitalizationStyle] = None
    require_end_punctuation: bool = False
    allowed_end_chars: Optional[List[str]] = None
    disallowed_chars: Optional[List[str]] = None
    strip_whitespace: bool = True


@dataclass
class FormattingConfig:
    """Configuration for text formatting validation.

    Attributes:
        style_config: Configuration for style validation
        strip_whitespace: Whether to strip whitespace before validation
        normalize_whitespace: Whether to normalize consecutive whitespace
        remove_extra_lines: Whether to remove extra blank lines
    """

    style_config: Optional[StyleConfig] = None
    strip_whitespace: bool = True
    normalize_whitespace: bool = False
    remove_extra_lines: bool = False


class StyleValidator:
    """Base class for text style validators."""

    def __init__(self, config: StyleConfig):
        """Initialize validator with a configuration.

        Args:
            config: Style validation configuration
        """
        self.config = config

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against style constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        raise NotImplementedError("Subclasses must implement validate method")


class DefaultStyleValidator(StyleValidator):
    """Default implementation of text style validator."""

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against style constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        if self.config.strip_whitespace:
            text = text.strip()

        errors = []

        # Capitalization validation
        if self.config.capitalization:
            cap_valid, cap_error = self._validate_capitalization(text)
            if not cap_valid:
                errors.append(cap_error)

        # End punctuation validation
        if self.config.require_end_punctuation or self.config.allowed_end_chars:
            end_valid, end_error = self._validate_ending(text)
            if not end_valid:
                errors.append(end_error)

        # Disallowed characters validation
        if self.config.disallowed_chars:
            char_valid, char_errors = self._validate_disallowed_chars(text)
            if not char_valid:
                errors.extend(char_errors)

        return not errors, errors

    def _validate_capitalization(self, text: str) -> Tuple[bool, str]:
        """Validate text capitalization style.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - Error message if validation failed
        """
        if not text:
            return True, ""

        style = self.config.capitalization
        if style == CapitalizationStyle.SENTENCE_CASE:
            is_valid = text[0].isupper() and text[1:].islower()
            return is_valid, "Text should be in sentence case" if not is_valid else ""

        elif style == CapitalizationStyle.TITLE_CASE:
            words = text.split()
            # Simple title case check - each major word should be capitalized
            is_valid = all(w[0].isupper() for w in words if len(w) > 3 or words.index(w) == 0)
            return is_valid, "Text should be in title case" if not is_valid else ""

        elif style == CapitalizationStyle.LOWERCASE:
            is_valid = text.islower()
            return is_valid, "Text should be lowercase" if not is_valid else ""

        elif style == CapitalizationStyle.UPPERCASE:
            is_valid = text.isupper()
            return is_valid, "Text should be uppercase" if not is_valid else ""

        elif style == CapitalizationStyle.CAPITALIZE_FIRST:
            is_valid = text[0].isupper() if text else True
            return is_valid, "First letter should be capitalized" if not is_valid else ""

        return True, ""

    def _validate_ending(self, text: str) -> Tuple[bool, str]:
        """Validate text ending.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - Error message if validation failed
        """
        if not text:
            return True, ""

        if self.config.require_end_punctuation:
            punct_pattern = r"[.!?]$"
            if not re.search(punct_pattern, text):
                return False, "Text must end with punctuation (., !, or ?)"

        if self.config.allowed_end_chars:
            last_char = text[-1] if text else ""
            if last_char not in self.config.allowed_end_chars:
                allowed = ", ".join(self.config.allowed_end_chars)
                return False, f"Text must end with one of these characters: {allowed}"

        return True, ""

    def _validate_disallowed_chars(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text doesn't contain disallowed characters.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages for each disallowed character found
        """
        if not text or not self.config.disallowed_chars:
            return True, []

        errors = []
        for char in self.config.disallowed_chars:
            if char in text:
                errors.append(f"Text contains disallowed character: '{char}'")

        return not errors, errors


class StyleRule(Rule):
    """Rule for validating text style constraints."""

    def __init__(self, validator: StyleValidator, config: Optional[Dict] = None, **kwargs):
        """Initialize the style rule.

        Args:
            validator: The validator to use for style validation
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(config=config, **kwargs)
        self.validator = validator

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Evaluate text against style constraints.

        Args:
            text: The text to evaluate

        Returns:
            RuleResult containing validation results
        """
        is_valid, errors = self.validator.validate(text)
        return RuleResult(
            passed=is_valid,
            rule_id=self.id,
            errors=errors,
        )


def create_style_rule(
    capitalization: Optional[CapitalizationStyle] = None,
    require_end_punctuation: bool = False,
    allowed_end_chars: Optional[List[str]] = None,
    disallowed_chars: Optional[List[str]] = None,
    strip_whitespace: bool = True,
    rule_id: Optional[str] = None,
    **kwargs,
) -> StyleRule:
    """Create a style validation rule with the specified constraints.

    Args:
        capitalization: Required capitalization style
        require_end_punctuation: Whether text must end with punctuation
        allowed_end_chars: List of allowed ending characters
        disallowed_chars: List of characters not allowed in the text
        strip_whitespace: Whether to strip whitespace before validation
        rule_id: Identifier for the rule
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured StyleRule
    """
    config = StyleConfig(
        capitalization=capitalization,
        require_end_punctuation=require_end_punctuation,
        allowed_end_chars=allowed_end_chars,
        disallowed_chars=disallowed_chars,
        strip_whitespace=strip_whitespace,
    )
    validator = DefaultStyleValidator(config)

    return StyleRule(
        validator=validator,
        id=rule_id or "style",
        **kwargs,
    )


class FormattingValidator:
    """Base class for text formatting validators."""

    def __init__(self, config: FormattingConfig):
        """Initialize validator with a configuration.

        Args:
            config: Formatting validation configuration
        """
        self.config = config

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against formatting constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        raise NotImplementedError("Subclasses must implement validate method")


class DefaultFormattingValidator(FormattingValidator):
    """Default implementation of text formatting validator."""

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against formatting constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        if not text:
            return True, []

        errors = []

        # Apply transformations if configured
        if self.config.strip_whitespace:
            text = text.strip()

        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text)

        if self.config.remove_extra_lines:
            text = re.sub(r"\n{3,}", "\n\n", text)

        # Validate against style config if provided
        if self.config.style_config:
            style_validator = DefaultStyleValidator(self.config.style_config)
            is_valid, style_errors = style_validator.validate(text)
            if not is_valid:
                errors.extend(style_errors)

        return not errors, errors


class FormattingRule(Rule):
    """Rule for validating text formatting constraints."""

    def __init__(self, validator: FormattingValidator, config: Optional[Dict] = None, **kwargs):
        """Initialize the formatting rule.

        Args:
            validator: The validator to use for formatting validation
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(config=config, **kwargs)
        self.validator = validator

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Evaluate text against formatting constraints.

        Args:
            text: The text to evaluate

        Returns:
            RuleResult containing validation results
        """
        is_valid, errors = self.validator.validate(text)
        return RuleResult(
            passed=is_valid,
            rule_id=self.id,
            errors=errors,
        )


def create_formatting_rule(
    style_config: Optional[StyleConfig] = None,
    strip_whitespace: bool = True,
    normalize_whitespace: bool = False,
    remove_extra_lines: bool = False,
    rule_id: Optional[str] = None,
    **kwargs,
) -> FormattingRule:
    """Create a formatting validation rule with the specified constraints.

    Args:
        style_config: Configuration for style validation
        strip_whitespace: Whether to strip whitespace before validation
        normalize_whitespace: Whether to normalize consecutive whitespace
        remove_extra_lines: Whether to remove extra blank lines
        rule_id: Identifier for the rule
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured FormattingRule
    """
    config = FormattingConfig(
        style_config=style_config,
        strip_whitespace=strip_whitespace,
        normalize_whitespace=normalize_whitespace,
        remove_extra_lines=remove_extra_lines,
    )
    validator = DefaultFormattingValidator(config)

    return FormattingRule(
        validator=validator,
        id=rule_id or "formatting",
        **kwargs,
    )
