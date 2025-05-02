"""
Style validation rules for text.

This module provides validators and rules for checking text styling constraints
such as capitalization, punctuation, and other formatting standards.

Usage Example:
    from sifaka.rules.formatting.style import create_style_rule, CapitalizationStyle

    # Create a style rule using the factory function
    rule = create_style_rule(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True,
        rule_id="sentence_style"
    )

    # Validate text
    result = rule.validate("This is a test.")
"""

import re
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from sifaka.rules.base import Rule, RuleResult, RuleConfig, BaseValidator


__all__ = [
    # Enums
    "CapitalizationStyle",
    # Config classes
    "StyleConfig",
    "FormattingConfig",
    # Validator classes
    "StyleValidator",
    "DefaultStyleValidator",
    "FormattingValidator",
    "DefaultFormattingValidator",
    # Rule classes
    "StyleRule",
    "FormattingRule",
    # Factory functions
    "create_style_validator",
    "create_style_rule",
    "create_formatting_validator",
    "create_formatting_rule",
    # Internal helpers
    "_CapitalizationAnalyzer",
    "_EndingAnalyzer",
    "_CharAnalyzer",
]


class CapitalizationStyle(Enum):
    """Enumeration of text capitalization styles."""

    SENTENCE_CASE = auto()  # First letter capitalized, rest lowercase
    TITLE_CASE = auto()  # Major Words Capitalized
    LOWERCASE = auto()  # all lowercase
    UPPERCASE = auto()  # ALL UPPERCASE
    CAPITALIZE_FIRST = auto()  # Only first letter capitalized


class StyleConfig(BaseModel):
    """Configuration for text style validation.

    Attributes:
        capitalization: Required capitalization style
        require_end_punctuation: Whether text must end with punctuation
        allowed_end_chars: List of allowed ending characters
        disallowed_chars: List of characters not allowed in the text
        strip_whitespace: Whether to strip whitespace before validation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capitalization: Optional[CapitalizationStyle] = Field(
        default=None,
        description="Required capitalization style",
    )
    require_end_punctuation: bool = Field(
        default=False,
        description="Whether text must end with punctuation",
    )
    allowed_end_chars: Optional[List[str]] = Field(
        default=None,
        description="List of allowed ending characters",
    )
    disallowed_chars: Optional[List[str]] = Field(
        default=None,
        description="List of characters not allowed in the text",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip whitespace before validation",
    )


class FormattingConfig(BaseModel):
    """Configuration for text formatting validation.

    Attributes:
        style_config: Configuration for style validation
        strip_whitespace: Whether to strip whitespace before validation
        normalize_whitespace: Whether to normalize consecutive whitespace
        remove_extra_lines: Whether to remove extra blank lines
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    style_config: Optional[StyleConfig] = Field(
        default=None,
        description="Configuration for style validation",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip whitespace before validation",
    )
    normalize_whitespace: bool = Field(
        default=False,
        description="Whether to normalize consecutive whitespace",
    )
    remove_extra_lines: bool = Field(
        default=False,
        description="Whether to remove extra blank lines",
    )


class StyleValidator(BaseValidator[str]):
    """Base class for text style validators."""

    def __init__(self, config: StyleConfig):
        """Initialize validator with a configuration.

        Args:
            config: Style validation configuration
        """
        super().__init__()
        self.config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text against style constraints.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        raise NotImplementedError("Subclasses must implement validate method")


class DefaultStyleValidator(StyleValidator):
    """Default style validator delegating logic to analyzers."""

    def __init__(self, config: StyleConfig):
        super().__init__(config)

        self._cap_analyzer = _CapitalizationAnalyzer(style=config.capitalization)
        self._end_analyzer = _EndingAnalyzer(
            require_end_punctuation=config.require_end_punctuation,
            allowed_end_chars=config.allowed_end_chars or [],
        )
        self._char_analyzer = _CharAnalyzer(disallowed=config.disallowed_chars or [])

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401
        """Validate *text* style by delegating to analyzers."""

        empty = self.handle_empty_text(text)
        if empty:
            return empty

        if self.config.strip_whitespace:
            text = text.strip()

        errors: List[str] = []

        if cap_err := self._cap_analyzer.analyze(text):
            errors.append(cap_err)

        if end_err := self._end_analyzer.analyze(text):
            errors.append(end_err)

        disallowed_found = self._char_analyzer.analyze(text)
        if disallowed_found:
            errors.extend([f"Disallowed character found: '{ch}'" for ch in disallowed_found])

        return RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Style validation successful",
            metadata={"errors": errors},
        )


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
            **kwargs: Additional validation context

        Returns:
            RuleResult containing validation results
        """
        result = self.validator.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self.id)


def create_style_validator(
    capitalization: Optional[CapitalizationStyle] = None,
    require_end_punctuation: bool = False,
    allowed_end_chars: Optional[List[str]] = None,
    disallowed_chars: Optional[List[str]] = None,
    strip_whitespace: bool = True,
    **kwargs,
) -> StyleValidator:
    """Create a style validator with the specified constraints.

    This factory function creates a configured StyleValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        capitalization: Required capitalization style
        require_end_punctuation: Whether text must end with punctuation
        allowed_end_chars: List of allowed ending characters
        disallowed_chars: List of characters not allowed in the text
        strip_whitespace: Whether to strip whitespace before validation
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured StyleValidator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    config = StyleConfig(
        capitalization=capitalization,
        require_end_punctuation=require_end_punctuation,
        allowed_end_chars=allowed_end_chars,
        disallowed_chars=disallowed_chars,
        strip_whitespace=strip_whitespace,
        **rule_config_params,
    )

    return DefaultStyleValidator(config)


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

    This factory function creates a configured StyleRule instance.
    It uses create_style_validator internally to create the validator.

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
    # Create validator using the validator factory
    validator = create_style_validator(
        capitalization=capitalization,
        require_end_punctuation=require_end_punctuation,
        allowed_end_chars=allowed_end_chars,
        disallowed_chars=disallowed_chars,
        strip_whitespace=strip_whitespace,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    return StyleRule(
        validator=validator,
        id=rule_id or "style",
        **rule_kwargs,
    )


class FormattingValidator(BaseValidator[str]):
    """Base class for text formatting validators."""

    def __init__(self, config: FormattingConfig):
        """Initialize validator with a configuration.

        Args:
            config: Formatting validation configuration
        """
        super().__init__()
        self.config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text against formatting constraints.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        raise NotImplementedError("Subclasses must implement validate method")


class DefaultFormattingValidator(FormattingValidator):
    """Default implementation of text formatting validator."""

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text against formatting constraints.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        errors = []
        original_text = text

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
            style_result = style_validator.validate(text, **kwargs)
            if not style_result.passed:
                errors.append(style_result.message)

        return RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Formatting validation successful",
            metadata={
                "original_length": len(original_text),
                "formatted_length": len(text),
                "strip_whitespace": self.config.strip_whitespace,
                "normalize_whitespace": self.config.normalize_whitespace,
                "remove_extra_lines": self.config.remove_extra_lines,
                "errors": errors,
            },
        )


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
            **kwargs: Additional validation context

        Returns:
            RuleResult containing validation results
        """
        result = self.validator.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self.id)


def create_formatting_validator(
    style_config: Optional[StyleConfig] = None,
    strip_whitespace: bool = True,
    normalize_whitespace: bool = False,
    remove_extra_lines: bool = False,
    **kwargs,
) -> FormattingValidator:
    """Create a formatting validator with the specified constraints.

    This factory function creates a configured FormattingValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        style_config: Configuration for style validation
        strip_whitespace: Whether to strip whitespace before validation
        normalize_whitespace: Whether to normalize consecutive whitespace
        remove_extra_lines: Whether to remove extra blank lines
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured FormattingValidator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    config = FormattingConfig(
        style_config=style_config,
        strip_whitespace=strip_whitespace,
        normalize_whitespace=normalize_whitespace,
        remove_extra_lines=remove_extra_lines,
        **rule_config_params,
    )

    return DefaultFormattingValidator(config)


def create_formatting_rule(
    style_config: Optional[StyleConfig] = None,
    strip_whitespace: bool = True,
    normalize_whitespace: bool = False,
    remove_extra_lines: bool = False,
    rule_id: Optional[str] = None,
    **kwargs,
) -> FormattingRule:
    """Create a formatting validation rule with the specified constraints.

    This factory function creates a configured FormattingRule instance.
    It uses create_formatting_validator internally to create the validator.

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
    # Create validator using the validator factory
    validator = create_formatting_validator(
        style_config=style_config,
        strip_whitespace=strip_whitespace,
        normalize_whitespace=normalize_whitespace,
        remove_extra_lines=remove_extra_lines,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    return FormattingRule(
        validator=validator,
        id=rule_id or "formatting",
        **rule_kwargs,
    )


# ---------------------------------------------------------------------------
# Analyzer helpers (Single Responsibility)
# ---------------------------------------------------------------------------


class _CapitalizationAnalyzer(BaseModel):
    """Validate capitalization according to the configured style."""

    style: Optional["CapitalizationStyle"] = None  # noqa: F821 forward ref

    def analyze(self, text: str) -> Optional[str]:
        if self.style is None or not text:
            return None

        # Sentence case
        if self.style == CapitalizationStyle.SENTENCE_CASE:
            if not (text[0].isupper() and text[1:].islower()):
                return "Text should be in sentence case"

        # Title case (simple heuristic)
        elif self.style == CapitalizationStyle.TITLE_CASE:
            if any(word and word[0].islower() for word in text.split()):
                return "Text should be in title case"

        # Lowercase
        elif self.style == CapitalizationStyle.LOWERCASE:
            if text.lower() != text:
                return "Text should be all lowercase"

        # Uppercase
        elif self.style == CapitalizationStyle.UPPERCASE:
            if text.upper() != text:
                return "Text should be all uppercase"

        # Capitalize first
        elif self.style == CapitalizationStyle.CAPITALIZE_FIRST:
            if not (text[0].isupper() and text[1:] == text[1:]):
                # second condition basically always true; we accept any remainder
                return "Only the first character should be capitalized"

        return None


class _EndingAnalyzer(BaseModel):
    """Check ending punctuation requirements."""

    require_end_punctuation: bool = False
    allowed_end_chars: List[str] = Field(default_factory=list)

    def analyze(self, text: str) -> Optional[str]:
        if not text:
            return None

        end_char = text[-1]

        if self.require_end_punctuation and end_char not in ".!?" and end_char not in self.allowed_end_chars:
            return "Text must end with punctuation"

        if self.allowed_end_chars and end_char not in self.allowed_end_chars:
            return f"Text must end with one of {self.allowed_end_chars}"

        return None


class _CharAnalyzer(BaseModel):
    """Detect presence of disallowed characters."""

    disallowed: List[str] = Field(default_factory=list)

    def analyze(self, text: str) -> List[str]:
        return [ch for ch in self.disallowed if ch in text]
