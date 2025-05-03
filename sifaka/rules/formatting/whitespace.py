"""
Whitespace validation rules for text.

This module provides validators and rules for checking text whitespace constraints
such as leading/trailing whitespace, spacing between words, and newline formatting.
"""

import re
from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

# Import the original RuleResult for type hints
from sifaka.rules.base import RuleResult as BaseRuleResult


class RuleResult:
    """Custom RuleResult class for whitespace validation."""

    def __init__(self, passed: bool, rule_id: str, errors: List[str]):
        """Initialize the rule result.

        Args:
            passed: Whether the validation passed
            rule_id: The ID of the rule
            errors: List of error messages
        """
        self.passed = passed
        self.rule_id = rule_id
        self.errors = errors
        self.message = "Whitespace validation " + ("passed" if passed else "failed")


class WhitespaceConfig(BaseModel):
    """Configuration for text whitespace validation.

    Attributes:
        allow_leading_whitespace: Whether to allow whitespace at the beginning of text
        allow_trailing_whitespace: Whether to allow whitespace at the end of text
        allow_multiple_spaces: Whether to allow multiple consecutive spaces
        allow_tabs: Whether to allow tab characters
        allow_newlines: Whether to allow newline characters
        max_newlines: Maximum number of consecutive newlines allowed
        normalize_whitespace: Whether to normalize whitespace during validation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow_leading_whitespace: bool = Field(
        default=False,
        description="Whether to allow whitespace at the beginning of text",
    )
    allow_trailing_whitespace: bool = Field(
        default=False,
        description="Whether to allow whitespace at the end of text",
    )
    allow_multiple_spaces: bool = Field(
        default=False,
        description="Whether to allow multiple consecutive spaces",
    )
    allow_tabs: bool = Field(
        default=False,
        description="Whether to allow tab characters",
    )
    allow_newlines: bool = Field(
        default=True,
        description="Whether to allow newline characters",
    )
    max_newlines: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of consecutive newlines allowed",
    )
    normalize_whitespace: bool = Field(
        default=False,
        description="Whether to normalize whitespace during validation",
    )

    @field_validator("max_newlines")
    @classmethod
    def validate_max_newlines(cls, v: Optional[int], info) -> Optional[int]:
        """Validate that max_newlines is only set if allow_newlines is True."""
        # In Pydantic v2, ValidationInfo object is passed instead of a dict
        # We need to access the data attribute to get the values
        allow_newlines = info.data.get("allow_newlines", True)
        if v is not None and not allow_newlines:
            raise ValueError("max_newlines can only be set if allow_newlines is True")
        return v


class WhitespaceValidator:
    """Base class for text whitespace validators."""

    def __init__(self, config: WhitespaceConfig):
        """Initialize validator with a configuration.

        Args:
            config: Whitespace validation configuration
        """
        self.config = config

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against whitespace constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        raise NotImplementedError("Subclasses must implement validate method")


class DefaultWhitespaceValidator(WhitespaceValidator):
    """Default implementation of text whitespace validator."""

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against whitespace constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        errors = []

        # Leading whitespace validation
        if not self.config.allow_leading_whitespace and text and text[0].isspace():
            errors.append("Text contains leading whitespace")

        # Trailing whitespace validation
        if not self.config.allow_trailing_whitespace and text and text[-1].isspace():
            errors.append("Text contains trailing whitespace")

        # Multiple spaces validation
        if not self.config.allow_multiple_spaces and "  " in text:
            errors.append("Text contains multiple consecutive spaces")

        # Tab character validation
        if not self.config.allow_tabs and "\t" in text:
            errors.append("Text contains tab characters")

        # Newline validation
        if not self.config.allow_newlines and "\n" in text:
            errors.append("Text contains newline characters")
        elif self.config.max_newlines is not None:
            newline_errors = self._validate_max_newlines(text)
            errors.extend(newline_errors)

        return not errors, errors

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: The text to normalize

        Returns:
            Text with normalized whitespace
        """
        # Replace tabs with spaces
        if not self.config.allow_tabs:
            text = text.replace("\t", " ")

        # Replace multiple spaces with single space
        if not self.config.allow_multiple_spaces:
            text = re.sub(r" +", " ", text)

        # Remove leading whitespace
        if not self.config.allow_leading_whitespace:
            text = text.lstrip()

        # Remove trailing whitespace
        if not self.config.allow_trailing_whitespace:
            text = text.rstrip()

        return text

    def _validate_max_newlines(self, text: str) -> List[str]:
        """Validate maximum consecutive newlines.

        Args:
            text: The text to validate

        Returns:
            List of error messages if validation failed
        """
        if not text or self.config.max_newlines is None:
            return []

        # Find sequences of newlines
        newline_sequences = re.findall(r"\n+", text)
        max_found = max([len(seq) for seq in newline_sequences]) if newline_sequences else 0

        if max_found > self.config.max_newlines:
            return [
                f"Text contains {max_found} consecutive newlines, maximum allowed is {self.config.max_newlines}"
            ]

        return []


class WhitespaceRule:
    """Rule for validating text whitespace constraints."""

    def __init__(self, validator: WhitespaceValidator, config: Optional[Dict] = None, **kwargs):
        """Initialize the whitespace rule.

        Args:
            validator: The validator to use for whitespace validation
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        self.validator = validator
        self.config = config
        self.id = kwargs.get("id", "rule")
        self.name = kwargs.get("name", "Whitespace Rule")
        self.description = kwargs.get("description", "Validates whitespace constraints")

    def validate(self, text: str) -> RuleResult:
        """Evaluate text against whitespace constraints.

        Args:
            text: The text to evaluate

        Returns:
            RuleResult containing validation results
        """
        # Use the validator to validate the text
        is_valid, errors = self.validator.validate(text)

        # Create and return the result
        return RuleResult(
            passed=is_valid,
            rule_id=self.id,
            errors=errors,
        )


def create_whitespace_rule(
    allow_leading_whitespace: bool = False,
    allow_trailing_whitespace: bool = False,
    allow_multiple_spaces: bool = False,
    allow_tabs: bool = False,
    allow_newlines: bool = True,
    max_newlines: Optional[int] = None,
    normalize_whitespace: bool = False,
    rule_id: Optional[str] = None,
    **kwargs,
) -> WhitespaceRule:
    """Create a whitespace validation rule with the specified constraints.

    Args:
        allow_leading_whitespace: Whether to allow whitespace at the beginning of text
        allow_trailing_whitespace: Whether to allow whitespace at the end of text
        allow_multiple_spaces: Whether to allow multiple consecutive spaces
        allow_tabs: Whether to allow tab characters
        allow_newlines: Whether to allow newline characters
        max_newlines: Maximum number of consecutive newlines allowed
        normalize_whitespace: Whether to normalize whitespace during validation
        rule_id: Identifier for the rule
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured WhitespaceRule
    """
    # Create the whitespace configuration
    config = WhitespaceConfig(
        allow_leading_whitespace=allow_leading_whitespace,
        allow_trailing_whitespace=allow_trailing_whitespace,
        allow_multiple_spaces=allow_multiple_spaces,
        allow_tabs=allow_tabs,
        allow_newlines=allow_newlines,
        max_newlines=max_newlines,
        normalize_whitespace=normalize_whitespace,
    )

    # Create the validator
    validator = DefaultWhitespaceValidator(config)

    # Create and return the rule
    return WhitespaceRule(
        validator=validator,
        id=rule_id or "whitespace",
        **kwargs,
    )
