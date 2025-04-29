"""
Length validation rules for text.

This module provides validators and rules for checking text length constraints.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

from sifaka.rules.base import Rule, RuleResult


@dataclass
class LengthConfig:
    """Configuration for text length validation.

    Attributes:
        min_chars: Minimum number of characters allowed (inclusive)
        max_chars: Maximum number of characters allowed (inclusive)
        min_words: Minimum number of words allowed (inclusive)
        max_words: Maximum number of words allowed (inclusive)
    """

    min_chars: Optional[int] = None
    max_chars: Optional[int] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None


class LengthValidator:
    """Base class for text length validators."""

    def __init__(self, config: LengthConfig):
        """Initialize validator with a configuration.

        Args:
            config: Length validation configuration
        """
        self.config = config

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against length constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        raise NotImplementedError("Subclasses must implement validate method")


class DefaultLengthValidator(LengthValidator):
    """Default implementation of text length validator."""

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against length constraints.

        Args:
            text: The text to validate

        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of error messages if validation failed
        """
        errors = []

        # Character length validation
        char_count = len(text)
        if self.config.min_chars is not None and char_count < self.config.min_chars:
            errors.append(
                f"Text is too short: {char_count} characters (minimum {self.config.min_chars})"
            )

        if self.config.max_chars is not None and char_count > self.config.max_chars:
            errors.append(
                f"Text is too long: {char_count} characters (maximum {self.config.max_chars})"
            )

        # Word count validation
        word_count = len(text.split())
        if self.config.min_words is not None and word_count < self.config.min_words:
            errors.append(
                f"Text has too few words: {word_count} words (minimum {self.config.min_words})"
            )

        if self.config.max_words is not None and word_count > self.config.max_words:
            errors.append(
                f"Text has too many words: {word_count} words (maximum {self.config.max_words})"
            )

        return not errors, errors


class LengthRuleValidator:
    """Validator adapter that implements RuleValidator protocol for LengthValidator."""

    def __init__(self, validator: LengthValidator):
        """Initialize with a LengthValidator."""
        self.validator = validator

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate the output using the wrapped validator."""
        is_valid, errors = self.validator.validate(output)
        return RuleResult(
            passed=is_valid,
            message=errors[0] if errors else "Text length validation successful",
            metadata={
                "char_count": len(output),
                "word_count": len(output.split()),
                "errors": errors,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can validate the output."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type:
        """Get the type this validator can validate."""
        return str


class LengthRule(Rule):
    """Rule for validating text length constraints."""

    def __init__(
        self,
        validator: LengthValidator,
        name: str = "length_rule",
        description: str = "Validates text length",
        config: Optional[Dict] = None,
        **kwargs,
    ):
        """Initialize the length rule.

        Args:
            validator: The validator to use for length validation
            name: The name of the rule
            description: Description of the rule
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        self._length_validator = validator
        self._rule_id = kwargs.pop("rule_id", name)  # Extract rule_id if present, default to name
        super().__init__(name=name, description=description, config=config, **kwargs)

    def _create_default_validator(self) -> LengthRuleValidator:
        """Create a default validator adapter for this rule."""
        return LengthRuleValidator(self._length_validator)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Evaluate text against length constraints.

        Args:
            text: The text to evaluate

        Returns:
            RuleResult containing validation results
        """
        is_valid, errors = self._length_validator.validate(text)
        return RuleResult(
            passed=is_valid,
            message=errors[0] if errors else "Text length validation successful",
            metadata={
                "char_count": len(text),
                "word_count": len(text.split()),
                "rule_id": self._rule_id,
            },
        )


def create_length_rule(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    rule_id: Optional[str] = None,
    **kwargs,
) -> LengthRule:
    """Create a length validation rule with the specified constraints.

    Args:
        min_chars: Minimum number of characters allowed
        max_chars: Maximum number of characters allowed
        min_words: Minimum number of words allowed
        max_words: Maximum number of words allowed
        rule_id: Identifier for the rule
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured LengthRule
    """
    config = LengthConfig(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        max_words=max_words,
    )
    validator = DefaultLengthValidator(config)

    # Use rule_id as name if provided, otherwise use "length_rule"
    name = rule_id if rule_id else "length_rule"

    return LengthRule(
        validator=validator,
        name=name,
        **kwargs,
    )
