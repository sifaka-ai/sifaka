"""
Length validation rules for text.

This module provides validators and rules for checking text length constraints.

Usage Example:
    from sifaka.rules.formatting.length import create_length_rule

    # Create a length rule using the factory function
    rule = create_length_rule(
        min_chars=10,
        max_chars=100,
        min_words=2,
        max_words=20,
        rule_id="length_constraint"
    )

    # Validate text
    result = rule.validate("This is a test.")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

from sifaka.rules.base import Rule, RuleResult, RuleConfig, BaseValidator


__all__ = [
    # Config classes
    "LengthConfig",
    # Validator classes
    "LengthValidator",
    "DefaultLengthValidator",
    "LengthRuleValidator",
    # Rule classes
    "LengthRule",
    # Factory functions
    "create_length_validator",
    "create_length_rule",
]


@dataclass(frozen=True)
class LengthConfig(RuleConfig):
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


class LengthValidator(BaseValidator[str]):
    """Base class for text length validators."""

    def __init__(self, config: LengthConfig):
        """Initialize validator with a configuration.

        Args:
            config: Length validation configuration
        """
        super().__init__()
        self.config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text against length constraints.

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


class DefaultLengthValidator(LengthValidator):
    """Default implementation of text length validator."""

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text against length constraints.

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

        return RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Text length validation successful",
            metadata={
                "char_count": char_count,
                "word_count": word_count,
                "errors": errors,
            },
        )


class LengthRuleValidator:
    """Validator adapter that implements RuleValidator protocol for LengthValidator."""

    def __init__(self, validator: LengthValidator):
        """Initialize with a LengthValidator."""
        self.validator = validator

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate the output using the wrapped validator."""
        return self.validator.validate(output, **kwargs)

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
            **kwargs: Additional validation context

        Returns:
            RuleResult containing validation results
        """
        result = self._length_validator.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._rule_id)


def create_length_validator(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    **kwargs,
) -> LengthValidator:
    """Create a length validator with the specified constraints.

    This factory function creates a configured LengthValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        min_chars: Minimum number of characters allowed
        max_chars: Maximum number of characters allowed
        min_words: Minimum number of words allowed
        max_words: Maximum number of words allowed
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured LengthValidator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    config = LengthConfig(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        max_words=max_words,
        **rule_config_params,
    )

    return DefaultLengthValidator(config)


def create_length_rule(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    rule_id: Optional[str] = None,
    **kwargs,
) -> LengthRule:
    """Create a length validation rule with the specified constraints.

    This factory function creates a configured LengthRule instance.
    It uses create_length_validator internally to create the validator.

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
    # Create validator using the validator factory
    validator = create_length_validator(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        max_words=max_words,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Use rule_id as name if provided, otherwise use "length_rule"
    name = rule_id if rule_id else "length_rule"

    return LengthRule(
        validator=validator,
        name=name,
        **rule_kwargs,
    )
