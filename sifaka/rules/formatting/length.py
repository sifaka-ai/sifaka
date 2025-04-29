"""
Length validation rules for text.

This module provides validators and rules for checking text length constraints.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

from sifaka.rules.base import Rule, RuleResult, RuleConfig


class LengthValidator:
    """Base class for text length validators."""

    def __init__(self, config: RuleConfig):
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
        params = self.config.params

        # Character length validation
        char_count = len(text)
        min_chars = params.get("min_chars")
        if min_chars is not None and char_count < min_chars:
            errors.append(f"Text is too short: {char_count} characters (minimum {min_chars})")

        max_chars = params.get("max_chars")
        if max_chars is not None and char_count > max_chars:
            errors.append(f"Text is too long: {char_count} characters (maximum {max_chars})")

        # Word count validation
        word_count = len(text.split())
        min_words = params.get("min_words")
        if min_words is not None and word_count < min_words:
            errors.append(f"Text has too few words: {word_count} words (minimum {min_words})")

        max_words = params.get("max_words")
        if max_words is not None and word_count > max_words:
            errors.append(f"Text has too many words: {word_count} words (maximum {max_words})")

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
    config = RuleConfig(
        params={
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_words": min_words,
            "max_words": max_words,
        }
    )
    validator = DefaultLengthValidator(config)

    # Use rule_id as name if provided, otherwise use "length_rule"
    name = rule_id if rule_id else "length_rule"

    return LengthRule(
        validator=validator,
        name=name,
        **kwargs,
    )
