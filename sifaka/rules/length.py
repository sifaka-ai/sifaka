"""
Length validation rules for Sifaka.

This module provides rules for validating text length, supporting both character and word count validation.
"""

from typing import Dict, Any, Protocol, runtime_checkable, Final, Optional, Type
from dataclasses import dataclass

from sifaka.rules.base import Rule, RuleResult


@dataclass(frozen=True)
class LengthConfig:
    """Immutable configuration for length validation."""

    min_length: int = 50
    max_length: int | None = 5000
    exact_length: int | None = None
    unit: str = "characters"  # "characters" or "words"
    cache_size: int = 10
    priority: int = 2
    cost: float = 1.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.unit not in ["characters", "words"]:
            raise ValueError("unit must be either 'characters' or 'words'")

        if self.exact_length is not None:
            if self.exact_length < 0:
                raise ValueError("exact_length must be non-negative")
            if self.min_length != 50 or self.max_length != 5000:
                raise ValueError("exact_length cannot be used with min_length or max_length")
        else:
            if self.min_length < 0:
                raise ValueError("min_length must be non-negative")
            if self.max_length is not None:
                if self.max_length < 0:
                    raise ValueError("max_length must be non-negative")
                if self.min_length > self.max_length:
                    raise ValueError("min_length cannot be greater than max_length")


@runtime_checkable
class LengthValidator(Protocol):
    """Protocol for length validation components."""

    @property
    def config(self) -> LengthConfig: ...

    def validate(self, text: str) -> RuleResult: ...


class DefaultLengthValidator:
    """Default implementation of length validation."""

    def __init__(self, config: LengthConfig) -> None:
        """Initialize the validator with configuration."""
        self._config = config

    @property
    def config(self) -> LengthConfig:
        """Get the validator configuration."""
        return self._config

    def _get_length(self, text: str) -> int:
        """Get the length of the text in the specified unit."""
        if self.config.unit == "words":
            return len(text.split())
        return len(text)

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text length is within acceptable bounds.

        Args:
            text: The text to validate

        Returns:
            RuleResult with validation results
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        length = self._get_length(text)
        metadata = {
            "length": length,
            "min_length": self.config.min_length,
            "max_length": self.config.max_length,
            "exact_length": self.config.exact_length,
            "unit": self.config.unit,
        }

        # Handle empty or whitespace-only text
        if not text.strip():
            return RuleResult(
                passed=False,
                message=f"Empty or whitespace-only text (0 {self.config.unit})",
                metadata=metadata,
            )

        # Check exact length if specified
        if self.config.exact_length is not None:
            if length != self.config.exact_length:
                return RuleResult(
                    passed=False,
                    message=f"Text {self.config.unit} count {length} does not match required count of {self.config.exact_length}",
                    metadata=metadata,
                )
            return RuleResult(
                passed=True,
                message=f"Text {self.config.unit} count matches required count of {self.config.exact_length}",
                metadata=metadata,
            )

        # Check length bounds
        issues = []
        if length < self.config.min_length:
            issues.append(f"below minimum of {self.config.min_length}")
        if self.config.max_length is not None and length > self.config.max_length:
            issues.append(f"exceeds maximum of {self.config.max_length}")

        if issues:
            return RuleResult(
                passed=False,
                message=f"Text {self.config.unit} count {length} {' and '.join(issues)}",
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message=f"Text {self.config.unit} count {length} meets requirements",
            metadata=metadata,
        )

    def can_validate(self, text: Any) -> bool:
        """Check if the validator can handle the input."""
        return isinstance(text, str)

    @property
    def validation_type(self) -> Type[str]:
        """Get the type of input this validator can handle."""
        return str


class LengthRule(Rule):
    """Rule that checks if the text length falls within specified bounds."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[LengthValidator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with length constraints.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Custom length validator implementation
            config: Length validation configuration dictionary
        """
        # Create the config object first
        length_config = LengthConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultLengthValidator(length_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, text: str) -> RuleResult:
        """
        Validate that the text length is within acceptable bounds.

        Args:
            text: The text to validate

        Returns:
            RuleResult with validation results
        """
        return self._validator.validate(text)


def create_length_rule(
    name: str = "length_rule",
    description: str = "Validates text length",
    config: Optional[Dict[str, Any]] = None,
) -> LengthRule:
    """
    Factory function to create a length rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Length validation configuration dictionary

    Returns:
        Configured LengthRule instance
    """
    length_config = LengthConfig(**(config or {}))
    return LengthRule(
        name=name,
        description=description,
        config=length_config,
    )


# Export public classes and functions
__all__ = [
    "LengthRule",
    "LengthConfig",
    "LengthValidator",
    "DefaultLengthValidator",
    "create_length_rule",
]
