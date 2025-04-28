"""
Length validation rules for Sifaka.

This module provides rules for validating text length, supporting both character and word count validation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult


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


class DefaultLengthValidator(BaseValidator[str]):
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

    def validate(self, text: str, **kwargs) -> RuleResult:
        """
        Validate that the text length is within acceptable bounds.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

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


class LengthRule(Rule[str, RuleResult, DefaultLengthValidator, Any]):
    """Rule that checks if the text length falls within specified bounds."""

    def __init__(
        self,
        name: str = "length_rule",
        description: str = "Checks if text length is within bounds",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultLengthValidator] = None,
    ) -> None:
        """
        Initialize the rule with length constraints.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Custom length validator implementation
        """
        # Store length parameters for creating the default validator
        self._length_params = {}
        if config and config.params:
            self._length_params = config.params

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultLengthValidator:
        """Create a default validator from config."""
        length_config = LengthConfig(**self._length_params)
        return DefaultLengthValidator(length_config)


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
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return LengthRule(
        name=name,
        description=description,
        config=rule_config,
    )


# Export public classes and functions
__all__ = [
    "LengthRule",
    "LengthConfig",
    "LengthValidator",
    "DefaultLengthValidator",
    "create_length_rule",
]
