"""
Base classes and protocols for factual validation.

This module provides the base classes and protocols for factual validation in Sifaka.
It defines the core interfaces and abstract classes that all factual validators must implement.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.factual.base import BaseFactualValidator

    class MyFactualValidator(BaseFactualValidator):
        def validate(self, text: str) -> RuleResult:
            # Implement validation logic here
            pass
"""

from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)


class FactualValidator(Protocol):
    """Protocol for factual validators."""

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for factual accuracy.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        ...


class BaseFactualValidator(BaseValidator[str]):
    """Base class for factual validators."""

    def __init__(self, config: Any) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        self._config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate the given text for factual accuracy.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult: The result of the validation
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        raise NotImplementedError("Subclasses must implement validate()")

    @property
    def validation_type(self) -> type[str]:
        """Get the type this validator can validate."""
        return str


class FactualConfig(BaseModel):
    """Base configuration for factual validation."""

    model_config = ConfigDict(frozen=True)

    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )
