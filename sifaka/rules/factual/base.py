"""
Base classes and protocols for factual validation.
"""

from typing import Any, Dict, Protocol, runtime_checkable

from sifaka.rules.base import BaseValidator, RuleResult


@runtime_checkable
class FactualValidator(Protocol):
    """Protocol for factual validation."""

    def validate(self, text: str) -> RuleResult: ...


class BaseFactualValidator(BaseValidator[str]):
    """Base class for factual validators."""

    def __init__(self, config: Any) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> Any:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text based on factual rules.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
