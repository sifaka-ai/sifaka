"""
Base classes and protocols for domain-specific validation rules.
"""

from typing import Any, Dict, Protocol, runtime_checkable

from sifaka.rules.base import BaseValidator, RuleResult


@runtime_checkable
class DomainValidator(Protocol):
    """Protocol for domain-specific validation."""

    def validate(self, text: str) -> RuleResult: ...


class BaseDomainValidator(BaseValidator[str]):
    """Base class for domain-specific validators."""

    def __init__(self, config: Any) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> Any:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text based on domain rules.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
