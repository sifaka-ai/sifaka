"""
Base classes and protocols for adapter-based rules.

This module provides the foundation for adapting various components to function as validation rules,
such as classifiers, models, or external services.
"""

from typing import Any, Protocol, TypeVar, runtime_checkable

from sifaka.rules.base import BaseValidator, ConfigurationError, RuleResult, ValidationError


T = TypeVar("T")


@runtime_checkable
class Adaptable(Protocol):
    """Protocol for components that can be adapted to rules."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...


class BaseAdapter(BaseValidator[Any]):
    """Base class for adapters that convert components to validators."""

    def __init__(self, adaptee: Adaptable) -> None:
        """
        Initialize with adaptee.

        Args:
            adaptee: The component being adapted

        Raises:
            ConfigurationError: If adaptee doesn't implement Adaptable protocol
        """
        self._validate_adaptee(adaptee)
        self._adaptee = adaptee

    def _validate_adaptee(self, adaptee: Any) -> None:
        """
        Validate that adaptee implements the required protocol.

        Args:
            adaptee: The component to validate

        Raises:
            ConfigurationError: If adaptee doesn't implement Adaptable protocol
        """
        if not isinstance(adaptee, Adaptable):
            raise ConfigurationError(
                f"Adaptee must implement Adaptable protocol, got {type(adaptee)}"
            )

    @property
    def adaptee(self) -> Adaptable:
        """Get the adaptee."""
        return self._adaptee

    def validate(self, input_value: Any, **kwargs) -> RuleResult:
        """
        Validate using the adaptee.

        Args:
            input_value: Input to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results

        Raises:
            ValidationError: If validation fails
        """
        try:
            # This is an abstract method that should be implemented by subclasses
            raise NotImplementedError("Subclasses must implement the validate method")
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e
