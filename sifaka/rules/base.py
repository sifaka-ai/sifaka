"""
Base classes for Sifaka rules.

This module defines the core architecture for rules and validators in Sifaka,
providing the foundation for all validation components in the framework.
It implements a delegation pattern where rules delegate validation work to
specialized validator components.

The module provides these key components:
- Rule: Base class for all validation rules
- BaseValidator: Base class for validation logic
- Protocol classes for interface definitions
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from .interfaces.rule import RuleProtocol
from .config import RuleConfig, RulePriority
from .result import RuleResult
from ..utils.logging import get_logger
from ..utils.state import StateManager, create_rule_state
from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable

logger = get_logger(__name__)

# Type variables for generic implementations
T = TypeVar("T")  # Input type
R = TypeVar("R", bound="RuleResult")  # Result type
V = TypeVar("V", bound="RuleValidator")  # Validator type
H = TypeVar("H", bound="RuleResultHandler")  # Handler type


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class ConfigurationError(Exception):
    """Base exception for configuration errors."""

    pass


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Validatable(Protocol[T_co]):
    """Protocol for objects that can be validated."""

    def is_valid(self) -> bool:
        """
        Check if the object is valid.

        Returns:
            True if the object is valid, False otherwise
        """
        ...

    def validate(self) -> None:
        """
        Validate the object, raising exceptions for invalid objects.

        Raises:
            ValidationError: If the object is invalid
        """
        ...

    @property
    def validation_errors(self) -> List[str]:
        """
        Get the list of validation errors.

        Returns:
            List of validation error messages
        """
        ...


T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class RuleValidator(Protocol[T_contra]):
    """Protocol for rule validation logic."""

    def validate(self, output: T_contra, **kwargs: Any) -> RuleResult:
        """
        Validate the output and return a result.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            Validation result

        Raises:
            ValidationError: If validation cannot be performed
        """
        ...

    def can_validate(self, output: T_contra) -> bool:
        """
        Check if this validator can validate the output.

        Args:
            output: The output to check

        Returns:
            True if this validator can validate the output
        """
        ...

    @property
    def validation_type(self) -> type:
        """
        Get the type this validator can validate.

        Returns:
            The type this validator can validate
        """
        ...


class BaseValidator(Validatable[T], Generic[T]):
    """Base class for validators."""

    def __init__(self, validation_type: Type[T] = str):
        """Initialize the validator."""
        self._validation_type = validation_type

    def can_validate(self, input: T) -> bool:
        """Check if this validator can validate the input."""
        return isinstance(input, self._validation_type)

    @abstractmethod
    def validate(self, input: T) -> RuleResult:
        """Validate the input."""
        ...


R_contra = TypeVar("R_contra", contravariant=True)


@runtime_checkable
class RuleResultHandler(Protocol[R_contra]):
    """Protocol for handling rule validation results."""

    def handle_result(self, result: R_contra) -> None:
        """
        Handle a validation result.

        Args:
            result: The validation result to handle
        """
        ...

    def should_continue(self, result: R_contra) -> bool:
        """
        Determine if validation should continue after this result.

        Args:
            result: The validation result to check

        Returns:
            True if validation should continue, False otherwise
        """
        ...

    def can_handle(self, result: R_contra) -> bool:
        """
        Check if this handler can handle the result.

        Args:
            result: The validation result to check

        Returns:
            True if this handler can handle the result
        """
        ...


class BaseRule(BaseComponent[T, RuleResult], Generic[T]):
    """Base class for all rules."""

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[RuleConfig] = None,
        validator: Optional[BaseValidator[T]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the rule."""
        super().__init__(name, description, config or RuleConfig(**kwargs))
        self._validator = validator or self._create_default_validator()

    @abstractmethod
    def _create_default_validator(self) -> BaseValidator[T]:
        """Create the default validator for this rule."""
        ...

    def validate(self, input: T) -> RuleResult:
        """Validate the input using the rule's validator."""
        # Validate input
        if not self.validate_input(input):
            return RuleResult(
                passed=False, message="Invalid input", metadata={"error_type": "invalid_input"}
            )

        # Handle empty input
        empty_result = self.handle_empty_input(input)
        if empty_result:
            return empty_result

        try:
            # Run validation
            if not self._validator.can_validate(input):
                return RuleResult(
                    passed=False,
                    message="Invalid input type",
                    metadata={"error_type": "invalid_type"},
                )

            return self._validator.validate(input)

        except Exception as e:
            logger.error(f"Error validating input: {e}")
            return RuleResult(
                passed=False, message=f"Error: {str(e)}", metadata={"error_type": type(e).__name__}
            )

    def process(self, input: T) -> RuleResult:
        """Process the input through the rule pipeline."""
        return self.validate(input)

    @classmethod
    def create(cls: Type["BaseRule"], name: str, description: str, **kwargs: Any) -> "BaseRule":
        """Create a new rule instance."""
        return cls(name=name, description=description, config=RuleConfig(**kwargs))


class FunctionValidator(BaseValidator[T]):
    """Validator that uses a function for validation."""

    def __init__(self, func: Callable[[T], RuleResult], validation_type: Type[T] = str):
        """
        Initialize the validator.

        Args:
            func: Function to use for validation
            validation_type: Type this validator can validate
        """
        self._func = func
        self._validation_type = validation_type

    def validate(self, output: T, **kwargs: Any) -> RuleResult:
        """
        Validate the output using the function.

        Args:
            output: The output to validate
            **kwargs: Additional validation options

        Returns:
            Validation result
        """
        # Handle empty text for string validators
        if isinstance(output, str):
            empty_result = self.handle_empty_text(output)
            if empty_result:
                return empty_result

        # Call the validation function
        return self._func(output)

    @property
    def validation_type(self) -> type:
        """
        Get the type this validator can validate.

        Returns:
            The type this validator can validate
        """
        return self._validation_type


class FunctionRule(BaseRule[T, RuleResult, FunctionValidator[T], None]):
    """Rule that uses a function for validation."""

    def __init__(
        self,
        name: str,
        func: Callable[[T], RuleResult],
        description: str = "",
        config: Optional[RuleConfig] = None,
        validation_type: Type[T] = str,
    ):
        """
        Initialize the rule.

        Args:
            name: The name of the rule
            func: Function to use for validation
            description: Description of the rule
            config: Rule configuration
            validation_type: Type this rule can validate
        """
        validator = FunctionValidator(func, validation_type)
        super().__init__(
            name=name,
            description=description or f"Function rule using {func.__name__}",
            config=config,
            validator=validator,
        )

    def _create_default_validator(self) -> FunctionValidator[T]:
        """
        Create a default validator for this rule.

        This method should never be called because we always provide a validator
        in the constructor.

        Returns:
            Default validator instance

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("FunctionRule requires a validator")


def create_rule(
    name: str,
    validator: RuleValidator,
    description: str = "",
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> BaseRule:
    """
    Create a rule with the given validator and configuration.

    Args:
        name: Name of the rule
        validator: Validator to use for validation
        description: Description of the rule
        config: Configuration for the rule
        **kwargs: Additional arguments for the rule constructor

    Returns:
        A new rule instance
    """
    from .factories import create_rule as factory_create_rule

    return factory_create_rule(
        name=name, validator=validator, description=description, config=config, **kwargs
    )


# Export these types
__all__ = [
    # Exception classes
    "ValidationError",
    "ConfigurationError",
    # Protocol classes
    "Validatable",
    "RuleValidator",
    "RuleResultHandler",
    # Base classes
    "BaseValidator",
    "BaseRule",
    # Concrete classes
    "FunctionValidator",
    "FunctionRule",
    # Factory functions
    "create_rule",
]
