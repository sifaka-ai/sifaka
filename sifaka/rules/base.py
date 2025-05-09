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


class BaseValidator(Generic[T]):
    """Base class for validators that implements the RuleValidator protocol."""

    def __init__(self):
        """Initialize the validator."""
        # Initialize state manager
        self._state = StateManager()
        self._state.update("initialized", False)
        self._state.update("cache", {})
        self._state.set_metadata("component_type", "validator")
        self._state.set_metadata("validation_count", 0)

    def validate(self, output: T, **kwargs: Any) -> RuleResult:
        """
        Validate the output.

        This method handles common validation logic, including empty text handling.
        Subclasses should override this method to implement their validation logic.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            Validation result

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        # Track validation count
        count = self._state.get_metadata("validation_count", 0)
        self._state.set_metadata("validation_count", count + 1)

        # Handle empty text for string validators
        if isinstance(output, str):
            from sifaka.utils.text import handle_empty_text

            empty_result = handle_empty_text(output, component_type="rule")
            if empty_result:
                # Track empty validations
                empty_count = self._state.get_metadata("empty_count", 0)
                self._state.set_metadata("empty_count", empty_count + 1)
                return empty_result

        # This is a placeholder implementation that should be overridden
        # by subclasses. We're using _ to indicate unused parameters.
        _ = output, kwargs
        raise NotImplementedError("Subclasses must implement validate method")

    def handle_empty_text(self, text: str) -> Optional[RuleResult]:
        """
        Handle empty text validation.

        Args:
            text: The text to check

        Returns:
            RuleResult if text is empty, None otherwise
        """
        from sifaka.utils.text import handle_empty_text

        return handle_empty_text(text, component_type="rule")

    def can_validate(self, output: T) -> bool:
        """
        Check if this validator can validate the output.

        Args:
            output: The output to check

        Returns:
            True if this validator can validate the output
        """
        # Track validation check
        check_count = self._state.get_metadata("can_validate_checks", 0)
        self._state.set_metadata("can_validate_checks", check_count + 1)

        result = isinstance(output, self.validation_type)

        # Track validation type matches
        if result:
            match_count = self._state.get_metadata("validation_type_matches", 0)
            self._state.set_metadata("validation_type_matches", match_count + 1)

        return result

    @property
    def validation_type(self) -> type:
        """
        Get the type this validator can validate.

        Returns:
            The type this validator can validate
        """
        # Default implementation returns str type
        return str

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics for this validator.

        Returns:
            Dictionary with statistics
        """
        return {
            "validation_count": self._state.get_metadata("validation_count", 0),
            "empty_count": self._state.get_metadata("empty_count", 0),
            "can_validate_checks": self._state.get_metadata("can_validate_checks", 0),
            "validation_type_matches": self._state.get_metadata("validation_type_matches", 0),
        }


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


class Rule(Generic[T, R, V, H], ABC):
    """Base class for all Sifaka rules."""

    # Add state manager as a private attribute
    _state = PrivateAttr(default_factory=create_rule_state)

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[RuleConfig] = None,
        validator: Optional[V] = None,
        result_handler: Optional[H] = None,
    ) -> None:
        """
        Initialize a rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
            result_handler: Optional handler for validation results

        Raises:
            ConfigurationError: If validator or handler doesn't meet requirements
        """
        self._name: Final[str] = name
        self._description: Final[str] = description
        self._config: Final[RuleConfig] = config or RuleConfig()

        # Initialize state
        self._state.update("initialized", False)
        self._state.update("cache", {})
        self._state.update("validation_count", 0)
        self._state.set_metadata("component_type", "rule")
        self._state.set_metadata("name", name)
        self._state.set_metadata("description", description)
        self._state.set_metadata("priority", str(self._config.priority))

        # Set validator (either provided or create default)
        if validator is not None:
            if not isinstance(validator, RuleValidator):
                raise ConfigurationError(
                    f"Validator must implement RuleValidator protocol, got {type(validator)}"
                )
            self._validator: V = validator
            self._state.update("validator", validator)
        else:
            # Create a default validator
            self._validator = self._create_default_validator()
            self._state.update("validator", self._validator)

        # Set result handler (either provided or create default)
        if result_handler is not None:
            if not isinstance(result_handler, RuleResultHandler):
                raise ConfigurationError(
                    f"Result handler must implement RuleResultHandler protocol, got {type(result_handler)}"
                )
            self._result_handler: Optional[H] = result_handler
            self._state.update("result_handler", result_handler)
        else:
            # No default handler
            self._result_handler = None

    def _create_default_validator(self) -> V:
        """
        Create a default validator for this rule.

        Returns:
            Default validator instance

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _create_default_validator")

    @property
    def name(self) -> str:
        """
        Get the name of the rule.

        Returns:
            The name of the rule
        """
        return self._name

    @property
    def description(self) -> str:
        """
        Get the description of the rule.

        Returns:
            The description of the rule
        """
        return self._description

    @property
    def config(self) -> RuleConfig:
        """
        Get the configuration of the rule.

        Returns:
            The configuration of the rule
        """
        return self._config

    @property
    def validator(self) -> V:
        """
        Get the validator used by this rule.

        Returns:
            The validator used by this rule
        """
        return self._validator

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate text against this rule.

        Args:
            text: The text to validate
            **kwargs: Additional validation options

        Returns:
            Validation result
        """
        # Check if result is already cached
        cache = self._state.get("cache", {})
        cache_key = self._create_cache_key(text, kwargs)

        if cache_key in cache and self._config.use_cache:
            self._state.set_metadata("cache_hit", True)
            return cache[cache_key]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Delegate validation to the validator
        result = self._validator.validate(text, **kwargs)

        # Add rule metadata to the result
        result_with_metadata = result.with_metadata(
            rule_name=self.name,
            rule_description=self.description,
        )

        # Handle the result if a handler is provided
        if self._result_handler is not None:
            self._result_handler.handle_result(result_with_metadata)

        # Update validation statistics
        validation_count = self._state.get("validation_count", 0)
        self._state.update("validation_count", validation_count + 1)

        # Update the pass/fail statistics
        if result.passed:
            pass_count = self._state.get_metadata("pass_count", 0)
            self._state.set_metadata("pass_count", pass_count + 1)
        else:
            fail_count = self._state.get_metadata("fail_count", 0)
            self._state.set_metadata("fail_count", fail_count + 1)

        # Cache the result if caching is enabled
        if self._config.use_cache:
            cache[cache_key] = result_with_metadata
            self._state.update("cache", cache)

        return result_with_metadata

    def _create_cache_key(self, text: str, kwargs: Dict[str, Any]) -> str:
        """
        Create a cache key for validation results.

        Args:
            text: The text to validate
            kwargs: Additional validation options

        Returns:
            Cache key as a string
        """
        # For simple cases, we can just use the text itself
        if not kwargs:
            return text[:100]  # Limit key size for large texts

        # For more complex cases with kwargs, create a hash
        key_parts = [text[:100]]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{str(v)}")

        combined = "|".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics for this rule.

        Returns:
            Dictionary with statistics
        """
        return {
            "validation_count": self._state.get("validation_count", 0),
            "pass_count": self._state.get_metadata("pass_count", 0),
            "fail_count": self._state.get_metadata("fail_count", 0),
            "cache_size": len(self._state.get("cache", {})),
        }

    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self._state.update("validation_count", 0)
        self._state.set_metadata("pass_count", 0)
        self._state.set_metadata("fail_count", 0)

    def clear_cache(self) -> None:
        """Clear the validation result cache."""
        self._state.update("cache", {})


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


class FunctionRule(Rule[T, RuleResult, FunctionValidator[T], None]):
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
) -> Rule:
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
    "Rule",
    # Concrete classes
    "FunctionValidator",
    "FunctionRule",
    # Factory functions
    "create_rule",
]
