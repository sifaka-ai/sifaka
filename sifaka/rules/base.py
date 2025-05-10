"""
Base classes for Sifaka rules.

This module defines the core architecture for rules and validators in Sifaka,
providing the foundation for all validation components in the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Protocol, runtime_checkable
import time

from sifaka.core.base import (
    BaseComponent,
    BaseConfig,
    BaseResult,
    ComponentResultEnum,
    Validatable,
    ValidationPattern,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
V = TypeVar("V", bound="BaseValidator")  # Validator type


class RuleConfig(BaseConfig):
    """Configuration for rules."""

    severity: str = "error"
    category: str = "general"
    tags: List[str] = []
    track_performance: bool = True
    track_errors: bool = True


class RuleResult(BaseResult):
    """Result from rule validation."""

    severity: str = "error"
    category: str = "general"
    tags: List[str] = []


@runtime_checkable
class RuleValidator(Protocol[T]):
    """Protocol for rule validation logic."""

    def validate(self, input: T) -> RuleResult:
        """Validate the input."""
        ...

    def can_validate(self, input: T) -> bool:
        """Check if this validator can validate the input."""
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
        start_time = time.time()

        # Validate input
        if not self.validate_input(input):
            return RuleResult(
                passed=False,
                message="Invalid input",
                metadata={"error_type": "invalid_input"},
                score=0.0,
                issues=["Invalid input type"],
                suggestions=["Provide valid input"],
                processing_time_ms=time.time() - start_time,
            )

        # Handle empty input
        empty_result = self.handle_empty_input(input)
        if empty_result:
            return empty_result.with_metadata(processing_time_ms=time.time() - start_time)

        try:
            # Run validation
            if not self._validator.can_validate(input):
                return RuleResult(
                    passed=False,
                    message="Invalid input type",
                    metadata={"error_type": "invalid_type"},
                    score=0.0,
                    issues=["Input type not supported"],
                    suggestions=["Use supported input type"],
                    processing_time_ms=time.time() - start_time,
                )

            result = self._validator.validate(input)
            self.update_statistics(result)
            return result.with_metadata(processing_time_ms=time.time() - start_time)

        except Exception as e:
            self.record_error(e)
            logger.error(f"Error validating input: {e}")
            return RuleResult(
                passed=False,
                message=f"Error: {str(e)}",
                metadata={"error_type": type(e).__name__},
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Retry with different input"],
                processing_time_ms=time.time() - start_time,
            )

    def process(self, input: T) -> RuleResult:
        """Process the input through the rule pipeline."""
        return self.validate(input)


class FunctionValidator(BaseValidator[T]):
    """Validator that uses a function for validation."""

    def __init__(self, func: callable[[T], RuleResult], validation_type: Type[T] = str):
        """Initialize the validator."""
        super().__init__(validation_type)
        self._func = func

    def validate(self, input: T) -> RuleResult:
        """Validate the input using the function."""
        return self._func(input)


class FunctionRule(BaseRule[T]):
    """Rule that uses a function for validation."""

    def __init__(
        self,
        name: str,
        func: callable[[T], RuleResult],
        description: str = "",
        config: Optional[RuleConfig] = None,
        validation_type: Type[T] = str,
    ):
        """Initialize the rule."""
        validator = FunctionValidator(func, validation_type)
        super().__init__(
            name=name,
            description=description or f"Function rule using {func.__name__}",
            config=config,
            validator=validator,
        )

    def _create_default_validator(self) -> BaseValidator[T]:
        """Create a default validator for this rule."""
        raise NotImplementedError("FunctionRule requires a validator")


def create_rule(
    name: str,
    validator: RuleValidator,
    description: str = "",
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> BaseRule:
    """Create a rule with the given validator and configuration."""
    return FunctionRule(
        name=name,
        func=validator.validate,
        description=description,
        config=config,
        **kwargs,
    )


__all__ = [
    "BaseValidator",
    "BaseRule",
    "RuleConfig",
    "RuleResult",
    "RuleValidator",
    "FunctionValidator",
    "FunctionRule",
    "create_rule",
]
