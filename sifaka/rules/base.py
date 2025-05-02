"""
Base classes for Sifaka rules.

This module defines the core architecture for rules and validators in Sifaka.

## Rule and Validator Relationship

Sifaka follows a delegation pattern for validation:

1. **Rules** are high-level containers that define what to validate
2. **Validators** implement the actual validation logic
3. Rules delegate validation work to their validators

This separation of concerns allows for:
- Reusing validation logic across different rules
- Testing validation logic independently
- Extending the framework with custom validators

## Instantiation Pattern

The recommended way to create rules and validators is through factory functions:

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a rule using the factory function
rule = create_length_rule(min_chars=10, max_chars=100)

# Validate text
result = rule.validate("This is a test")
```

Each rule type provides two factory functions:
1. `create_X_validator()` - Creates a standalone validator
2. `create_X_rule()` - Creates a rule with a validator

## Implementation Patterns

For detailed guidance on implementing rules and validators, see:
- [Rules and Validators Documentation](../../docs/rules_and_validators.md)
- [Rules README](./README.md)

These documents provide comprehensive information on:
- Standard implementation patterns
- Empty text handling
- Configuration patterns
- Best practices
- Example implementations
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
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, Field, ConfigDict

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic implementations
T = TypeVar("T")  # Input type
R = TypeVar("R", bound="RuleResult")  # Result type
V = TypeVar("V", bound="RuleValidator")  # Validator type
H = TypeVar("H", bound="RuleResultHandler")  # Handler type


class ValidationError(Exception):
    """Base exception for validation errors."""


class ConfigurationError(Exception):
    """Base exception for configuration errors."""


class RulePriority(Enum):
    """Priority levels for rule execution."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Validatable(Protocol[T_co]):
    """Protocol for objects that can be validated."""

    def is_valid(self) -> bool: ...
    def validate(self) -> None: ...
    @property
    def validation_errors(self) -> List[str]: ...


T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class RuleValidator(Protocol[T_contra]):
    """
    Protocol for rule validation logic.

    This protocol defines the interface that all validators must implement.
    Rules delegate validation work to objects implementing this protocol.

    The BaseValidator class provides a default implementation of this protocol,
    and most validators should extend BaseValidator rather than implementing
    this protocol directly.
    """

    def validate(self, output: T_contra, **kwargs: Any) -> "RuleResult": ...

    def can_validate(self, output: T_contra) -> bool: ...

    @property
    def validation_type(self) -> type: ...


class BaseValidator(Generic[T]):
    """
    Base class for validators that implements the RuleValidator protocol.

    Validators implement the actual validation logic and are used by Rules.
    Rules delegate validation work to their validators, following a clear
    separation of concerns.

    Validators should be created using factory functions rather than direct instantiation.
    Each validator type provides a `create_X_validator()` factory function.

    When implementing a custom validator:
    1. Extend this class or a domain-specific validator base class
    2. Override the validate() method to implement your validation logic
    3. Create a factory function for consistent instantiation
    """

    def validate(self, output: T, **kwargs: Any) -> "RuleResult":
        """
        Validate the output.

        This method handles common validation logic, including empty text handling.
        Subclasses should override this method to implement their validation logic.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Handle empty text for string validators
        if isinstance(output, str) and not output.strip():
            return RuleResult(
                passed=True,  # Default to passing for empty text
                message="Empty text validation skipped",
                metadata={"reason": "empty_input"},
            )

        # This is a placeholder implementation that should be overridden
        # by subclasses. We're using _ to indicate unused parameters.
        _ = output, kwargs
        raise NotImplementedError("Subclasses must implement validate method")

    def handle_empty_text(self, text: str) -> Optional["RuleResult"]:
        """
        Handle empty text validation.

        Args:
            text: The text to check

        Returns:
            RuleResult if text is empty, None otherwise
        """
        if not text.strip():
            return RuleResult(
                passed=True,
                message="Empty text validation skipped",
                metadata={"reason": "empty_input"},
            )
        return None

    def can_validate(self, output: T) -> bool:
        """
        Check if this validator can validate the output.

        Args:
            output: The output to check

        Returns:
            True if this validator can validate the output
        """
        return isinstance(output, self.validation_type)

    @property
    def validation_type(self) -> type:
        """
        Get the type this validator can validate.

        Returns:
            The type this validator can validate
        """
        # Default implementation returns str type
        return str


R_contra = TypeVar("R_contra", contravariant=True)


@runtime_checkable
class RuleResultHandler(Protocol[R_contra]):
    """Protocol for handling rule validation results."""

    def handle_result(self, result: R_contra) -> None: ...

    def should_continue(self, result: R_contra) -> bool: ...

    def can_handle(self, result: R_contra) -> bool: ...


class RuleResult(BaseModel):
    """Immutable result of a rule validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool = Field(description="Whether the validation passed")
    message: str = Field(description="Description of the validation result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the validation result",
    )
    score: Optional[float] = Field(
        default=None,
        description="Optional score for the validation result",
    )

    @property
    def failed(self) -> bool:
        """Check if validation failed."""
        return not self.passed

    def __bool__(self) -> bool:
        """Convert to boolean."""
        return self.passed

    def with_metadata(self, **kwargs: Any) -> "RuleResult":
        """
        Create a new result with additional metadata.

        Args:
            **kwargs: Additional metadata to add

        Returns:
            New result with additional metadata
        """
        return RuleResult(
            passed=self.passed,
            message=self.message,
            metadata={**self.metadata, **kwargs},
            score=self.score,
        )


class RuleConfig(BaseModel):
    """
    Immutable configuration for rules.

    This class provides a consistent way to configure rules.
    All rule-specific configuration options should be placed in the params dictionary:

    ```python
    config = RuleConfig(
        priority=RulePriority.HIGH,
        cost=1.0,
        params={
            "option1": "value1",
            "option2": "value2",
        }
    )
    ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    priority: RulePriority = Field(
        default=RulePriority.MEDIUM,
        description="Priority level for rule execution",
    )
    cache_size: int = Field(
        default=0,
        ge=0,
        description="Size of the validation cache",
    )
    cost: int = Field(
        default=1,
        ge=0,
        description="Cost of running the rule",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule-specific configuration parameters",
    )

    def with_options(self, **kwargs: Any) -> "RuleConfig":
        """
        Create a new config with updated options.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options
        """
        return RuleConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "RuleConfig":
        """
        Create a new config with updated params.

        Args:
            **kwargs: Params to update

        Returns:
            New config with updated params
        """
        return RuleConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )


class Rule(Generic[T, R, V, H], ABC):
    """
    Base class for all Sifaka rules.

    A rule validates an input against a specific criterion using a validator
    and optionally processes the results using a handler.

    Rules follow a delegation pattern:
    1. The Rule receives text to validate
    2. The Rule delegates validation to its Validator
    3. The Validator performs the actual validation logic
    4. The Validator returns a result to the Rule
    5. The Rule may perform additional processing on the result

    Rules should be created using factory functions rather than direct instantiation.
    Each rule type provides a `create_X_rule()` factory function.

    Type Parameters:
        T: The type of input to validate
        R: The type of validation result
        V: The type of validator (must implement RuleValidator[T])
        H: The type of result handler (must implement RuleResultHandler[R])
    """

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
        """
        self._name: Final[str] = name
        self._description: Final[str] = description
        self._config: Final[RuleConfig] = config or RuleConfig()

        # Set validator (either provided or create default)
        if validator is not None:
            if not isinstance(validator, RuleValidator):
                raise ConfigurationError(
                    f"Validator must implement RuleValidator protocol, got {type(validator)}"
                )
            self._validator: V = validator
        else:
            # Create a default validator
            self._validator = self._create_default_validator()

        # Validate and set handler if provided
        if result_handler is not None:
            if not isinstance(result_handler, RuleResultHandler):
                raise ConfigurationError(
                    f"Handler must implement RuleResultHandler protocol, got {type(result_handler)}"
                )
            if not result_handler.can_handle(cast(R, RuleResult(passed=True, message="test"))):
                raise ConfigurationError("Handler cannot handle the result type")
        self._result_handler: Final[Optional[H]] = result_handler

        # Initialize cache if enabled
        if self._config.cache_size > 0:
            # Instead of replacing the method, we'll create a cached validator wrapper
            # that delegates to the original validator
            original_validator = self._validator

            class CachedValidator(RuleValidator[T]):
                def __init__(self, validator: RuleValidator[T], cache_size: int):
                    self._validator = validator
                    self._cache_size = cache_size
                    self._cached_validate = lru_cache(maxsize=cache_size)(self._validator.validate)

                def validate(self, output: T, **kwargs: Any) -> RuleResult:
                    return self._cached_validate(output, **kwargs)

                def can_validate(self, output: T) -> bool:
                    return self._validator.can_validate(output)

                @property
                def validation_type(self) -> type:
                    return self._validator.validation_type

            # Replace the validator with a cached version
            self._validator = cast(V, CachedValidator(original_validator, self._config.cache_size))

    @abstractmethod
    def _create_default_validator(self) -> V:
        """
        Create a default validator for this rule.

        This method is called when a rule is instantiated without a validator.
        It's part of the delegation pattern where rules delegate validation
        to validator objects.

        When implementing a custom rule:
        1. Override this method to create a validator for your rule
        2. Use factory functions to create validators consistently
        3. Pass configuration from the rule to the validator

        Returns:
            A validator instance compatible with this rule
        """
        raise NotImplementedError("Subclasses must implement _create_default_validator")

    @property
    def name(self) -> str:
        """Get the rule name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the rule description."""
        return self._description

    @property
    def config(self) -> RuleConfig:
        """Get the rule configuration."""
        return self._config

    def _get_cache_key(self, output: T, **kwargs: Any) -> str:
        """Generate a cache key for the output."""
        hasher = hashlib.md5()
        hasher.update(str(output).encode())
        hasher.update(str(self._config).encode())
        hasher.update(str(sorted(kwargs.items())).encode())
        return hasher.hexdigest()

    def validate(self, output: T, **kwargs: Any) -> R:
        """
        Validate an output.

        This method handles:
        1. Input validation
        2. Type checking
        3. Caching
        4. Result handling
        5. Error handling

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            Validation result

        Raises:
            ValidationError: If validation fails
            TypeError: If output type is invalid
        """
        # Validate input type
        if not isinstance(output, self._validator.validation_type):
            raise TypeError(f"Output must be of type {self._validator.validation_type}")

        # Check if validator can handle the input
        if not self._validator.can_validate(output):
            raise ValidationError(f"Validator cannot handle input: {output}")

        try:
            # Use the validator (caching is handled at the validator level)
            result = self._validator.validate(output, **kwargs)

            # Handle result if handler is provided
            if self._result_handler is not None:
                self._result_handler.handle_result(result)
                if not self._result_handler.should_continue(result):
                    return cast(R, result)

            return cast(R, result)

        except Exception as e:
            # Convert to ValidationError
            raise ValidationError(f"Validation failed: {str(e)}") from e


class FunctionValidator(BaseValidator[str]):
    """Validator that wraps a function for validation."""

    def __init__(self, func: Callable[..., Any]) -> None:
        """Initialize with a validation function."""
        self._func = func

    def validate(self, output: str, **kwargs: Any) -> RuleResult:
        """Validate using the wrapped function."""
        # Handle empty text
        empty_result = self.handle_empty_text(output)
        if empty_result:
            return empty_result

        result = self._func(output, **kwargs)

        if isinstance(result, bool):
            return RuleResult(
                passed=result, message="" if result else f"Rule {self._func.__name__} failed"
            )
        elif isinstance(result, RuleResult):
            return result
        elif isinstance(result, tuple):
            passed, message, *rest = result
            metadata = rest[0] if rest else {}
            return RuleResult(passed=passed, message=message, metadata=metadata)
        else:
            raise ValidationError(
                f"Function {self._func.__name__} returned invalid type: {type(result)}"
            )


class FunctionRule(Rule[str, RuleResult, FunctionValidator, RuleResultHandler[RuleResult]]):
    """
    A rule that wraps a function for simple validation.

    The function must have one of these signatures:
    1. (str) -> bool
    2. (str) -> RuleResult
    3. (str) -> tuple[bool, str]
    4. (str) -> tuple[bool, str, dict]
    """

    def __init__(
        self,
        func: Callable[
            ..., Union[bool, RuleResult, Tuple[bool, str], Tuple[bool, str, Dict[str, Any]]]
        ],
        name: str,
        description: str = "",
        config: Optional[RuleConfig] = None,
    ) -> None:
        """Initialize a function-based rule."""
        self._func = func
        super().__init__(
            name=name,
            description=description or func.__doc__ or "",
            config=config,
        )

    def _create_default_validator(self) -> FunctionValidator:
        """Create a default validator from the function."""
        return FunctionValidator(self._func)


@runtime_checkable
class RuleProtocol(Protocol):
    """Protocol defining the interface for rules."""

    @property
    def name(self) -> str:
        """Get rule name."""
        ...

    @property
    def description(self) -> str:
        """Get rule description."""
        ...

    def validate(self, output: Any, **kwargs: Any) -> "RuleResult":
        """Validate output against rule criteria."""
        ...

    @property
    def config(self) -> "RuleConfig":
        """Get rule configuration."""
        ...


# Export these types
__all__ = [
    "Rule",
    "RuleConfig",
    "RuleResult",
    "RuleProtocol",
    "RuleValidator",
    "BaseValidator",
    "RuleResultHandler",
    "FunctionValidator",
    "FunctionRule",
    "ValidationError",
    "ConfigurationError",
    "RulePriority",
]
