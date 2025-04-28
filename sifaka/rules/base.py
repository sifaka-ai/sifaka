"""
Base classes for Sifaka rules.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

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


@runtime_checkable
class Validatable(Protocol[T]):
    """Protocol for objects that can be validated."""

    def is_valid(self) -> bool: ...
    def validate(self) -> None: ...
    @property
    def validation_errors(self) -> list[str]: ...


@runtime_checkable
class RuleValidator(Protocol[T]):
    """Protocol for rule validation logic."""

    @abstractmethod
    def validate(self, output: T, **kwargs) -> "RuleResult": ...

    @abstractmethod
    def can_validate(self, output: T) -> bool: ...

    @property
    @abstractmethod
    def validation_type(self) -> type[T]: ...


@runtime_checkable
class RuleResultHandler(Protocol[R]):
    """Protocol for handling rule validation results."""

    @abstractmethod
    def handle_result(self, result: R) -> None: ...

    @abstractmethod
    def should_continue(self, result: R) -> bool: ...

    @abstractmethod
    def can_handle(self, result: R) -> bool: ...


@dataclass(frozen=True)
class RuleResult:
    """Immutable result of a rule validation."""

    passed: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")

    @property
    def failed(self) -> bool:
        """Return whether the validation failed."""
        return not self.passed

    def __bool__(self) -> bool:
        """Return whether the validation passed."""
        return self.passed

    def with_metadata(self, **kwargs: Any) -> "RuleResult":
        """Create a new result with additional metadata."""
        return RuleResult(
            passed=self.passed,
            message=self.message,
            metadata={**self.metadata, **kwargs},
            score=self.score,
        )


@dataclass(frozen=True)
class RuleConfig:
    """Immutable configuration for rules."""

    priority: RulePriority = RulePriority.MEDIUM
    cache_size: int = 0
    cost: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cache_size < 0:
            raise ConfigurationError("Cache size must be non-negative")
        if self.cost < 0:
            raise ConfigurationError("Cost must be non-negative")

    def with_options(self, **kwargs: Any) -> "RuleConfig":
        """Create a new config with updated options."""
        return RuleConfig(**{**self.__dict__, **kwargs})


class Rule(Generic[T, R, V, H], ABC):
    """
    Base class for all Sifaka rules.

    A rule validates an input against a specific criterion using a validator
    and optionally processes the results using a handler.

    Type Parameters:
        T: The type of input to validate
        R: The type of validation result
        V: The type of validator
        H: The type of result handler
    """

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[V] = None,
        config: Optional[RuleConfig] = None,
        result_handler: Optional[H] = None,
    ) -> None:
        """
        Initialize a rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional validator implementation
            config: Rule configuration
            result_handler: Optional handler for validation results
        """
        self._name: Final[str] = name
        self._description: Final[str] = description
        self._config: Final[RuleConfig] = config or RuleConfig()

        # Validate and set validator if provided
        if validator is not None:
            if not isinstance(validator, RuleValidator):
                raise ConfigurationError(
                    f"Validator must implement RuleValidator protocol, got {type(validator)}"
                )
            self._validator: Final[V] = validator
        else:
            # Create a default validator that delegates to _validate_impl
            class DefaultValidator(RuleValidator[T]):
                def __init__(self, rule: Rule) -> None:
                    self._rule = rule

                def validate(self, output: T, **kwargs) -> RuleResult:
                    return self._rule._validate_impl(output, **kwargs)

                def can_validate(self, output: T) -> bool:
                    return isinstance(output, str)

                @property
                def validation_type(self) -> type[T]:
                    return str

            self._validator = DefaultValidator(self)

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
            self._validate_cached = lru_cache(maxsize=self._config.cache_size)(self._validate_impl)
        else:
            self._validate_cached = self._validate_impl

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

    def _get_cache_key(self, output: T, **kwargs) -> str:
        """Generate a cache key for the output."""
        hasher = hashlib.md5()
        hasher.update(str(output).encode())
        hasher.update(str(self._config).encode())
        hasher.update(str(sorted(kwargs.items())).encode())
        return hasher.hexdigest()

    @abstractmethod
    def _validate_impl(self, output: T, **kwargs) -> R:
        """
        Implement the validation logic.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            Validation result

        Raises:
            ValidationError: If validation fails
        """

    def validate(self, output: T, **kwargs) -> R:
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
            # Get from cache or validate
            if self._config.cache_size > 0:
                self._get_cache_key(output, **kwargs)
                result = self._validate_cached(output, **kwargs)
            else:
                result = self._validate_impl(output, **kwargs)

            # Handle result if handler is provided
            if self._result_handler is not None:
                self._result_handler.handle_result(result)
                if not self._result_handler.should_continue(result):
                    return result

            return result

        except Exception as e:
            # Convert to ValidationError
            raise ValidationError(f"Validation failed: {str(e)}") from e


class FunctionRule(Rule[str, RuleResult, RuleValidator[str], RuleResultHandler[RuleResult]]):
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
        super().__init__(
            name=name,
            description=description or func.__doc__ or "",
            validator=self._create_validator(func),
            config=config,
        )
        self._func = func

    def _create_validator(self, func: Callable) -> RuleValidator[str]:
        """Create a validator from the function."""

        class FunctionValidator(RuleValidator[str]):
            def __init__(self, func: Callable) -> None:
                self._func = func

            def validate(self, output: str, **kwargs) -> RuleResult:
                result = self._func(output, **kwargs)

                if isinstance(result, bool):
                    return RuleResult(
                        passed=result, message="" if result else f"Rule {func.__name__} failed"
                    )
                elif isinstance(result, RuleResult):
                    return result
                elif isinstance(result, tuple):
                    passed, message, *rest = result
                    metadata = rest[0] if rest else {}
                    return RuleResult(passed=passed, message=message, metadata=metadata)
                else:
                    raise ValidationError(
                        f"Function {func.__name__} returned invalid type: {type(result)}"
                    )

            def can_validate(self, output: str) -> bool:
                return isinstance(output, str)

            @property
            def validation_type(self) -> type[str]:
                return str

        return FunctionValidator(func)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Implement validation using the wrapped function."""
        return self._validator.validate(output, **kwargs)


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

    def validate(self, output: Any, **kwargs) -> "RuleResult":
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
    "RuleResultHandler",
    "ValidationError",
    "ConfigurationError",
]
