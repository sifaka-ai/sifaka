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


class BaseValidator(Generic[T]):
    """Base class for validators that implements the RuleValidator protocol."""

    def validate(self, output: T, **kwargs) -> "RuleResult":
        """
        Validate the output.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # This is a placeholder implementation that should be overridden
        # by subclasses. We're using _ to indicate unused parameters.
        _ = output, kwargs
        raise NotImplementedError("Subclasses must implement validate method")

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
    def validation_type(self) -> type[T]:
        """
        Get the type this validator can validate.

        Returns:
            The type this validator can validate
        """
        return str  # Default to string, override in subclasses


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
    """
    Immutable configuration for rules.

    This class provides a consistent way to configure rules with both params and metadata fields.
    The preferred way to configure rules is to use the params dictionary:

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

    The metadata field is kept for backward compatibility.
    """

    priority: RulePriority = RulePriority.MEDIUM
    cache_size: int = 0
    cost: int = 1
    params: Dict[str, Any] = field(default_factory=dict)

    # Keep metadata for backward compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cache_size < 0:
            raise ConfigurationError("Cache size must be non-negative")
        if self.cost < 0:
            raise ConfigurationError("Cost must be non-negative")

        # For backward compatibility, if metadata is provided but params is empty,
        # copy metadata to params
        if self.metadata and not self.params:
            object.__setattr__(self, "params", dict(self.metadata))

    def with_options(self, **kwargs: Any) -> "RuleConfig":
        """Create a new config with updated options."""
        return RuleConfig(**{**self.__dict__, **kwargs})

    def with_params(self, **kwargs: Any) -> "RuleConfig":
        """Create a new config with updated parameters."""
        new_params = {**self.params, **kwargs}
        return RuleConfig(
            priority=self.priority,
            cache_size=self.cache_size,
            cost=self.cost,
            params=new_params,
            metadata=self.metadata,
        )


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
            self._validator: Final[V] = validator
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
            self._validator_validate = self._validator.validate
            self._validator.validate = lru_cache(maxsize=self._config.cache_size)(
                self._validator_validate
            )

    @abstractmethod
    def _create_default_validator(self) -> V:
        """Create a default validator for this rule."""
        pass

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
            # Use the validator (caching is handled at the validator level)
            result = self._validator.validate(output, **kwargs)

            # Handle result if handler is provided
            if self._result_handler is not None:
                self._result_handler.handle_result(result)
                if not self._result_handler.should_continue(result):
                    return result

            return result

        except Exception as e:
            # Convert to ValidationError
            raise ValidationError(f"Validation failed: {str(e)}") from e


class FunctionValidator(BaseValidator[str]):
    """Validator that wraps a function for validation."""

    def __init__(self, func: Callable) -> None:
        """Initialize with a validation function."""
        self._func = func

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate using the wrapped function."""
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
    "BaseValidator",
    "RuleResultHandler",
    "FunctionValidator",
    "FunctionRule",
    "ValidationError",
    "ConfigurationError",
    "RulePriority",
]
