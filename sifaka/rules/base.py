"""
Base classes for Sifaka rules.

This module defines the core architecture for rules and validators in Sifaka.

## Architecture Overview

Sifaka follows a delegation pattern for validation:

1. **Rules** are high-level containers that define what to validate
2. **Validators** implement the actual validation logic
3. Rules delegate validation work to their validators

This separation of concerns allows for:
- Reusing validation logic across different rules
- Testing validation logic independently
- Extending the framework with custom validators

## Component Lifecycle

### Validators
1. **Initialization**: Create with configuration
2. **Validation**: Process input text and apply validation logic
3. **Result Creation**: Return standardized RuleResult objects

### Rules
1. **Initialization**: Set up with name, description, config, validator
2. **Delegation**: Forward validation requests to validator
3. **Result Processing**: Enhance validation results with rule metadata

## Instantiation Pattern

The recommended way to create rules and validators is through factory functions:

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a rule using the factory function
rule = create_length_rule(min_chars=10, max_chars=100)

# Validate text
result = rule.validate("This is a test")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
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
    Type,
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
    """
    Priority levels for rule execution.

    Rules with higher priority are generally executed first.
    This allows critical validations to fail early in a chain.
    """

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Validatable(Protocol[T_co]):
    """
    Protocol for objects that can be validated.

    Implementing classes should provide methods to check validity,
    trigger validation, and expose validation errors.

    Type Parameters:
        T_co: The covariant result type
    """

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
    """
    Protocol for rule validation logic.

    This protocol defines the interface that all validators must implement.
    Rules delegate validation work to objects implementing this protocol.

    The BaseValidator class provides a default implementation of this protocol,
    and most validators should extend BaseValidator rather than implementing
    this protocol directly.

    Type Parameters:
        T_contra: The contravariant input type

    Lifecycle:
    1. Initialization: Set up with any necessary configuration
    2. Validation: Process input and apply validation logic
    3. Result: Return standardized validation results

    Examples:
        ```python
        from sifaka.rules.base import RuleValidator, RuleResult

        class MyValidator(RuleValidator[str]):
            def validate(self, text: str, **kwargs) -> RuleResult:
                is_valid = len(text) > 10
                return RuleResult(
                    passed=is_valid,
                    message="Text length validation" +
                            (" passed" if is_valid else " failed")
                )

            def can_validate(self, text: str) -> bool:
                return isinstance(text, str)

            @property
            def validation_type(self) -> type:
                return str
        ```
    """

    def validate(self, output: T_contra, **kwargs: Any) -> "RuleResult":
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
    """
    Base class for validators that implements the RuleValidator protocol.

    Validators implement the actual validation logic and are used by Rules.
    Rules delegate validation work to their validators, following a clear
    separation of concerns.

    Validators should be created using factory functions rather than direct instantiation.
    Each validator type provides a `create_X_validator()` factory function.

    Type Parameters:
        T: The input type to validate

    Lifecycle:
    1. Initialization: Set up validator configuration
    2. Validation Request: Receive input to validate
    3. Input Processing: Apply validation logic to input
    4. Result Creation: Return standardized validation results

    Examples:
        ```python
        from sifaka.rules.base import BaseValidator, RuleResult

        class LengthValidator(BaseValidator[str]):
            def __init__(self, min_length: int):
                self.min_length = min_length

            def validate(self, text: str, **kwargs) -> RuleResult:
                # Handle empty text first
                empty_result = self.handle_empty_text(text)
                if empty_result:
                    return empty_result

                # Apply validation logic
                is_valid = len(text) >= self.min_length
                return RuleResult(
                    passed=is_valid,
                    message=f"Text length is {len(text)}" +
                            (f", which meets minimum of {self.min_length}"
                             if is_valid
                             else f", which is below minimum of {self.min_length}")
                )
        ```
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

        Raises:
            NotImplementedError: If not implemented by subclass
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

        This is a utility method to provide consistent handling of empty text
        across different validators. Validators should call this method at the
        beginning of their validate method when processing string inputs.

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
    """
    Protocol for handling rule validation results.

    Result handlers provide a way to process validation results
    and determine if validation should continue after a result
    is received.

    Type Parameters:
        R_contra: The contravariant result type

    Lifecycle:
    1. Initialization: Set up handler configuration
    2. Result Reception: Receive validation results
    3. Processing: Apply handler logic to results
    4. Continuation Decision: Determine if validation should continue

    Examples:
        ```python
        from sifaka.rules.base import RuleResultHandler, RuleResult

        class LoggingHandler(RuleResultHandler[RuleResult]):
            def handle_result(self, result: RuleResult) -> None:
                print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")

            def should_continue(self, result: RuleResult) -> bool:
                # Continue validation even if this rule failed
                return True

            def can_handle(self, result: RuleResult) -> bool:
                return isinstance(result, RuleResult)
        ```
    """

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


class RuleResult(BaseModel):
    """
    Immutable result of a rule validation.

    This class provides a standardized way to represent validation results
    with metadata and optional scores. It is immutable to prevent accidental
    modification after creation.

    Lifecycle:
    1. Creation: Instantiated with validation outcome
    2. Usage: Accessed by rules and handlers
    3. Augmentation: New instances created with additional metadata

    Examples:
        ```python
        # Create a successful validation result
        result = RuleResult(
            passed=True,
            message="Text meets all requirements",
            metadata={"char_count": 150, "word_count": 23},
            score=0.95
        )

        # Access properties
        if result.passed:
            print(f"Validation passed: {result.message}")
            print(f"Score: {result.score}")

        # Create a new result with additional metadata
        enhanced_result = result.with_metadata(rule_id="length_rule")
        ```
    """

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
        """
        Check if validation failed.

        Returns:
            True if validation failed, False if it passed
        """
        return not self.passed

    def __bool__(self) -> bool:
        """
        Convert to boolean.

        Returns:
            True if validation passed, False if it failed
        """
        return self.passed

    def with_metadata(self, **kwargs: Any) -> "RuleResult":
        """
        Create a new result with additional metadata.

        This method is useful for adding rule-specific information
        to a result created by a validator.

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

    Lifecycle:
    1. Creation: Instantiated with configuration options
    2. Usage: Accessed by rules during setup and validation
    3. Modification: New instances created with updated options

    Examples:
        ```python
        # Create a rule configuration
        config = RuleConfig(
            priority=RulePriority.HIGH,
            cost=1.0,
            params={
                "option1": "value1",
                "option2": "value2",
            }
        )

        # Create a new configuration with updated options
        updated_config = config.with_options(priority=RulePriority.CRITICAL)

        # Create a new configuration with updated params
        parameterized_config = config.with_params(option3="value3")
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

        This method is useful for updating top-level configuration
        options without modifying the params dictionary.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options
        """
        return RuleConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "RuleConfig":
        """
        Create a new config with updated params.

        This method is useful for updating or adding rule-specific
        parameters without modifying other configuration options.

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

    Lifecycle:
    1. Initialization: Set up with name, description, config, validator, handler
    2. Validation Request: Receive input to validate
    3. Delegation: Forward validation to validator
    4. Result Processing: Apply rule-specific enhancements to results
    5. Result Handling: Forward results to handler if provided

    Examples:
        ```python
        from sifaka.rules.base import Rule, RuleResult, RuleConfig

        class LengthRule(Rule[str, RuleResult, LengthValidator, None]):
            def __init__(
                self,
                min_length: int,
                name: str = "length_rule",
                description: str = "Validates text length",
                config: Optional[RuleConfig] = None,
            ):
                self.min_length = min_length
                super().__init__(name=name, description=description, config=config)

            def _create_default_validator(self) -> LengthValidator:
                return LengthValidator(min_length=self.min_length)

            # validate method is inherited from Rule base class
        ```
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

        Raises:
            ConfigurationError: If validator or handler doesn't meet requirements
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

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _create_default_validator")

    @property
    def name(self) -> str:
        """
        Get the rule name.

        Returns:
            The name of the rule
        """
        return self._name

    @property
    def description(self) -> str:
        """
        Get the rule description.

        Returns:
            The description of the rule
        """
        return self._description

    @property
    def config(self) -> RuleConfig:
        """
        Get the rule configuration.

        Returns:
            The configuration of the rule
        """
        return self._config

    def _get_cache_key(self, output: T, **kwargs: Any) -> str:
        """
        Generate a cache key for the output.

        This method is used internally for caching validation results.

        Args:
            output: The output to generate a cache key for
            **kwargs: Additional validation context

        Returns:
            A string cache key
        """
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
    """
    Validator that wraps a function for validation.

    This validator makes it easy to use simple functions for validation
    without creating a full validator class. It handles various return
    types from the wrapped function and converts them to RuleResult objects.

    Lifecycle:
    1. Initialization: Wrap a validation function
    2. Validation: Call the function with input
    3. Result Conversion: Convert function output to RuleResult

    Examples:
        ```python
        # Create a validator from a simple function
        def is_long_enough(text: str) -> bool:
            return len(text) > 10

        validator = FunctionValidator(is_long_enough)

        # Use the validator
        result = validator.validate("Hello world")
        ```
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        """
        Initialize with a validation function.

        Args:
            func: The function to use for validation
        """
        self._func = func

    def validate(self, output: str, **kwargs: Any) -> RuleResult:
        """
        Validate using the wrapped function.

        This method calls the wrapped function and converts its
        return value to a RuleResult.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result

        Raises:
            ValidationError: If the function returns an invalid type
        """
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

    Lifecycle:
    1. Initialization: Wrap a validation function
    2. Validation: Delegate to FunctionValidator

    Examples:
        ```python
        # Create a rule from a simple function
        def contains_keywords(text: str, keywords: List[str]) -> bool:
            return any(keyword in text.lower() for keyword in keywords)

        rule = FunctionRule(
            func=contains_keywords,
            name="keyword_rule",
            description="Checks if text contains required keywords"
        )

        # Use the rule with kwargs
        result = rule.validate("Hello world", keywords=["hello", "world"])
        ```
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
        """
        Initialize a function-based rule.

        Args:
            func: The function to use for validation
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
        """
        self._func = func
        super().__init__(
            name=name,
            description=description or func.__doc__ or "",
            config=config,
        )

    def _create_default_validator(self) -> FunctionValidator:
        """
        Create a default validator from the function.

        Returns:
            A FunctionValidator wrapping the function
        """
        return FunctionValidator(self._func)


@runtime_checkable
class RuleProtocol(Protocol):
    """
    Protocol defining the interface for rules.

    This protocol is useful for type checking code that works with rules
    without requiring a specific rule implementation.

    Lifecycle:
    1. Access: Get rule properties
    2. Validation: Validate input against rule

    Examples:
        ```python
        def process_with_rule(rule: RuleProtocol, text: str) -> None:
            print(f"Validating with rule: {rule.name}")
            result = rule.validate(text)
            print(f"Result: {result.message}")
        ```
    """

    @property
    def name(self) -> str:
        """
        Get rule name.

        Returns:
            The name of the rule
        """
        ...

    @property
    def description(self) -> str:
        """
        Get rule description.

        Returns:
            The description of the rule
        """
        ...

    def validate(self, output: Any, **kwargs: Any) -> "RuleResult":
        """
        Validate output against rule criteria.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            Validation result

        Raises:
            ValidationError: If validation fails
        """
        ...

    @property
    def config(self) -> "RuleConfig":
        """
        Get rule configuration.

        Returns:
            The configuration of the rule
        """
        ...


def create_rule(
    rule_type: Type[Rule],
    name: str,
    description: str = "",
    config: Optional[RuleConfig] = None,
    validator: Optional[RuleValidator] = None,
    **kwargs: Any
) -> Rule:
    """
    Factory function to create a rule with standardized configuration.

    This function simplifies the creation of rules by providing a
    consistent interface. It's useful for creating rules in a generic way.

    Args:
        rule_type: The class of the rule to create
        name: The name of the rule
        description: Description of the rule
        config: Optional rule configuration
        validator: Optional validator implementation
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A configured rule instance

    Examples:
        ```python
        # Create a length rule
        from sifaka.rules.formatting.length import LengthRule, LengthValidator

        validator = LengthValidator(min_chars=10, max_chars=100)
        rule = create_rule(
            rule_type=LengthRule,
            name="custom_length_rule",
            description="Custom rule for length validation",
            validator=validator
        )
        ```
    """
    return rule_type(
        name=name,
        description=description,
        config=config,
        validator=validator,
        **kwargs
    )


# Export these types
__all__ = [
    # Exception classes
    "ValidationError",
    "ConfigurationError",
    # Enum classes
    "RulePriority",
    # Protocol classes
    "Validatable",
    "RuleValidator",
    "RuleResultHandler",
    "RuleProtocol",
    # Model classes
    "RuleResult",
    "RuleConfig",
    # Base classes
    "BaseValidator",
    "Rule",
    # Concrete classes
    "FunctionValidator",
    "FunctionRule",
    # Factory functions
    "create_rule",
]
