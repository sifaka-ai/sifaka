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

## Error Handling Patterns

Sifaka implements several patterns for robust error handling:

### 1. Empty Input Handling
All validators should handle empty inputs gracefully, typically by:
- Checking for empty input early in validation
- Returning a standardized empty input result
- Using the `handle_empty_text()` utility method

### 2. Type Safety
Validators enforce type safety through:
- Using the `can_validate()` method to check input types
- Exposing a `validation_type` property
- Properly handling unexpected input types

### 3. Exception Management
The framework uses a clear exception hierarchy:
- `ValidationError`: For issues during validation
- `ConfigurationError`: For issues with rule configuration
- Runtime exceptions should be caught and converted to meaningful validation results

### 4. Graceful Degradation
When unexpected conditions occur:
- Return a meaningful RuleResult rather than raising exceptions
- Include diagnostic information in the result metadata
- Log the issue for debugging

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
    """
    Base exception for validation errors.

    This exception should be raised when validation cannot be performed
    due to issues with the input or validation process.

    Examples:
        ```python
        def validate_number(value: str) -> bool:
            try:
                float(value)
                return True
            except ValueError:
                raise ValidationError(f"'{value}' is not a valid number")
        ```
    """
    pass


class ConfigurationError(Exception):
    """
    Base exception for configuration errors.

    This exception should be raised when a rule or validator
    is incorrectly configured.

    Examples:
        ```python
        class LengthValidator(BaseValidator[str]):
            def __init__(self, min_length: int, max_length: int):
                if min_length < 0:
                    raise ConfigurationError("min_length must be non-negative")
                if max_length < min_length:
                    raise ConfigurationError(
                        f"max_length ({max_length}) must be >= min_length ({min_length})"
                    )
                self.min_length = min_length
                self.max_length = max_length
        ```
    """
    pass


class RulePriority(Enum):
    """
    Priority levels for rule execution.

    Rules with higher priority are generally executed first.
    This allows critical validations to fail early in a chain.

    Examples:
        ```python
        # Create a rule with high priority
        config = RuleConfig(priority=RulePriority.HIGH)

        # Create a security rule with critical priority
        security_rule = create_security_rule(
            config=RuleConfig(priority=RulePriority.CRITICAL)
        )

        # Sort rules by priority
        sorted_rules = sorted(
            rules,
            key=lambda r: r.config.priority,
            reverse=True  # Higher priority first
        )
        ```
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

    Examples:
        ```python
        from sifaka.rules.base import Validatable
        from typing import List

        class ValidatableConfig(Validatable[bool]):
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value
                self._errors: List[str] = []

            def is_valid(self) -> bool:
                self._errors = []

                if not self.name:
                    self._errors.append("Name cannot be empty")

                if self.value < 0:
                    self._errors.append("Value must be non-negative")

                return len(self._errors) == 0

            def validate(self) -> None:
                if not self.is_valid():
                    error_msg = "; ".join(self._errors)
                    raise ValidationError(f"Invalid configuration: {error_msg}")

            @property
            def validation_errors(self) -> List[str]:
                return self._errors.copy()
        ```
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

    ## Lifecycle

    1. **Initialization**: Set up with any necessary configuration
       - Initialize with validation parameters
       - Validate configuration options
       - Set up any required resources

    2. **Validation**: Process input and apply validation logic
       - Check input type compatibility
       - Handle empty or null inputs
       - Apply core validation logic
       - Create appropriate result object

    3. **Result**: Return standardized validation results
       - Return RuleResult with passed/failed status
       - Include descriptive message
       - Add relevant metadata

    ## Error Handling

    Validators should implement these error handling patterns:

    - **Type Checking**: Validate input types with `can_validate()`
    - **Empty Input**: Handle empty inputs gracefully
    - **Configuration Errors**: Validate configuration at initialization time
    - **Runtime Errors**: Catch and convert to appropriate RuleResults
    - **Performance Issues**: Implement timeouts or circuit breakers for expensive validations

    ## Examples

    ```python
    from sifaka.rules.base import RuleValidator, RuleResult
    import re

    class RegexValidator(RuleValidator[str]):
        def __init__(self, pattern: str):
            try:
                self.pattern = pattern
                self.regex = re.compile(pattern)
            except re.error as e:
                raise ConfigurationError(f"Invalid regex pattern: {e}")

        def validate(self, text: str, **kwargs) -> RuleResult:
            # Handle empty text
            if not text.strip():
                return RuleResult(
                    passed=True,
                    message="Empty text validation skipped",
                    metadata={"reason": "empty_input"}
                )

            # Apply validation
            try:
                is_match = bool(self.regex.search(text))
                return RuleResult(
                    passed=is_match,
                    message=f"Text {'matches' if is_match else 'does not match'} pattern",
                    metadata={
                        "pattern": self.pattern,
                        "text_length": len(text)
                    }
                )
            except Exception as e:
                # Handle unexpected errors gracefully
                logger.error(f"Regex validation error: {e}")
                return RuleResult(
                    passed=False,
                    message=f"Validation error: {str(e)}",
                    metadata={"error_type": type(e).__name__}
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

    ## Lifecycle

    1. **Initialization**: Set up validator configuration
       - Initialize with validation parameters
       - Validate configuration values
       - Set up any required resources

    2. **Validation Request**: Receive input to validate
       - Check input compatibility
       - Handle special cases (e.g., empty text)

    3. **Input Processing**: Apply validation logic to input
       - Apply core validation logic
       - Handle errors gracefully
       - Calculate validation metrics

    4. **Result Creation**: Return standardized validation results
       - Create RuleResult with appropriate status
       - Include descriptive message
       - Add relevant metadata

    ## Error Handling

    BaseValidator implements these error handling patterns:

    - **Empty Text Handling**: The `handle_empty_text()` method provides standardized
      handling for empty inputs.

    - **Type Checking**: The `can_validate()` method checks input compatibility.

    - **Safe Validation**: Subclasses should implement safe validation that catches
      and properly handles unexpected errors.

    ## Examples

    ```python
    from sifaka.rules.base import BaseValidator, RuleResult

    class LengthValidator(BaseValidator[str]):
        def __init__(self, min_length: int = 0, max_length: Optional[int] = None):
            # Validate configuration
            if min_length < 0:
                raise ConfigurationError("min_length must be non-negative")
            if max_length is not None and max_length < min_length:
                raise ConfigurationError(
                    f"max_length ({max_length}) must be >= min_length ({min_length})"
                )

            self.min_length = min_length
            self.max_length = max_length

        def validate(self, text: str, **kwargs) -> RuleResult:
            # Handle empty text first
            empty_result = self.handle_empty_text(text)
            if empty_result:
                return empty_result

            # Safe validation with try/except
            try:
                # Calculate text length
                text_length = len(text)

                # Check minimum length
                if text_length < self.min_length:
                    return RuleResult(
                        passed=False,
                        message=f"Text length ({text_length}) is below minimum ({self.min_length})",
                        metadata={
                            "text_length": text_length,
                            "min_length": self.min_length,
                            "max_length": self.max_length
                        }
                    )

                # Check maximum length if specified
                if self.max_length is not None and text_length > self.max_length:
                    return RuleResult(
                        passed=False,
                        message=f"Text length ({text_length}) exceeds maximum ({self.max_length})",
                        metadata={
                            "text_length": text_length,
                            "min_length": self.min_length,
                            "max_length": self.max_length
                        }
                    )

                # Validation passed
                return RuleResult(
                    passed=True,
                    message=(
                        f"Text length ({text_length}) is within limits " +
                        f"(min: {self.min_length}" +
                        (f", max: {self.max_length}" if self.max_length is not None else "")
                        + ")"
                    ),
                    metadata={
                        "text_length": text_length,
                        "min_length": self.min_length,
                        "max_length": self.max_length
                    }
                )
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Length validation error: {e}")
                return RuleResult(
                    passed=False,
                    message=f"Validation error: {str(e)}",
                    metadata={"error_type": type(e).__name__}
                )

        # Create factory function
        @staticmethod
        def create(min_length: int = 0, max_length: Optional[int] = None) -> "LengthValidator":
            return LengthValidator(min_length=min_length, max_length=max_length)
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

        Examples:
            ```python
            # Basic validation
            validator = EmailValidator()
            result = validator.validate("user@example.com")

            # Validation with context
            validator = SentimentValidator()
            result = validator.validate(
                "I love this product!",
                context={"domain": "product_review"}
            )

            # Error handling
            try:
                validator.validate(123)  # Wrong type
            except Exception as e:
                print(f"Validation error: {e}")
            ```
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

        Examples:
            ```python
            def validate(self, text: str, **kwargs) -> RuleResult:
                # Handle empty text first
                empty_result = self.handle_empty_text(text)
                if empty_result:
                    return empty_result

                # Proceed with normal validation
                # ...
            ```
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

        Examples:
            ```python
            # Check if validator can handle input
            if validator.can_validate(input_data):
                result = validator.validate(input_data)
            else:
                print(f"Validator cannot handle {type(input_data).__name__}")
                # Find a compatible validator or convert the input
            ```
        """
        return isinstance(output, self.validation_type)

    @property
    def validation_type(self) -> type:
        """
        Get the type this validator can validate.

        Returns:
            The type this validator can validate

        Examples:
            ```python
            # Check if validator is compatible with input type
            if isinstance(input_data, validator.validation_type):
                result = validator.validate(input_data)
            else:
                print(f"Input type {type(input_data).__name__} not compatible with "
                      f"validator type {validator.validation_type.__name__}")
            ```
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

    ## Lifecycle

    1. **Initialization**: Set up handler configuration
       - Initialize with handler options
       - Configure logging or monitoring

    2. **Result Reception**: Receive validation results
       - Get result from rule validation
       - Check if handler can process the result

    3. **Processing**: Apply handler logic to results
       - Apply handler-specific logic
       - Log or store results
       - Trigger actions based on result

    4. **Continuation Decision**: Determine if validation should continue
       - Decide based on result and configuration
       - Return boolean indicating whether to continue

    ## Error Handling

    Handlers should implement these error handling patterns:

    - **Type Checking**: Verify result compatibility with `can_handle()`
    - **Safe Processing**: Handle results without raising exceptions
    - **Logging**: Record significant events and errors

    ## Examples

    ```python
    from sifaka.rules.base import RuleResultHandler, RuleResult
    import logging

    class LoggingHandler(RuleResultHandler[RuleResult]):
        def __init__(self, logger=None, log_level=logging.INFO, stop_on_failure=False):
            self.logger = logger or logging.getLogger(__name__)
            self.log_level = log_level
            self.stop_on_failure = stop_on_failure

        def handle_result(self, result: RuleResult) -> None:
            # Format the log message
            status = "PASSED" if result.passed else "FAILED"
            message = f"Validation {status}: {result.message}"

            # Include metadata for failures or if debug level
            if not result.passed or self.log_level <= logging.DEBUG:
                metadata_str = ", ".join(f"{k}={v}" for k, v in result.metadata.items())
                message += f" [{metadata_str}]"

            # Log the result
            self.logger.log(self.log_level, message)

            # Additional actions for failures
            if not result.passed and self.log_level <= logging.WARNING:
                self.logger.warning("Validation failure detected!")

        def should_continue(self, result: RuleResult) -> bool:
            # Continue unless configured to stop on failure and result failed
            return not (self.stop_on_failure and not result.passed)

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

    ## Lifecycle

    1. **Creation**: Instantiated with validation outcome
       - Set passed/failed status
       - Provide descriptive message
       - Include relevant metadata

    2. **Usage**: Accessed by rules and handlers
       - Check validation status
       - Process result message
       - Analyze metadata

    3. **Augmentation**: New instances created with additional metadata
       - Create enhanced results with with_metadata()
       - Add rule-specific information
       - Include performance metrics

    ## Examples

    ```python
    # Create a successful validation result
    result = RuleResult(
        passed=True,
        message="Text meets all requirements",
        metadata={"char_count": 150, "word_count": 23},
        score=0.95
    )

    # Create a failed validation result
    result = RuleResult(
        passed=False,
        message="Text contains prohibited content",
        metadata={
            "prohibited_terms": ["xyz", "abc"],
            "severity": "high"
        }
    )

    # Access properties
    if result.passed:
        print(f"Validation passed: {result.message}")
    else:
        print(f"Validation failed: {result.message}")
        print(f"Details: {result.metadata}")

    # Use as boolean (will be True if passed)
    if result:
        print("Validation passed!")
    else:
        print("Validation failed!")

    # Create a new result with additional metadata
    enhanced_result = result.with_metadata(
        rule_id="content_rule",
        execution_time_ms=42
    )

    # Chain metadata additions
    final_result = result.with_metadata(rule_id="content_rule").with_metadata(priority="high")
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

        Examples:
            ```python
            # Check for failure
            result = rule.validate(text)
            if result.failed:
                print(f"Validation failed: {result.message}")
                # Handle failure case
            ```
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
