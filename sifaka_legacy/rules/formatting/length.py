"""
Length-based formatting rules for Sifaka.

This module provides rules for validating and enforcing text length constraints.

## Overview
The length validation rules help ensure that text meets specific length requirements
in terms of character count and word count. This is useful for enforcing content
guidelines, preventing excessively long or short responses, and maintaining
consistent formatting across documents.

## Components
- **LengthConfig**: Configuration for length validation
- **LengthValidator**: Base class for length validators
- **DefaultLengthValidator**: Default implementation of length validator
- **LengthRuleValidator**: Adapter for LengthValidator to Rule system
- **LengthRule**: Rule for validating text length
- **Factory Functions**: create_length_validator, create_length_rule

## Usage Examples
```python
from sifaka.rules.formatting.length import create_length_rule

# Create a length rule
rule = create_length_rule(
    min_chars=10,
    max_chars=100,
    min_words=2,
    max_words=20
)

# Validate text
result = rule.validate("This is a test") if rule else ""
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
print(f"Character count: {result.metadata['char_count']}")
print(f"Word count: {result.metadata['word_count']}")
```

## Error Handling
- Empty text handling through BaseValidator.handle_empty_text
- Validation errors with detailed messages and suggestions
- Safely executed operations with standardized error handling
- Caching for performance optimization
"""

from typing import Optional, Any, Union, cast, TypeVar, Callable, Dict
import time
from pydantic import Field, ConfigDict, field_validator, BaseModel, PrivateAttr
from sifaka.rules.base import Rule, RuleConfig, RuleResult, BaseValidator
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager, create_rule_state
from sifaka.utils.errors import ErrorResult, RuleError
from sifaka.utils.errors.safe_execution import safely_execute_component_operation

logger = get_logger(__name__)
__all__ = [
    "LengthConfig",
    "LengthValidator",
    "DefaultLengthValidator",
    "LengthRuleValidator",
    "LengthRule",
    "create_length_validator",
    "create_length_rule",
]

# Type variable for our rule operations
T = TypeVar("T")


# Create our own safely_execute_rule function that works properly with mypy
def safely_execute_rule(
    operation: Callable[[], T],
    component_name: str,
    default_result: Optional[Union[T, ErrorResult]] = None,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Union[T, ErrorResult]:
    """Safely execute a rule operation."""
    return safely_execute_component_operation(
        operation=operation,
        component_name=component_name,
        component_type="Rule",
        error_class=RuleError,
        default_result=default_result,
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )


class LengthConfig(BaseModel):
    """
    Configuration for text length validation.

    This configuration class defines the parameters for length validation,
    including character and word count constraints.

    Examples:
        ```python
        from sifaka.rules.formatting.length import LengthConfig

        # Create configuration
        config = LengthConfig(
            min_chars=10,
            max_chars=100,
            min_words=2,
            max_words=20
        )

        # Use with validator
        validator = DefaultLengthValidator(config)
        ```

    Attributes:
        min_chars: Minimum number of characters allowed (inclusive)
        max_chars: Maximum number of characters allowed (inclusive)
        min_words: Minimum number of words allowed (inclusive)
        max_words: Maximum number of words allowed (inclusive)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    min_chars: Optional[int] = Field(
        default=None, ge=0, description="Minimum number of characters allowed (inclusive)"
    )
    max_chars: Optional[int] = Field(
        default=None, ge=0, description="Maximum number of characters allowed (inclusive)"
    )
    min_words: Optional[int] = Field(
        default=None, ge=0, description="Minimum number of words allowed (inclusive)"
    )
    max_words: Optional[int] = Field(
        default=None, ge=0, description="Maximum number of words allowed (inclusive)"
    )

    @field_validator("max_chars")
    @classmethod
    def validate_max_chars(cls, v: Optional[int], info) -> Optional[int]:
        """Validate that max_chars is greater than min_chars if specified."""
        min_chars = info.data.get("min_chars")
        if v is not None and min_chars is not None and v < min_chars:
            raise ValueError("max_chars must be greater than or equal to min_chars")
        return v

    @field_validator("max_words")
    @classmethod
    def validate_max_words(cls, v: Optional[int], info) -> Optional[int]:
        """Validate that max_words is greater than min_words if specified."""
        min_words = info.data.get("min_words")
        if v is not None and min_words is not None and v < min_words:
            raise ValueError("max_words must be greater than or equal to min_words")
        return v


class LengthValidator(BaseValidator[str]):
    """
    Base class for text length validators.

    This abstract validator defines the interface for length validation
    but delegates the actual validation logic to concrete implementations.

    Lifecycle:
        1. Initialization: Set up with length constraints
        2. Validation: Apply constraints to input text
        3. Result: Return standardized validation results

    Examples:
        ```python
        from sifaka.rules.formatting.length import create_length_validator

        # Create a validator using the factory function
        validator = create_length_validator(
            min_chars=10,
            max_chars=100
        )

        # Validate text
        result = (validator.validate("This is a test")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, config: LengthConfig) -> None:
        """
        Initialize validator with a configuration.

        Args:
            config: Length validation configuration
        """
        super().__init__(validation_type=str)
        self.config = config

    def validate(self, text: str) -> RuleResult:
        """
        Validate text against length constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        empty_result = self.handle_empty_text(text)
        if empty_result is not None:
            # Ensure we return the correct type
            if isinstance(empty_result, RuleResult):
                return empty_result
            # If it's not a RuleResult, create one
            return RuleResult(
                passed=False,
                message="Empty input not allowed",
                metadata={"error": "Empty input not allowed"},
                score=0.0,
                issues=["Empty input not allowed"],
                suggestions=["Provide non-empty input"],
                processing_time_ms=0.0,
            )
        raise NotImplementedError("Subclasses must implement validate method")


class DefaultLengthValidator(LengthValidator):
    """
    Default implementation of text length validator.

    This validator implements the standard length validation logic,
    checking both character count and word count against configured constraints.

    Lifecycle:
        1. Initialization: Set up with length constraints
        2. Validation: Check text against character and word count constraints
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.length import LengthConfig, DefaultLengthValidator

        # Create configuration
        config = LengthConfig(min_chars=10, max_chars=100)

        # Create validator
        validator = DefaultLengthValidator(config)

        # Validate text
        result = (validator.validate("This is a test")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    _state_manager: "StateManager" = PrivateAttr()

    def __init__(self, config: LengthConfig) -> None:
        """
        Initialize validator with a configuration.

        Args:
            config: Length validation configuration
        """
        super().__init__(config)
        object.__setattr__(self, "_state_manager", create_rule_state())
        self._state_manager.update("config", config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("validation_count", 0)
        self._state_manager.update("error_count", 0)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    def validate(self, text: str) -> RuleResult:
        """
        Validate text against length constraints.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()
        validation_count = self._state_manager.get("validation_count", 0)
        self._state_manager.update("validation_count", validation_count + 1)

        def validation_operation() -> RuleResult:
            empty_result = self.handle_empty_text(text)
            if empty_result is not None:
                if isinstance(empty_result, RuleResult):
                    return empty_result
                # If it's not a RuleResult, create one
                return RuleResult(
                    passed=False,
                    message="Empty input not allowed",
                    metadata={"error": "Empty input not allowed"},
                    score=0.0,
                    issues=["Empty input not allowed"],
                    suggestions=["Provide non-empty input"],
                    processing_time_ms=0.0,
                )

            errors = []
            suggestions = []
            config = self._state_manager.get("config")
            char_count = len(text)
            if config.min_chars is not None and char_count < config.min_chars:
                error_msg = (
                    f"Text is too short: {char_count} characters (minimum {config.min_chars})"
                )
                errors.append(error_msg)
                suggestions.append(f"Add at least {config.min_chars - char_count} more characters")
            if config.max_chars is not None and char_count > config.max_chars:
                error_msg = (
                    f"Text is too long: {char_count} characters (maximum {config.max_chars})"
                )
                errors.append(error_msg)
                suggestions.append(f"Remove at least {char_count - config.max_chars} characters")
            word_count = len(text.split())
            if config.min_words is not None and word_count < config.min_words:
                error_msg = (
                    f"Text has too few words: {word_count} words (minimum {config.min_words})"
                )
                errors.append(error_msg)
                suggestions.append(f"Add at least {config.min_words - word_count} more words")
            if config.max_words is not None and word_count > config.max_words:
                error_msg = (
                    f"Text has too many words: {word_count} words (maximum {config.max_words})"
                )
                errors.append(error_msg)
                suggestions.append(f"Remove at least {word_count - config.max_words} words")
            result = RuleResult(
                passed=not errors,
                message=errors[0] if errors else "Text length validation successful",
                metadata={
                    "char_count": char_count,
                    "word_count": word_count,
                    "errors": errors,
                    "validator_type": self.__class__.__name__,
                    "validation_time": time.time(),
                },
                score=1.0 if not errors else 0.0,
                issues=errors,
                suggestions=suggestions,
                processing_time_ms=(time.time() - start_time),
            )
            return result

        # Call safely_execute_rule with positional arguments
        result = safely_execute_rule(
            validation_operation,  # operation (1st positional arg)
            self.__class__.__name__,  # component_name (2nd positional arg)
        )

        if isinstance(result, ErrorResult):
            error_count = self._state_manager.get("error_count", 0)
            self._state_manager.update("error_count", error_count + 1)
            return RuleResult(
                passed=False,
                message=f"Validation error: {getattr(result, 'error_message', 'Unknown error')}",
                metadata=getattr(result, "metadata", {}),
                score=0.0,
                issues=[f"Validation error: {getattr(result, 'error_message', 'Unknown error')}"],
                suggestions=["Check your input and try again"],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        if not result.passed:
            error_count = self._state_manager.get("error_count", 0)
            self._state_manager.update("error_count", error_count + 1)

        cache = self._state_manager.get("cache", {})
        if len(cache) < 100:
            cache_key = hash(text)
            cache[cache_key] = result
            self._state_manager.update("cache", cache)

        # Since we've checked that result is not an ErrorResult, it must be a RuleResult
        # or something compatible with RuleResult, so no need for a cast
        return result


class LengthRuleValidator:
    """
    Validator adapter that implements RuleValidator protocol for LengthValidator.

    This adapter wraps a LengthValidator to make it compatible with the
    Rule validation system by implementing the RuleValidator protocol.

    Lifecycle:
        1. Initialization: Wrap a LengthValidator
        2. Validation: Delegate to the wrapped validator
        3. Type Checking: Provide type information for validation

    Examples:
        ```python
        from sifaka.rules.formatting.length import create_length_validator, LengthRuleValidator

        # Create a validator
        validator = create_length_validator(min_chars=10, max_chars=100)

        # Create an adapter
        adapter = LengthRuleValidator(validator)

        # Use the adapter
        if (adapter.can_validate("This is a test"):
            result = (adapter.validate("This is a test")
        ```
    """

    def __init__(self, validator: LengthValidator) -> None:
        """
        Initialize with a LengthValidator.

        Args:
            validator: The validator to wrap
        """
        self.validator = validator

    def validate(self, output: str) -> RuleResult:
        """
        Validate the output using the wrapped validator.

        Args:
            output: The text to validate

        Returns:
            Validation result
        """
        # Get the result from the validator
        result = self.validator.validate(output)
        # Ensure we return a RuleResult
        if isinstance(result, RuleResult):
            return result
        # If it's not a RuleResult for some reason, create one
        return RuleResult(
            passed=False,
            message="Validation failed with unknown error",
            metadata={"error": "Unknown validation error"},
            score=0.0,
            issues=["Unknown validation error"],
            suggestions=["Check your input and try again"],
            processing_time_ms=0.0,
        )

    def can_validate(self, output: str) -> bool:
        """
        Check if this validator can validate the output.

        Args:
            output: The output to check

        Returns:
            True if this validator can validate the output, False otherwise
        """
        return isinstance(output, str)

    @property
    def validation_type(self) -> type:
        """
        Get the type this validator can validate.

        Returns:
            The type this validator can validate
        """
        return str


class LengthRule(Rule[str]):
    """
    Rule for validating text length constraints.

    This rule validates that text meets specified length requirements
    in terms of character count and word count.

    Lifecycle:
        1. Initialization: Set up with length constraints
        2. Validation: Check text against constraints
        3. Result: Return standardized validation results with metadata

    Examples:
        ```python
        from sifaka.rules.formatting.length import create_length_rule

        # Create a rule using the factory function
        rule = create_length_rule(
            min_chars=10,
            max_chars=100,
            min_words=2,
            max_words=20
        )

        # Validate text
        result = (rule.validate("This is a test")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")

        # Access metadata
        if not result.passed:
            print(f"Character count: {result.metadata['char_count']}")
            print(f"Word count: {result.metadata['word_count']}")
        ```
    """

    _state_manager: "StateManager" = PrivateAttr()

    def __init__(
        self,
        validator: LengthValidator,
        name: str = "length_rule",
        description: str = "Validates text length",
        config: Optional[RuleConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the length rule.

        Args:
            validator: The validator to use for length validation
            name: The name of the rule
            description: Description of the rule
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        validator_adapter = LengthRuleValidator(validator)
        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name, description=description, rule_id=kwargs.pop("rule_id", name), **kwargs
            ),
            validator=validator_adapter,
        )
        self._length_validator = validator
        self._state_manager.update("validator", validator)
        self._state_manager.update("validator_adapter", validator_adapter)
        if hasattr(validator, "config"):
            self._state_manager.update("config", validator.config)
        self._state_manager.set_metadata("rule_type", "length")
        self._state_manager.set_metadata("validator_type", validator.__class__.__name__)

    def warm_up(self) -> None:
        """
        Warm up the rule by initializing resources.

        This method ensures that the rule is properly initialized
        and ready to use, including loading any necessary resources.
        """
        if self._state_manager.get("initialized", False):
            logger.debug(f"Rule {self.name} already initialized")
            return

        def warm_up_operation() -> None:
            if not self._state_manager.get("cache"):
                self._state_manager.update("cache", {})
            self._state_manager.update("initialized", True)
            logger.info(f"Rule {self.name} initialized successfully")

        # Call safely_execute_rule with positional arguments
        _ = safely_execute_rule(
            warm_up_operation,  # operation (1st positional arg)
            self.name,  # component_name (2nd positional arg)
        )

    def cleanup(self) -> None:
        """
        Clean up resources used by the rule.

        This method ensures that any resources used by the rule
        are properly released when the rule is no longer needed.
        """

        def cleanup_operation() -> None:
            self._state_manager.update("cache", {})
            self._state_manager.update("initialized", False)
            self._state_manager.set_metadata("cleanup_time", time.time())
            logger.info(f"Rule {self.name} cleaned up successfully")

        # Call safely_execute_rule with positional arguments
        _ = safely_execute_rule(
            cleanup_operation,  # operation (1st positional arg)
            self.name,  # component_name (2nd positional arg)
        )

    def _create_default_validator(self) -> LengthRuleValidator:
        """
        Create a default validator adapter for this rule.

        This method is not used since we create the validator in __init__,
        but it's required by the abstract base class.

        Raises:
            NotImplementedError: Always raised since this method should not be called
        """
        raise NotImplementedError("LengthRule requires a validator to be passed in __init__")


def create_length_validator(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    **kwargs,
) -> LengthValidator:
    """
    Create a length validator with the specified constraints.

    This factory function creates a configured LengthValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        min_chars: Minimum number of characters allowed
        max_chars: Maximum number of characters allowed
        min_words: Minimum number of words allowed
        max_words: Maximum number of words allowed
        **kwargs: Additional configuration parameters including:
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation
            - params: Dictionary of additional parameters

    Returns:
        Configured LengthValidator

    Examples:
        ```python
        from sifaka.rules.formatting.length import create_length_validator

        # Create a basic validator
        validator = create_length_validator(min_chars=10, max_chars=100)

        # Create a validator with word count constraints
        validator = create_length_validator(
            min_chars=10,
            max_chars=100,
            min_words=2,
            max_words=20
        )

        # Create a validator with additional configuration
        validator = create_length_validator(
            min_chars=10,
            max_chars=100,
            priority=2,
            cache_size=100
        )
        ```

    Raises:
        ValueError: If max_chars < min_chars or max_words < min_words
    """
    config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            config_params[param] = kwargs.pop(param)
    config = LengthConfig(
        min_chars=min_chars, max_chars=max_chars, min_words=min_words, max_words=max_words, **kwargs
    )
    return DefaultLengthValidator(config)


def create_length_rule(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> LengthRule:
    """
    Create a length validation rule with the specified constraints.

    This factory function creates a configured LengthRule instance.
    It uses create_length_validator internally to create the validator.

    Args:
        min_chars: Minimum number of characters allowed
        max_chars: Maximum number of characters allowed
        min_words: Minimum number of words allowed
        max_words: Maximum number of words allowed
        rule_id: Unique identifier for the rule
        name: Name of the rule (defaults to rule_id or "length_rule")
        description: Description of the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation
            - params: Dictionary of additional parameters

    Returns:
        Configured LengthRule

    Examples:
        ```python
        from sifaka.rules.formatting.length import create_length_rule

        # Create a basic rule
        rule = create_length_rule(min_chars=10, max_chars=100)

        # Create a rule with word count constraints
        rule = create_length_rule(
            min_chars=10,
            max_chars=100,
            min_words=2,
            max_words=20,
            rule_id="comprehensive_length"
        )

        # Create a rule with additional configuration
        rule = create_length_rule(
            min_chars=10,
            max_chars=100,
            priority=2,
            cache_size=100,
            description="Validates text length between 10-100 characters",
            severity="warning",
            category="formatting",
            tags=["length", "formatting"]
        )
        ```

    Raises:
        ValueError: If max_chars < min_chars or max_words < min_words
    """

    def factory_operation() -> LengthRule:
        validator = create_length_validator(
            min_chars=min_chars, max_chars=max_chars, min_words=min_words, max_words=max_words
        )
        rule_name = name or rule_id or "length_rule"
        rule_description = description
        if rule_description is None:
            char_range = (
                f"{min_chars}-{max_chars}"
                if min_chars is not None or max_chars is not None
                else "any"
            )
            word_range = (
                f"{min_words}-{max_words}"
                if min_words is not None or max_words is not None
                else "any"
            )
            rule_description = f"Validates text length (chars: {char_range}, words: {word_range})"

        config = RuleConfig(
            name=rule_name, description=rule_description, rule_id=rule_id or rule_name, **kwargs
        )

        rule = LengthRule(
            validator=validator, name=rule_name, description=rule_description, config=config
        )
        return rule

    # Call safely_execute_rule with positional arguments
    result = safely_execute_rule(
        factory_operation,  # operation (1st positional arg)
        "create_length_rule",  # component_name (2nd positional arg)
    )

    if isinstance(result, ErrorResult):
        raise ValueError(
            f"Failed to create length rule: {getattr(result, 'error_message', 'Unknown error')}"
        )

    # Now since we know it's not an ErrorResult, it must be a LengthRule
    return result
