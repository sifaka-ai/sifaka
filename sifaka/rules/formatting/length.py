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
result = rule.validate("This is a test")
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

from typing import Optional, Dict, Any
import time
from pydantic import Field, ConfigDict, field_validator, BaseModel, PrivateAttr

from sifaka.rules.base import Rule, RuleConfig, RuleResult, BaseValidator
from sifaka.utils.errors.safe_execution import safely_execute_rule, RuleError
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager, create_rule_state
from sifaka.utils.patterns import compile_pattern, match_pattern, find_patterns

logger = get_logger(__name__)

__all__ = [
    # Config classes
    "LengthConfig",
    # Validator classes
    "LengthValidator",
    "DefaultLengthValidator",
    "LengthRuleValidator",
    # Rule classes
    "LengthRule",
    # Factory functions
    "create_length_validator",
    "create_length_rule",
]


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
        default=None,
        ge=0,
        description="Minimum number of characters allowed (inclusive)",
    )
    max_chars: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of characters allowed (inclusive)",
    )
    min_words: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of words allowed (inclusive)",
    )
    max_words: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of words allowed (inclusive)",
    )

    @field_validator("max_chars")
    @classmethod
    def validate_max_chars(cls, v: Optional[int], info) -> Optional[int]:
        """Validate that max_chars is greater than min_chars if specified."""
        if (
            v is not None
            and info.data.get("min_chars") is not None
            and v < info.data.get("min_chars")
        ):
            raise ValueError("max_chars must be greater than or equal to min_chars")
        return v

    @field_validator("max_words")
    @classmethod
    def validate_max_words(cls, v: Optional[int], info) -> Optional[int]:
        """Validate that max_words is greater than min_words if specified."""
        if (
            v is not None
            and info.data.get("min_words") is not None
            and v < info.data.get("min_words")
        ):
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
        result = validator.validate("This is a test")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, config: LengthConfig):
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
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

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
        result = validator.validate("This is a test")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    # Declare the private attribute but don't use default_factory
    _state_manager: "StateManager" = PrivateAttr()

    def __init__(self, config: LengthConfig):
        """
        Initialize validator with a configuration.

        Args:
            config: Length validation configuration
        """
        super().__init__(config)

        # Initialize the state manager explicitly for Pydantic v2 compatibility
        object.__setattr__(self, "_state_manager", create_rule_state())

        # Initialize state
        self._state_manager.update("config", config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("validation_count", 0)
        self._state_manager.update("error_count", 0)
        self._state_manager.update("cache", {})

        # Set metadata
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

        # Update validation count
        validation_count = self._state_manager.get("validation_count", 0)
        self._state_manager.update("validation_count", validation_count + 1)

        # Define the validation operation
        def validation_operation():
            # Handle empty text
            empty_result = self.handle_empty_text(text)
            if empty_result:
                return empty_result

            errors = []
            suggestions = []

            # Get config from state
            config = self._state_manager.get("config")

            # Character length validation
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

            # Word count validation
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

            # Create result
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
                processing_time_ms=time.time() - start_time,
            )

            return result

        # Use the standardized safely_execute_rule function
        result = safely_execute_rule(
            operation=validation_operation,
            component_name=self.__class__.__name__,
            additional_metadata={"text_length": len(text)},
        )

        # Update error count if validation failed
        if not getattr(result, "passed", False):
            error_count = self._state_manager.get("error_count", 0)
            self._state_manager.update("error_count", error_count + 1)

        # Update cache
        cache = self._state_manager.get("cache", {})
        if len(cache) < 100:  # Limit cache size
            cache_key = hash(text)
            cache[cache_key] = result
            self._state_manager.update("cache", cache)

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
        if adapter.can_validate("This is a test"):
            result = adapter.validate("This is a test")
        ```
    """

    def __init__(self, validator: LengthValidator):
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
        return self.validator.validate(output)

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
        result = rule.validate("This is a test")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")

        # Access metadata
        if not result.passed:
            print(f"Character count: {result.metadata['char_count']}")
            print(f"Word count: {result.metadata['word_count']}")
        ```
    """

    # Declare the private attribute but don't use default_factory
    _state_manager: "StateManager" = PrivateAttr()

    def __init__(
        self,
        validator: LengthValidator,
        name: str = "length_rule",
        description: str = "Validates text length",
        config: Optional[RuleConfig] = None,
        **kwargs,
    ):
        """
        Initialize the length rule.

        Args:
            validator: The validator to use for length validation
            name: The name of the rule
            description: Description of the rule
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        # Create the validator adapter
        validator_adapter = LengthRuleValidator(validator)

        # Initialize base class with the validator adapter
        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name, description=description, rule_id=kwargs.pop("rule_id", name), **kwargs
            ),
            validator=validator_adapter,
        )

        # Store the validator for reference
        self._length_validator = validator

        # Initialize additional state (state_manager is already initialized in super().__init__)
        self._state_manager.update("validator", validator)
        self._state_manager.update("validator_adapter", validator_adapter)

        # Store the validator's config in the state manager
        if hasattr(validator, "config"):
            self._state_manager.update("config", validator.config)

        # Set metadata
        self._state_manager.set_metadata("rule_type", "length")
        self._state_manager.set_metadata("validator_type", validator.__class__.__name__)

    def warm_up(self) -> None:
        """
        Warm up the rule by initializing resources.

        This method ensures that the rule is properly initialized
        and ready to use, including loading any necessary resources.
        """
        # Check if already initialized
        if self._state_manager.get("initialized", False):
            logger.debug(f"Rule {self.name} already initialized")
            return

        # Define the warm-up operation
        def warm_up_operation():
            # Initialize cache if needed
            if not self._state_manager.get("cache"):
                self._state_manager.update("cache", {})

            # Mark as initialized
            self._state_manager.update("initialized", True)
            logger.info(f"Rule {self.name} initialized successfully")

        # Use the standardized safely_execute_rule function
        safely_execute_rule(
            operation=warm_up_operation,
            component_name=self.name,
            additional_metadata={"method": "warm_up"},
        )

    def cleanup(self) -> None:
        """
        Clean up resources used by the rule.

        This method ensures that any resources used by the rule
        are properly released when the rule is no longer needed.
        """

        # Define the cleanup operation
        def cleanup_operation():
            # Clear cache
            self._state_manager.update("cache", {})

            # Mark as not initialized
            self._state_manager.update("initialized", False)
            self._state_manager.set_metadata("cleanup_time", time.time())

            logger.info(f"Rule {self.name} cleaned up successfully")

        # Use the standardized safely_execute_rule function
        safely_execute_rule(
            operation=cleanup_operation,
            component_name=self.name,
            additional_metadata={"method": "cleanup"},
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
    # Extract config parameters from kwargs
    config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            config_params[param] = kwargs.pop(param)

    # Create the configuration
    config = LengthConfig(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        max_words=max_words,
        **kwargs,
    )

    # Create and return the validator
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

    # Define the factory operation
    def factory_operation():
        # Create validator using the validator factory
        validator = create_length_validator(
            min_chars=min_chars,
            max_chars=max_chars,
            min_words=min_words,
            max_words=max_words,
        )

        # Determine rule name
        rule_name = name or rule_id or "length_rule"

        # Create rule description if not provided
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

        # Create rule config
        config = RuleConfig(
            name=rule_name,
            description=rule_description,
            rule_id=rule_id or rule_name,
            **kwargs,
        )

        # Create and return the rule
        rule = LengthRule(
            validator=validator,
            name=rule_name,
            description=rule_description,
            config=config,
        )

        # Initialize the rule
        rule.warm_up()

        return rule

    # Use the standardized safely_execute_rule function
    return safely_execute_rule(
        operation=factory_operation,
        component_name="create_length_rule",
        additional_metadata={
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_words": min_words,
            "max_words": max_words,
        },
    )
