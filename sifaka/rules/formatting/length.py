"""
Length validation rules for text.

This module provides validators and rules for checking text length constraints,
including character count and word count validation. It follows the standard
Sifaka validation pattern with separate validator and rule components.

The module provides two main factory functions:
- create_length_validator(): Creates a standalone validator
- create_length_rule(): Creates a rule with a validator

Lifecycle:
    1. Configuration: Set up length constraints (min/max chars, min/max words)
    2. Validation: Apply constraints to input text
    3. Result: Return standardized validation results with detailed metadata

Usage Example:
    ```python
    from sifaka.rules.formatting.length import create_length_rule

    # Create a length rule using the factory function
    rule = create_length_rule(
        min_chars=10,
        max_chars=100,
        min_words=2,
        max_words=20,
        rule_id="length_constraint"
    )

    # Validate text
    result = rule.validate("This is a test.")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")

    # Access metadata
    if not result.passed:
        print(f"Character count: {result.metadata['char_count']}")
        print(f"Word count: {result.metadata['word_count']}")
        print(f"Errors: {result.metadata['errors']}")
    ```
"""

from typing import Dict, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

from sifaka.rules.base import Rule, RuleResult, BaseValidator, RuleConfig
from sifaka.utils import RuleState, create_rule_state


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
        """Initialize validator with a configuration.

        Args:
            config: Length validation configuration
        """
        super().__init__()
        self.config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text against length constraints.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

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

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text against length constraints.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        errors = []

        # Character length validation
        char_count = len(text)
        if self.config.min_chars is not None and char_count < self.config.min_chars:
            errors.append(
                f"Text is too short: {char_count} characters (minimum {self.config.min_chars})"
            )

        if self.config.max_chars is not None and char_count > self.config.max_chars:
            errors.append(
                f"Text is too long: {char_count} characters (maximum {self.config.max_chars})"
            )

        # Word count validation
        word_count = len(text.split())
        if self.config.min_words is not None and word_count < self.config.min_words:
            errors.append(
                f"Text has too few words: {word_count} words (minimum {self.config.min_words})"
            )

        if self.config.max_words is not None and word_count > self.config.max_words:
            errors.append(
                f"Text has too many words: {word_count} words (maximum {self.config.max_words})"
            )

        return RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Text length validation successful",
            metadata={
                "char_count": char_count,
                "word_count": word_count,
                "errors": errors,
            },
        )


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
        """Initialize with a LengthValidator."""
        self.validator = validator

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate the output using the wrapped validator."""
        return self.validator.validate(output, **kwargs)

    def can_validate(self, output: str) -> bool:
        """Check if this validator can validate the output."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type:
        """Get the type this validator can validate."""
        return str


class LengthRule(Rule):
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

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(
        self,
        validator: LengthValidator,
        name: str = "length_rule",
        description: str = "Validates text length",
        config: Optional[RuleConfig] = None,
        **kwargs,
    ):
        """Initialize the length rule.

        Args:
            validator: The validator to use for length validation
            name: The name of the rule
            description: Description of the rule
            config: Additional configuration for the rule
            **kwargs: Additional keyword arguments for the rule
        """
        # Initialize state
        state = self._state_manager.get_state()
        state.cache = {}

        # Store validator in state
        rule_id = kwargs.pop("rule_id", name)  # Extract rule_id if present, default to name
        state.cache["rule_id"] = rule_id
        state.cache["length_validator"] = validator

        # Initialize base class
        super().__init__(name=name, description=description, config=config, **kwargs)

    def _create_default_validator(self) -> LengthRuleValidator:
        """Create a default validator adapter for this rule."""
        # Get state
        state = self._state_manager.get_state()

        # Get validator from state
        length_validator = state.cache.get("length_validator")

        # Create and return adapter
        return LengthRuleValidator(length_validator)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Evaluate text against length constraints.

        Args:
            text: The text to evaluate
            **kwargs: Additional validation context

        Returns:
            RuleResult containing validation results
        """
        # Get state
        state = self._state_manager.get_state()

        # Get validator and rule_id from state
        length_validator = state.cache.get("length_validator")
        rule_id = state.cache.get("rule_id")

        # Validate text
        result = length_validator.validate(text, **kwargs)

        # Add rule_id to metadata
        return result.with_metadata(rule_id=rule_id)


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
            priority="HIGH",
            cache_size=100
        )
        ```

    Raises:
        ValueError: If max_chars < min_chars or max_words < min_words
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    config = LengthConfig(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        max_words=max_words,
        **rule_config_params,
    )

    return DefaultLengthValidator(config)


def create_length_rule(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    rule_id: Optional[str] = None,
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
        rule_id: Identifier for the rule (also used as name if provided)
        **kwargs: Additional keyword arguments including:
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation
            - params: Dictionary of additional parameters
            - description: Description of the rule

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
            priority="HIGH",
            cache_size=100,
            description="Validates text length between 10-100 characters"
        )
        ```

    Raises:
        ValueError: If max_chars < min_chars or max_words < min_words
    """
    # Create validator using the validator factory
    validator = create_length_validator(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        max_words=max_words,
    )

    # Extract rule config parameters
    priority = kwargs.pop("priority", None)
    cache_size = kwargs.pop("cache_size", None)
    cost = kwargs.pop("cost", None)
    params = kwargs.pop("params", {})

    # Create rule config
    config = None
    if any([priority, cache_size, cost, params]):
        config = RuleConfig(
            priority=priority,
            cache_size=cache_size,
            cost=cost,
            params=params,
        )

    # Use rule_id as name if provided, otherwise use "length_rule"
    name = rule_id if rule_id else "length_rule"

    # Extract description if provided
    description = kwargs.pop(
        "description",
        f"Validates text length (chars: {min_chars}-{max_chars}, words: {min_words}-{max_words})",
    )

    return LengthRule(
        validator=validator,
        name=name,
        description=description,
        config=config,
        rule_id=rule_id,
        **kwargs,
    )
