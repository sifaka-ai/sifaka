"""
Template for standardized Sifaka rule implementation.

This template provides the standard structure that all rule implementations
should follow to ensure consistency across the Sifaka framework.

Usage Example:
    from sifaka.rules.my_domain import create_my_rule

    # Create a rule using the factory function
    my_rule = create_my_rule(
        param1=value1,
        param2=value2,
        rule_id="my_custom_rule"
    )

    # Validate text
    result = my_rule.validate("This is a test.")
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
)


# Define type variables
T = TypeVar("T")  # Input type
V = TypeVar("V", bound=BaseValidator)  # Validator type


class RuleNameConfig(BaseModel):
    """
    Configuration for rule_name validation.

    All configuration parameters should be defined here
    with proper typing, validation, and documentation.

    Attributes:
        param1: Description of parameter 1
        param2: Description of parameter 2
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    param1: type = Field(
        default=default_value,
        description="Description of parameter 1",
        # Add validation constraints as needed
    )

    param2: type = Field(
        default=default_value,
        description="Description of parameter 2",
        # Add validation constraints as needed
    )

    # Add field validators if needed
    # @field_validator("param_name")
    # @classmethod
    # def validate_param_name(cls, v: type, info: Any) -> type:
    #     """Validate parameter."""
    #     if validation_condition:
    #         raise ValueError("Validation error message")
    #     return v


class RuleNameValidator(BaseValidator[T]):
    """
    Validator for rule_name.

    This validator implements the specific validation logic
    for this rule type.

    Lifecycle:
    1. Initialization: Set up with configuration
    2. Validation: Process input and apply validation logic
    3. Result: Return standardized validation results
    """

    def __init__(self, config: RuleNameConfig):
        """
        Initialize with configuration.

        Args:
            config: Configuration for validation
        """
        super().__init__()
        self._config = config

    @property
    def config(self) -> RuleNameConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate the input.

        Args:
            input_value: Input to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Handle empty input (for string inputs)
        if isinstance(input_value, str):
            empty_result = self.handle_empty_text(input_value)
            if empty_result:
                return empty_result

        try:
            # Implement validation logic here
            is_valid = self._validate_input(input_value)

            if is_valid:
                return RuleResult(
                    passed=True,
                    message="Validation successful",
                    metadata={
                        "input_type": type(input_value).__name__,
                        # Add relevant metadata
                    },
                )
            else:
                return RuleResult(
                    passed=False,
                    message="Validation failed: detailed reason",
                    metadata={
                        "input_type": type(input_value).__name__,
                        "errors": ["Specific error details"],
                        # Add relevant metadata
                    },
                )
        except Exception as e:
            # Handle exceptions properly
            return RuleResult(
                passed=False,
                message=f"Validation error: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "input_type": type(input_value).__name__,
                },
            )

    def _validate_input(self, input_value: T) -> bool:
        """
        Internal method to implement validation logic.

        Args:
            input_value: Input to validate

        Returns:
            True if validation passes, False otherwise

        Raises:
            Exception: Any validation-specific exceptions
        """
        # Implement specific validation logic
        # Return True if validation passes, False otherwise
        return True


class RuleNameRule(Rule[T, RuleResult, RuleNameValidator]):
    """
    Rule for rule_name validation.

    This rule delegates validation to RuleNameValidator.

    Lifecycle:
    1. Initialization: Set up with configuration
    2. Validation: Delegate to validator
    3. Result: Return validation result with rule metadata

    Examples:
        ```python
        from sifaka.rules.domain import RuleNameRule

        rule = RuleNameRule(
            param1=value1,
            param2=value2,
            name="custom_name",
            description="Custom description"
        )

        result = rule.validate("Input to validate")
        ```
    """

    def __init__(
        self,
        param1: type = default_value,
        param2: type = default_value,
        rule_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize rule with configuration.

        Args:
            param1: Parameter 1
            param2: Parameter 2
            rule_id: Unique identifier for the rule
            name: Name of the rule
            description: Description of the rule
            **kwargs: Additional rule configuration
        """
        # Set default name and description if not provided
        if name is None:
            name = "rule_name_rule"
        if description is None:
            description = "Validates input using rule_name criteria"

        # Create rule configuration
        rule_config = RuleConfig(
            params={
                "param1": param1,
                "param2": param2,
            }
        )

        # Store essential attributes
        self._rule_id = rule_id or name

        # Initialize with base class
        super().__init__(
            name=name,
            description=description,
            config=rule_config,
            **kwargs,
        )

    def _create_default_validator(self) -> RuleNameValidator:
        """
        Create default validator from configuration.

        Returns:
            Configured validator
        """
        # Extract configuration from rule config
        params = self.config.params

        # Create and return validator configuration
        config = RuleNameConfig(
            param1=params.get("param1", default_value),
            param2=params.get("param2", default_value),
        )

        return RuleNameValidator(config)

    @property
    def rule_id(self) -> str:
        """Get the rule ID."""
        return self._rule_id

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate input.

        Args:
            input_value: Input to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Delegate to validator
        result = self._validator.validate(input_value, **kwargs)

        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._rule_id)


def create_rule_name_validator(
    param1: type = default_value,
    param2: type = default_value,
    **kwargs,
) -> RuleNameValidator:
    """
    Create a rule_name validator.

    This factory function creates a configured validator.
    It's useful when you need a validator without creating a full rule.

    Args:
        param1: Parameter 1
        param2: Parameter 2
        **kwargs: Additional validator configuration

    Returns:
        Configured validator
    """
    # Create configuration
    config = RuleNameConfig(
        param1=param1,
        param2=param2,
        **{k: v for k, v in kwargs.items() if k in RuleNameConfig.__fields__},
    )

    # Create and return validator
    return RuleNameValidator(config)


def create_rule_name_rule(
    param1: type = default_value,
    param2: type = default_value,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> RuleNameRule:
    """
    Create a rule_name rule.

    This factory function creates a configured rule.

    Args:
        param1: Parameter 1
        param2: Parameter 2
        rule_id: Unique identifier for the rule
        name: Name of the rule
        description: Description of the rule
        **kwargs: Additional rule configuration

    Returns:
        Configured rule

    Examples:
        ```python
        from sifaka.rules.domain import create_rule_name_rule

        rule = create_rule_name_rule(
            param1=value1,
            param2=value2,
            rule_id="custom_rule"
        )

        result = rule.validate("Input to validate")
        ```
    """
    return RuleNameRule(
        param1=param1,
        param2=param2,
        rule_id=rule_id,
        name=name,
        description=description,
        **kwargs,
    )


# Export public components
__all__ = [
    # Config classes
    "RuleNameConfig",
    # Validator classes
    "RuleNameValidator",
    # Rule classes
    "RuleNameRule",
    # Factory functions
    "create_rule_name_validator",
    "create_rule_name_rule",
]