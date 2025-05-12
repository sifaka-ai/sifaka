"""
Base Classes for Sifaka Rules

This module defines the core architecture for rules in Sifaka,
providing the foundation for all validation components in the framework.

## Overview
The rules system is a key validation component in Sifaka that enables
content and format validation across the framework. It follows a delegation pattern
where high-level rule containers define what to validate, while validators
implement the actual validation logic.

## Components
- Rule: Base class for all rules
- RuleConfig: Configuration for rules
- RuleResult: Result from rule validation
- FunctionRule: Rule that uses a function for validation

## Architecture
The rules system follows a delegation pattern:
1. Rules are high-level containers that define what to validate
2. Validators implement the actual validation logic
3. Rules delegate validation work to their validators

This separation of concerns allows for:
- Reusing validation logic across different rules
- Testing validation logic independently
- Extending the framework with custom validators

## Usage Examples
```python
from sifaka.rules.base import FunctionRule, RuleResult
from sifaka.utils.config.rules import RuleConfig

# Create a simple function-based rule
def validate_length(text: str) -> RuleResult:
    if len(text) < 10:
        return RuleResult(
            passed=False,
            message="Text is too short",
            issues=["Text must be at least 10 characters"]
        )
    return RuleResult(passed=True, message="Text length is valid")

# Create a rule with the validation function
rule = FunctionRule(
    name="length_rule",
    func=validate_length,
    description="Validates text length",
    config=RuleConfig(severity="warning", category="formatting")
)

# Validate some text
result = rule.validate("Hello")  # Will fail
print(result.passed)  # False
print(result.message)  # "Text is too short"

result = rule.validate("Hello, world! This is a longer text.")  # Will pass
print(result.passed)  # True
```

## Error Handling
Rules use a standardized error handling approach:
- Validation errors are captured and returned as RuleResult objects
- System errors are logged and can be tracked via the rule's state manager
- The safely_execute_rule utility is used to ensure consistent error handling

## State Management
Rules use the StateManager from sifaka.utils.state for managing internal state:
- Performance metrics are tracked when enabled
- Error occurrences are recorded when enabled
- Rule metadata is stored for debugging and monitoring
"""

from abc import abstractmethod
from typing import (
    Any,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
)
import time
from datetime import datetime
from pydantic import Field, ConfigDict, PrivateAttr

from sifaka.core.base import (
    BaseComponent,
    BaseConfig,
    BaseResult,
)
from sifaka.utils.errors.safe_execution import safely_execute_rule
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_rule_state, StateManager

from .validators import RuleValidator, FunctionValidator, BaseValidator

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
V = TypeVar("V", bound=BaseValidator)  # Validator type


class RuleConfig(BaseConfig):
    """
    Configuration for rules.

    This class extends BaseConfig to add rule-specific configuration options.

    Attributes:
        severity: Severity level for rule violations (error, warning, info)
        category: Category of the rule (formatting, content, etc.)
        tags: List of tags for categorizing the rule
        track_performance: Whether to track performance metrics
        track_errors: Whether to track error occurrences
        rule_id: Unique identifier for the rule
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: str = Field(
        default="error",
        description="Severity level for rule violations",
    )
    category: str = Field(
        default="general",
        description="Category of the rule",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorizing the rule",
    )
    track_performance: bool = Field(
        default=True,
        description="Whether to track performance metrics",
    )
    track_errors: bool = Field(
        default=True,
        description="Whether to track error occurrences",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the rule",
    )


from ..core.results import RuleResult


class Rule(BaseComponent[T, RuleResult], Generic[T]):
    """
    Base class for all rules.

    Rules are high-level components that define what to validate.
    They delegate the actual validation work to validators.

    ## Architecture
    Rules follow a component-based architecture:
    - They extend BaseComponent for consistent interfaces
    - They use validators for the actual validation logic
    - They manage state using StateManager
    - They produce standardized RuleResult objects

    ## Lifecycle
    1. Initialization: Set up with validator and configuration
       - Create default validator if none provided
       - Initialize state manager with metadata
    2. Validation: Delegate to validator and process results
       - Validate input type compatibility
       - Handle empty inputs
       - Delegate to validator for actual validation
       - Process and standardize results
    3. Result: Return standardized validation results with metadata
       - Add rule metadata (severity, category, tags)
       - Track performance metrics
       - Record errors if they occur

    ## Error Handling
    - Input validation errors return a failed RuleResult
    - Validator compatibility errors return a failed RuleResult
    - Implementation errors are caught and converted to failed RuleResults
    - All errors are logged and can be tracked via statistics

    ## Examples
    ```python
    from sifaka.rules.base import Rule, RuleResult
    from sifaka.rules.validators import RuleValidator

    # Create a custom validator
    class LengthValidator(RuleValidator[str]):
        def validate(self, input: str) -> RuleResult:
            if len(input) < 10:
                return RuleResult(
                    passed=False,
                    message="Text is too short",
                    issues=["Text must be at least 10 characters"]
                )
            return RuleResult(passed=True, message="Text length is valid")

        def can_validate(self, input: Any) -> bool:
            return isinstance(input, str)

    # Create a custom rule
    class LengthRule(Rule[str]):
        def _create_default_validator(self) -> RuleValidator[str]:
            return LengthValidator()

    # Use the rule
    rule = LengthRule(
        name="length_rule",
        description="Validates text length"
    )
    result = rule.validate("Hello")  # Will fail
    ```
    """

    # Declare the private attribute but don't use default_factory
    _state_manager: StateManager = PrivateAttr()

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[RuleConfig] = None,
        validator: Optional[RuleValidator[T]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the rule.

        Creates a new rule with the specified name, description, configuration,
        and validator. If no validator is provided, the _create_default_validator
        method is called to create one.

        Args:
            name: Name of the rule
            description: Description of the rule
            config: Configuration for the rule
            validator: Validator to use for validation
            **kwargs: Additional keyword arguments for configuration including:
                - severity: Severity level for rule violations
                - category: Category of the rule
                - tags: List of tags for categorizing the rule
                - priority: Priority level for validation
                - cache_size: Size of the validation cache
                - cost: Computational cost of validation

        Raises:
            ValueError: If the rule cannot be initialized properly
            TypeError: If the validator is not compatible with the rule

        Example:
            ```python
            from sifaka.rules.base import Rule
            from sifaka.rules.validators import RuleValidator

            # Create a custom validator
            validator = CustomValidator()

            # Create a rule with the validator
            rule = Rule(
                name="custom_rule",
                description="Custom validation rule",
                validator=validator,
                severity="warning",
                category="content"
            )
            ```
        """
        super().__init__(name, description, config or RuleConfig(**kwargs))
        self._validator = validator or self._create_default_validator()

        # Initialize the state manager explicitly for Pydantic v2 compatibility
        object.__setattr__(self, "_state_manager", create_rule_state())

        # Initialize rule-specific state
        self._state_manager.set_metadata("rule_type", self.__class__.__name__)
        self._state_manager.set_metadata("rule_id", getattr(config, "rule_id", None) or name)
        self._state_manager.set_metadata(
            "validator_type",
            (
                self._validator.__class__.__name__
                if hasattr(self._validator, "__class__")
                else "Unknown"
            ),
        )

    @abstractmethod
    def _create_default_validator(self) -> RuleValidator[T]:
        """
        Create the default validator for this rule.

        Returns:
            Default validator instance
        """
        ...

    def model_validate(self, input: T) -> RuleResult:
        """
        Validate the input using the rule's validator.

        Args:
            input: The input to validate

        Returns:
            Validation result
        """
        start_time = time.time()
        rule_id = self._state_manager.get_metadata("rule_id")

        # Validate input
        if not self.validate_input(input):
            result = RuleResult(
                passed=False,
                message="Invalid input",
                metadata={
                    "error_type": "invalid_input",
                    "rule_id": rule_id,
                    "rule_type": self.__class__.__name__,
                },
                score=0.0,
                issues=["Invalid input type"],
                suggestions=["Provide valid input"],
                processing_time_ms=time.time() - start_time,
                rule_id=rule_id,
                severity=self.config.severity,
                category=self.config.category,
                tags=self.config.tags,
            )
            self.update_statistics(result)
            return result

        # Handle empty input
        empty_result = self.handle_empty_input(input)
        if empty_result:
            result = (
                empty_result.with_metadata(
                    processing_time_ms=time.time() - start_time,
                    rule_id=rule_id,
                    rule_type=self.__class__.__name__,
                )
                .with_rule_id(rule_id)
                .with_severity(self.config.severity)
                .with_category(self.config.category)
                .with_tags(self.config.tags)
            )
            self.update_statistics(result)
            return result

        # Check if validator can validate the input
        if not self._validator.can_validate(input):
            result = RuleResult(
                passed=False,
                message="Invalid input type",
                metadata={
                    "error_type": "invalid_type",
                    "rule_id": rule_id,
                    "rule_type": self.__class__.__name__,
                },
                score=0.0,
                issues=["Input type not supported"],
                suggestions=["Use supported input type"],
                processing_time_ms=time.time() - start_time,
                rule_id=rule_id,
                severity=self.config.severity,
                category=self.config.category,
                tags=self.config.tags,
            )
            self.update_statistics(result)
            return result

        # Define the validation operation
        def validation_operation():
            # Get validation result from validator
            result = self._validator.validate(input)

            # Add rule metadata to result
            result = (
                result.with_metadata(
                    processing_time_ms=time.time() - start_time,
                    rule_id=rule_id,
                    rule_type=self.__class__.__name__,
                )
                .with_rule_id(rule_id)
                .with_severity(self.config.severity)
                .with_category(self.config.category)
                .with_tags(self.config.tags)
            )

            return result

        # Use the standardized safely_execute_rule function
        result = safely_execute_rule(
            operation=validation_operation,
            component_name=self.name,
            additional_metadata={
                "rule_id": rule_id,
                "rule_type": self.__class__.__name__,
            },
        )

        # If the result is an ErrorResult, convert it to a RuleResult
        if isinstance(result, dict) and result.get("error_type"):
            # Record the error
            self.record_error(Exception(result.get("error_message", "Unknown error")))

            # Create a RuleResult from the error
            result = RuleResult(
                passed=False,
                message=result.get("error_message", "Validation failed"),
                metadata={
                    "error_type": result.get("error_type"),
                    "rule_id": rule_id,
                    "rule_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[f"Validation error: {result.get('error_message')}"],
                suggestions=["Retry with different input"],
                processing_time_ms=time.time() - start_time,
                rule_id=rule_id,
                severity=self.config.severity,
                category=self.config.category,
                tags=self.config.tags,
            )

        # Update statistics
        self.update_statistics(result)
        return result

    def process(self, input: T) -> RuleResult:
        """
        Process the input through the rule pipeline.

        This method is required by the BaseComponent interface.
        For rules, it simply delegates to model_validate.

        Note:
            Previously, rules had a validate() method which has been removed.
            Use model_validate() instead for all validation operations.

        Args:
            input: The input to process

        Returns:
            Processing result (same as validation result)
        """
        return self.model_validate(input)


class FunctionRule(Rule[T]):
    """
    Rule that uses a function for validation.

    This rule wraps a function that performs validation,
    making it easy to create simple rules without defining new classes.

    ## Architecture
    FunctionRule extends the base Rule class but simplifies rule creation by:
    - Accepting a validation function in the constructor
    - Creating a FunctionValidator that wraps the function
    - Delegating validation to the function-based validator

    ## Lifecycle
    1. Initialization: Set up with validation function and configuration
       - Create a FunctionValidator with the provided function
       - Initialize base rule with the validator
    2. Validation: Delegate to the function-based validator
       - Input validation follows the base Rule pattern
       - The wrapped function is called by the validator
    3. Result: Return standardized validation results

    ## Error Handling
    - Function errors are caught and converted to failed RuleResults
    - Type compatibility is checked before calling the function
    - All errors are logged and can be tracked via statistics

    ## Examples
    ```python
    from sifaka.rules.base import FunctionRule, RuleResult

    # Define a validation function
    def validate_length(text: str) -> RuleResult:
        if len(text) < 10:
            return RuleResult(
                passed=False,
                message="Text is too short",
                issues=["Text must be at least 10 characters"],
                score=0.0
            )
        return RuleResult(
            passed=True,
            message="Text length is valid",
            score=1.0
        )

    # Create a rule with the function
    rule = FunctionRule(
        name="length_rule",
        func=validate_length,
        description="Validates text length",
        validation_type=str,  # Optional: specify input type
        severity="warning",   # Optional: set severity level
        category="formatting" # Optional: set category
    )

    # Validate text
    result = rule.validate("Hello")  # Will fail
    print(result.passed)  # False
    print(result.message)  # "Text is too short"

    result = rule.validate("Hello, world! This is a longer text.")  # Will pass
    print(result.passed)  # True
    ```
    """

    def __init__(
        self,
        name: str,
        func: Any,
        description: str = "",
        config: Optional[RuleConfig] = None,
        validation_type: Type[T] = str,
        **kwargs: Any,
    ):
        """
        Initialize the rule.

        Args:
            name: Name of the rule
            func: Function that performs validation
            description: Description of the rule
            config: Configuration for the rule
            validation_type: Type this rule can validate
            **kwargs: Additional keyword arguments for configuration
        """
        # Create the validator
        validator = FunctionValidator(func, validation_type)

        # Initialize base class with the validator
        super().__init__(
            name=name,
            description=description or f"Function rule using {func.__name__}",
            config=config or RuleConfig(**kwargs),
            validator=validator,
        )

        # Store the function for reference
        self._func = func

    def _create_default_validator(self) -> RuleValidator[T]:
        """
        Create a default validator for this rule.

        This method is not used since we create the validator in __init__,
        but it's required by the abstract base class.

        Returns:
            RuleValidator[T]: This method does not actually return a validator.

        Raises:
            NotImplementedError: Always raised since this method should not be called

        Note:
            This method exists only to satisfy the abstract method requirement
            from the base Rule class. It should never be called in practice
            because FunctionRule always creates its validator in __init__.
        """
        raise NotImplementedError("FunctionRule requires a validator to be passed in __init__")


__all__ = [
    "Rule",
    "RuleConfig",
    "RuleResult",
    "FunctionRule",
]
