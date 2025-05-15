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
from typing import Any, Generic, List, Optional, Type, TypeVar, Dict, Union
import time
from pydantic import Field, ConfigDict, PrivateAttr
from sifaka.core.base import BaseComponent, BaseConfig

# Import utilities
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_rule_state, StateManager
from .validators import RuleValidator, FunctionValidator, BaseValidator

logger = get_logger(__name__)
T = TypeVar("T")
R = TypeVar("R")
V = TypeVar("V", bound=BaseValidator)


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
    severity: str = Field(default="error", description="Severity level for rule violations")
    category: str = Field(default="general", description="Category of the rule")
    tags: List[str] = Field(
        default_factory=list, description="List of tags for categorizing the rule"
    )
    track_performance: bool = Field(
        default=True, description="Whether to track performance metrics"
    )
    track_errors: bool = Field(default=True, description="Whether to track error occurrences")
    rule_id: Optional[str] = Field(default=None, description="Unique identifier for the rule")


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
        object.__setattr__(self, "_state_manager", create_rule_state())
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

        # Helper method to update statistics with a result
        def update_stats(result: RuleResult) -> None:
            """Update statistics using the RuleResult's properties directly."""
            # Import directly here to avoid circular imports
            from sifaka.utils.result_types import BaseResult
            from sifaka.utils.common import update_statistics

            # Track execution time
            execution_time = (time.time() - start_time) / 1000.0  # convert to seconds
            success = result.passed

            # Update the statistics using the common utility
            update_statistics(
                state_manager=self._state_manager,
                execution_time=execution_time,
                success=success,
            )

        # Helper method to create a RuleResult from common parameters
        def create_rule_result(
            passed: bool,
            message: str,
            metadata: Optional[Dict[str, Any]] = None,
            score: float = 0.0,
            issues: Optional[List[str]] = None,
            suggestions: Optional[List[str]] = None,
        ) -> RuleResult:
            return RuleResult(
                passed=passed,
                message=message,
                metadata={
                    **(metadata or {}),
                    "rule_id": rule_id,
                    "rule_type": self.__class__.__name__,
                },
                score=score,
                issues=issues or [],
                suggestions=suggestions or [],
                processing_time_ms=time.time() - start_time,
                rule_id=rule_id,
                severity=self.config.severity if isinstance(self.config, RuleConfig) else "error",
                category=self.config.category if isinstance(self.config, RuleConfig) else "general",
                tags=self.config.tags if isinstance(self.config, RuleConfig) else [],
            )

        # Helper function to handle empty input, returning a RuleResult
        def handle_empty_text(text: str) -> Optional[RuleResult]:
            """Handle empty text validation returning a RuleResult."""
            from sifaka.utils.text import is_empty_text

            if isinstance(text, str) and is_empty_text(text):
                return create_rule_result(
                    passed=False,
                    message="Empty input",
                    metadata={"error_type": "empty_input"},
                    issues=["Input is empty"],
                    suggestions=["Provide non-empty input"],
                )
            if not text:
                return create_rule_result(
                    passed=False,
                    message="Empty input",
                    metadata={"error_type": "empty_input"},
                    issues=["Input is empty"],
                    suggestions=["Provide non-empty input"],
                )
            return None

        # Check if the input can be validated
        if not self.validate_input(input):
            result = create_rule_result(
                passed=False,
                message="Invalid input",
                metadata={"error_type": "invalid_input"},
                issues=["Invalid input type"],
                suggestions=["Provide valid input"],
            )
            update_stats(result)
            return result

        # Check for empty input
        text_input = str(input) if input is not None else ""
        empty_result = handle_empty_text(text_input)
        if empty_result:
            update_stats(empty_result)
            return empty_result

        # Check if the validator can validate the input
        if not self._validator or not self._validator.can_validate(input):
            result = create_rule_result(
                passed=False,
                message="Invalid input type",
                metadata={"error_type": "invalid_type"},
                issues=["Input type not supported"],
                suggestions=["Use supported input type"],
            )
            update_stats(result)
            return result

        # Create validation operation
        def validation_operation() -> RuleResult:
            # Directly cast the validation result to RuleResult
            # This helps mypy understand that we explicitly want a RuleResult
            from typing import cast

            # Get the raw validation result
            raw_result = self._validator.validate(input)

            # Create a dict of all the attributes we need
            result_dict = {
                "passed": getattr(raw_result, "passed", False),
                "message": getattr(raw_result, "message", "Validation completed"),
                "metadata": {
                    **(getattr(raw_result, "metadata", {}) or {}),
                    "processing_time_ms": time.time() - start_time,
                    "rule_id": rule_id,
                    "rule_type": self.__class__.__name__,
                },
                "score": getattr(raw_result, "score", 0.0),
                "issues": getattr(raw_result, "issues", []) or [],
                "suggestions": getattr(raw_result, "suggestions", []) or [],
                "processing_time_ms": time.time() - start_time,
                "rule_id": rule_id,
            }

            # Add rule-specific properties if we have a RuleConfig
            if isinstance(self.config, RuleConfig):
                result_dict["severity"] = self.config.severity
                result_dict["category"] = self.config.category
                result_dict["tags"] = self.config.tags
            else:
                result_dict["severity"] = "error"
                result_dict["category"] = "general"
                result_dict["tags"] = []

            # Create and return a new RuleResult
            return RuleResult(**result_dict)

        # Import the correct function for safe execution
        from sifaka.utils.common import safely_execute
        from typing import cast, Dict, Any

        # Call safely_execute with the correct parameters
        try:
            # Safely execute the validation operation
            # We explicitly annotate the result to help mypy understand the type
            operation_result: Union[RuleResult, Dict[str, Any]] = safely_execute(
                operation=validation_operation,
                component_name=self.name,
                state_manager=self._state_manager,
                component_type="Rule",
            )
        except Exception as e:
            # Handle any exception from safely_execute
            self.record_error(e)
            result = create_rule_result(
                passed=False,
                message=f"Validation failed: {str(e)}",
                metadata={"error_type": "execution_error"},
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Retry with different input"],
            )
            update_stats(result)
            return result

        # Handle the result based on type
        if (
            isinstance(operation_result, dict)
            and operation_result
            and operation_result.get("error_type")
        ):
            # Handle error result (dict with error info)
            self.record_error(Exception(operation_result.get("error_message", "Unknown error")))
            result = create_rule_result(
                passed=False,
                message=operation_result.get("error_message", "Validation failed"),
                metadata={"error_type": operation_result.get("error_type")},
                issues=[
                    f"Validation error: {operation_result.get('error_message', 'Unknown error')}"
                ],
                suggestions=["Retry with different input"],
            )
        elif isinstance(operation_result, RuleResult):
            # Already a RuleResult, use it directly
            result = operation_result
        else:
            # For any other type, create a new RuleResult
            try:
                passed = bool(getattr(operation_result, "passed", False))
            except (AttributeError, TypeError):
                passed = False

            try:
                message = str(getattr(operation_result, "message", "Validation completed"))
            except (AttributeError, TypeError):
                message = "Validation completed"

            try:
                op_metadata = dict(getattr(operation_result, "metadata", {}))
            except (AttributeError, TypeError):
                op_metadata = {}

            try:
                score = float(getattr(operation_result, "score", 0.0))
            except (AttributeError, TypeError, ValueError):
                score = 0.0

            try:
                issues = list(getattr(operation_result, "issues", []))
            except (AttributeError, TypeError):
                issues = []

            try:
                suggestions = list(getattr(operation_result, "suggestions", []))
            except (AttributeError, TypeError):
                suggestions = []

            # Create the RuleResult
            result = create_rule_result(
                passed=passed,
                message=message,
                metadata=op_metadata,
                score=score,
                issues=issues,
                suggestions=suggestions,
            )

        # Update statistics and return the result
        update_stats(result)
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
        validation_type: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> None:
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
        validator = FunctionValidator(func, validation_type or str)
        super().__init__(
            name=name,
            description=description or f"Function rule using {func.__name__}",
            config=config or RuleConfig(**kwargs),
            validator=validator,
        )
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


__all__ = ["Rule", "RuleConfig", "RuleResult", "FunctionRule"]
