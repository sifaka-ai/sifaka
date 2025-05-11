"""
Base classes for Sifaka rules.

This module defines the core architecture for rules in Sifaka,
providing the foundation for all validation components in the framework.

The rules system follows a delegation pattern:
1. Rules are high-level containers that define what to validate
2. Validators implement the actual validation logic
3. Rules delegate validation work to their validators

This separation of concerns allows for:
- Reusing validation logic across different rules
- Testing validation logic independently
- Extending the framework with custom validators
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
from sifaka.utils.error_patterns import safely_execute_rule
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_rule_state

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


class RuleResult(BaseResult):
    """
    Result from rule validation.

    This class extends BaseResult to add rule-specific result information.

    Attributes:
        severity: Severity level of the result (error, warning, info)
        category: Category of the rule that produced this result
        tags: List of tags associated with the rule
        rule_id: Identifier of the rule that produced this result
        validation_time: When the validation was performed
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    severity: str = Field(
        default="error",
        description="Severity level of the result",
    )
    category: str = Field(
        default="general",
        description="Category of the rule that produced this result",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags associated with the rule",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="Identifier of the rule that produced this result",
    )
    validation_time: datetime = Field(
        default_factory=datetime.now,
        description="When the validation was performed",
    )

    def with_rule_id(self, rule_id: str) -> "RuleResult":
        """Create a new result with the rule ID set."""
        return self.model_copy(update={"rule_id": rule_id})

    def with_severity(self, severity: str) -> "RuleResult":
        """Create a new result with updated severity."""
        return self.model_copy(update={"severity": severity})

    def with_category(self, category: str) -> "RuleResult":
        """Create a new result with updated category."""
        return self.model_copy(update={"category": category})

    def with_tags(self, tags: List[str]) -> "RuleResult":
        """Create a new result with updated tags."""
        return self.model_copy(update={"tags": tags})


class Rule(BaseComponent[T, RuleResult], Generic[T]):
    """
    Base class for all rules.

    Rules are high-level components that define what to validate.
    They delegate the actual validation work to validators.

    Lifecycle:
        1. Initialization: Set up with validator and configuration
        2. Validation: Delegate to validator and process results
        3. Result: Return standardized validation results with metadata
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_rule_state)

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

        Args:
            name: Name of the rule
            description: Description of the rule
            config: Configuration for the rule
            validator: Validator to use for validation
            **kwargs: Additional keyword arguments for configuration
        """
        super().__init__(name, description, config or RuleConfig(**kwargs))
        self._validator = validator or self._create_default_validator()

        # Initialize rule-specific state
        self._state_manager.set_metadata("rule_type", self.__class__.__name__)
        self._state_manager.set_metadata("rule_id", getattr(config, "rule_id", None) or name)
        self._state_manager.set_metadata(
            "validator_type", getattr(self._validator, "__class__", {}).get("__name__", "Unknown")
        )

    @abstractmethod
    def _create_default_validator(self) -> RuleValidator[T]:
        """
        Create the default validator for this rule.

        Returns:
            Default validator instance
        """
        ...

    def validate(self, input: T) -> RuleResult:
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
            rule_name=self.name,
            component_name=self.__class__.__name__,
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
        For rules, it simply delegates to validate.

        Args:
            input: The input to process

        Returns:
            Processing result (same as validation result)
        """
        return self.validate(input)


class FunctionRule(Rule[T]):
    """
    Rule that uses a function for validation.

    This rule wraps a function that performs validation,
    making it easy to create simple rules without defining new classes.

    Examples:
        ```python
        def validate_length(text: str) -> RuleResult:
            if len(text) < 10:
                return RuleResult(
                    passed=False,
                    message="Text is too short",
                    issues=["Text must be at least 10 characters"]
                )
            return RuleResult(passed=True, message="Text length is valid")

        rule = FunctionRule(
            name="length_rule",
            func=validate_length,
            description="Validates text length"
        )
        result = rule.validate("Hello")  # Will fail
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

        Raises:
            NotImplementedError: Always raised since this method should not be called
        """
        raise NotImplementedError("FunctionRule requires a validator to be passed in __init__")


__all__ = [
    "Rule",
    "RuleConfig",
    "RuleResult",
    "FunctionRule",
]
