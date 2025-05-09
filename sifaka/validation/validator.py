"""
Validator module for Sifaka.

This module provides the Validator class for validating outputs against rules.
It handles rule execution, result aggregation, and error message extraction.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, ConfigDict

from ..rules.base import Rule
from .models import ValidationResult

# Define a type variable for the output type
T = TypeVar("T")


class ValidatorConfig(BaseModel):
    """
    Configuration for the Validator.

    This class provides configuration options for the Validator,
    controlling how rules are executed and results are processed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize rules by cost (lowest first)",
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to stop validation after the first failure",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validator-specific configuration parameters",
    )


class Validator(Generic[T]):
    """
    Validator class that handles validation of outputs against rules.

    This class is responsible for:
    1. Running validation rules against an output
    2. Collecting and aggregating results
    3. Providing access to error messages

    The validator follows a simple workflow:
    1. Initialize with a set of rules
    2. Validate outputs using validate()
    3. Extract error messages if needed using get_error_messages()

    Examples:
        ```python
        from sifaka.validation.validator import Validator
        from sifaka.rules import create_length_rule

        # Create rules
        rules = [create_length_rule(min_chars=10)]

        # Create validator
        validator = Validator(rules)

        # Validate output
        result = validator.validate("Short")
        if not result.all_passed:
            errors = validator.get_error_messages(result)
            print(f"Validation failed: {errors}")
        ```
    """

    def __init__(
        self, 
        rules: List[Rule],
        config: Optional[ValidatorConfig] = None
    ):
        """
        Initialize a Validator instance.

        Args:
            rules: List of validation rules to apply. Each rule must implement
                  the Rule protocol and provide a validate() method.
            config: Configuration for the validator

        Raises:
            ValueError: If rules list is empty
            TypeError: If any rule does not implement the Rule protocol
        """
        if not rules:
            raise ValueError("Rules list cannot be empty")
        
        self.rules = rules
        self.config = config or ValidatorConfig()
        
        # Sort rules by cost if prioritization is enabled
        if self.config.prioritize_by_cost:
            self.rules = sorted(
                self.rules, 
                key=lambda rule: getattr(rule.config, "cost", float("inf"))
            )

    def validate(self, output: T) -> ValidationResult[T]:
        """
        Validate the output against all rules.

        This method:
        1. Applies each rule to the output
        2. Collects results from all rules
        3. Determines overall validation status
        4. Returns a ValidationResult with all details

        Args:
            output: The output to validate. Must be of the type specified
                   when creating the Validator instance.

        Returns:
            ValidationResult containing:
            - The original output
            - Results from each rule
            - Overall validation status

        Raises:
            ValueError: If output is None
            RuntimeError: If any rule fails during validation
        """
        if output is None:
            raise ValueError("Output cannot be None")

        rule_results = []
        metadata = {"validator_config": self.config.model_dump()}

        for rule in self.rules:
            try:
                result = rule.validate(output)
                rule_results.append(result)
                
                # If fail_fast is enabled and a rule failed, stop validation
                if self.config.fail_fast and not result.passed:
                    metadata["fail_fast_triggered"] = True
                    break
                    
            except Exception as e:
                raise RuntimeError(f"Rule validation failed: {str(e)}")

        return ValidationResult(
            output=output,
            rule_results=rule_results,
            metadata=metadata
        )

    def get_error_messages(self, validation_result: ValidationResult[T]) -> List[str]:
        """
        Get error messages from failed validations.

        This method extracts error messages from all failed rules in the
        validation result. It filters out passed rules and returns only
        messages from rules that failed validation.

        Args:
            validation_result: The validation result to extract errors from.
                             Must be a result from this validator's validate()
                             method.

        Returns:
            List of error messages from failed validations. Each message
            describes why a particular rule failed.

        Example:
            ```python
            result = validator.validate("Short")
            errors = validator.get_error_messages(result)
            # errors = ["Text must be at least 10 characters long"]
            ```
        """
        return [r.message for r in validation_result.rule_results if not r.passed]
