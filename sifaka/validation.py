"""
Validation module for Sifaka.

This module provides components for validating outputs against rules. It includes:
- ValidationResult: A generic container for validation results
- Validator: A class that applies validation rules to outputs

The validation process follows these steps:
1. Initialize a Validator with a set of rules
2. Pass an output to the validate() method
3. Receive a ValidationResult containing:
   - The original output
   - Results for each rule
   - Overall validation status
4. Extract error messages if needed

Example:
    ```python
    from sifaka.validation import Validator
    from sifaka.rules import LengthRule, ContentRule

    # Create rules
    rules = [
        LengthRule(min_length=10),
        ContentRule(forbidden_words=["bad", "inappropriate"])
    ]

    # Create validator
    validator = Validator(rules)

    # Validate output
    result = validator.validate("This is a test output")
    if not result.all_passed:
        errors = validator.get_error_messages(result)
        print(f"Validation failed: {errors}")
    ```
"""

from dataclasses import dataclass
from typing import List, Optional, TypeVar, Generic

from .rules import Rule, RuleResult

OutputType = TypeVar("OutputType")


@dataclass
class ValidationResult(Generic[OutputType]):
    """
    Result from validation, including the output and validation details.

    This class serves as a container for validation results, providing:
    - The original output being validated
    - Results from each validation rule
    - Overall validation status

    Attributes:
        output: The output that was validated
        rule_results: List of results from each validation rule
        all_passed: Boolean indicating if all rules passed

    Example:
        ```python
        result = ValidationResult(
            output="Test output",
            rule_results=[RuleResult(passed=True)],
            all_passed=True
        )
        ```
    """

    output: OutputType
    rule_results: List[RuleResult]
    all_passed: bool


class Validator(Generic[OutputType]):
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

    Example:
        ```python
        validator = Validator([LengthRule(min_length=10)])
        result = validator.validate("Short")
        if not result.all_passed:
            errors = validator.get_error_messages(result)
        ```
    """

    def __init__(self, rules: List[Rule]):
        """
        Initialize a Validator instance.

        Args:
            rules: List of validation rules to apply. Each rule must implement
                  the Rule protocol and provide a validate() method.

        Raises:
            ValueError: If rules list is empty
            TypeError: If any rule does not implement the Rule protocol
        """
        if not rules:
            raise ValueError("Rules list cannot be empty")
        self.rules = rules

    def validate(self, output: OutputType) -> ValidationResult[OutputType]:
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
            ValueError: If output is None or empty
            RuntimeError: If any rule fails during validation
        """
        if output is None:
            raise ValueError("Output cannot be None")

        rule_results = []
        all_passed = True

        for rule in self.rules:
            try:
                result = rule.validate(output)
                rule_results.append(result)
                if not result.passed:
                    all_passed = False
            except Exception as e:
                raise RuntimeError(f"Rule validation failed: {str(e)}")

        return ValidationResult(
            output=output,
            rule_results=rule_results,
            all_passed=all_passed
        )

    def get_error_messages(self, validation_result: ValidationResult[OutputType]) -> List[str]:
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