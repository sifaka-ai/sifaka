"""
Validation module for Sifaka.

This module provides components for validating outputs against rules.
"""

from dataclasses import dataclass
from typing import List, Optional, TypeVar, Generic

from .rules import Rule, RuleResult

OutputType = TypeVar("OutputType")


@dataclass
class ValidationResult(Generic[OutputType]):
    """Result from validation, including the output and validation details."""

    output: OutputType
    rule_results: List[RuleResult]
    all_passed: bool


class Validator(Generic[OutputType]):
    """
    Validator class that handles validation of outputs against rules.

    This class is responsible for running validation rules against an output
    and collecting the results.
    """

    def __init__(self, rules: List[Rule]):
        """
        Initialize a Validator instance.

        Args:
            rules: List of validation rules to apply
        """
        self.rules = rules

    def validate(self, output: OutputType) -> ValidationResult[OutputType]:
        """
        Validate the output against all rules.

        Args:
            output: The output to validate

        Returns:
            ValidationResult containing the output, rule results, and validation status
        """
        rule_results = []
        all_passed = True

        for rule in self.rules:
            result = rule.validate(output)
            rule_results.append(result)
            if not result.passed:
                all_passed = False

        return ValidationResult(
            output=output,
            rule_results=rule_results,
            all_passed=all_passed
        )

    def get_error_messages(self, validation_result: ValidationResult[OutputType]) -> List[str]:
        """
        Get error messages from failed validations.

        Args:
            validation_result: The validation result to extract errors from

        Returns:
            List of error messages from failed validations
        """
        return [r.message for r in validation_result.rule_results if not r.passed]