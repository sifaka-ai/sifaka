"""
Pydantic models for validation.

This module provides Pydantic models for validation results,
ensuring type safety and validation for validation-related data structures.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from typing_extensions import Annotated

from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator, model_validator

from ..rules.base import RuleResult

# Define a type variable for the output type
T = TypeVar("T")


class ValidationResult(BaseModel, Generic[T]):
    """
    Result from validation, including the output and validation details.

    This class serves as a container for validation results, providing:
    - The original output being validated
    - Results from each validation rule
    - Overall validation status

    Examples:
        ```python
        from sifaka.validation.models import ValidationResult
        from sifaka.rules.base import RuleResult

        # Create a validation result
        result = ValidationResult(
            output="Test output",
            rule_results=[
                RuleResult(
                    rule_id="length_rule",
                    passed=True,
                    message="Length is acceptable"
                ),
                RuleResult(
                    rule_id="content_rule",
                    passed=False,
                    message="Contains prohibited content",
                    metadata={"prohibited_terms": ["xyz"]}
                )
            ]
        )

        # Check if all rules passed
        if result.all_passed:
            print("Validation successful!")
        else:
            print("Validation failed!")
            for rule_result in result.failed_results:
                print(f"- {rule_result.message}")
        ```

    Attributes:
        output: The output that was validated
        rule_results: List of results from each validation rule
    """

    model_config = ConfigDict(frozen=True)

    output: T = Field(description="The output that was validated")
    rule_results: List[RuleResult] = Field(
        default_factory=list, description="Results from each validation rule"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the validation"
    )

    @property
    def all_passed(self) -> bool:
        """
        Check if all rules passed.

        Returns:
            True if all rules passed, False otherwise
        """
        return all(r.passed for r in self.rule_results)

    @property
    def any_passed(self) -> bool:
        """
        Check if any rules passed.

        Returns:
            True if any rules passed, False otherwise
        """
        return any(r.passed for r in self.rule_results)

    @property
    def failed_results(self) -> List[RuleResult]:
        """
        Get the results of failed rules.

        Returns:
            List of results from rules that failed
        """
        return [r for r in self.rule_results if not r.passed]

    @property
    def passed_results(self) -> List[RuleResult]:
        """
        Get the results of passed rules.

        Returns:
            List of results from rules that passed
        """
        return [r for r in self.rule_results if r.passed]

    def __bool__(self) -> bool:
        """
        Convert to boolean.

        Returns:
            True if all rules passed, False otherwise
        """
        return self.all_passed
