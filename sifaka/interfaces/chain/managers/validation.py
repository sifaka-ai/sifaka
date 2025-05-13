"""
Validation manager interface for Sifaka.

This module defines the interface for validation managers in the Sifaka framework.
These interfaces establish a common contract for validation manager behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **ValidationManager**: Interface for validation managers

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, List, Protocol, TypeVar, runtime_checkable

# Type variables
InputType = TypeVar("InputType", contravariant=True)
ValidationResultType = TypeVar("ValidationResultType", covariant=True)


@runtime_checkable
class ValidationManager(Protocol[InputType, ValidationResultType]):
    """
    Interface for validation managers.

    This interface defines the contract for components that manage validation.
    It ensures that validation managers can validate inputs against rules and
    aggregate validation results.

    ## Lifecycle

    1. **Initialization**: Set up validation management resources
    2. **Rule Registration**: Register rules for validation
    3. **Input Validation**: Validate inputs against rules
    4. **Result Aggregation**: Aggregate validation results
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a validate method to validate inputs against rules
    - Provide methods to register and manage rules
    - Aggregate validation results
    """

    @abstractmethod
    def validate(self, input_value: InputType) -> ValidationResultType:
        """
        Validate an input value against registered rules.

        Args:
            input_value: The input value to validate

        Returns:
            A validation result

        Raises:
            ValueError: If the input value is invalid
        """

    @abstractmethod
    def add_rule(self, rule: Any) -> None:
        """
        Add a rule for validation.

        Args:
            rule: The rule to add

        Raises:
            ValueError: If the rule is invalid
        """

    @abstractmethod
    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from validation.

        Args:
            rule_name: The name of the rule to remove

        Raises:
            ValueError: If the rule is not found
        """

    @abstractmethod
    def get_rules(self) -> List[Any]:
        """
        Get all registered rules.

        Returns:
            A list of registered rules
        """
