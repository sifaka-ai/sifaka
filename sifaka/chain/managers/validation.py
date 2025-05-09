"""
Validation Manager Module

Manages validation rules and execution for Sifaka's chain system.

## Overview
This module provides the ValidationManager class which handles validation of outputs
against rules. It manages rule registration, validation execution, and error message
generation, providing a centralized way to handle validation in the chain system.

## Components
1. **ValidationManager**: Main validation management class
   - Rule management
   - Validation execution
   - Error message generation

2. **Validator**: Core validation engine
   - Rule execution
   - Result aggregation
   - Error handling

## Usage Examples
```python
from sifaka.chain.managers.validation import ValidationManager
from sifaka.rules import create_length_rule, create_toxicity_rule

# Create rules
rules = [
    create_length_rule(min_length=10, max_length=1000),
    create_toxicity_rule(threshold=0.7)
]

# Create validation manager
manager = ValidationManager(
    rules=rules,
    prioritize_by_cost=True,
    fail_fast=True
)

# Validate output
result = manager.validate("Some output text")

# Check validation result
if result.all_passed:
    print("Validation passed!")
else:
    error_messages = manager.get_error_messages(result)
    print("Validation failed:")
    for msg in error_messages:
        print(f"- {msg}")

# Add new rule
new_rule = create_length_rule(min_length=20)
manager.add_rule(new_rule)

# Remove rule
manager.remove_rule("length_rule")
```

## Error Handling
- ValueError: Raised for invalid rules or rule operations
- ValidationError: Raised when validation fails
- TypeError: Raised for type validation failures

## Configuration
- rules: List of validation rules to apply
- prioritize_by_cost: Whether to sort rules by cost (lowest first)
- fail_fast: Whether to stop after first failure
"""

from typing import Any, Generic, List, Optional, TypeVar

from ..interfaces.manager import ValidationManagerProtocol
from ...rules import Rule
from ...validation.models import ValidationResult
from ...validation.validator import Validator, ValidatorConfig
from ...utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ValidationManager(
    ValidationManagerProtocol[OutputType, ValidationResult[OutputType]], Generic[OutputType]
):
    """
    Manages validation rules and execution for chains.

    ## Overview
    This class provides centralized management of validation rules and their
    execution. It handles rule registration, validation execution, and error
    message generation, implementing the ValidationManagerProtocol interface.

    ## Architecture
    ValidationManager follows a manager pattern:
    1. **Rule Management**: Handles rule registration and removal
    2. **Validation Execution**: Coordinates rule execution
    3. **Result Handling**: Manages validation results and errors

    ## Lifecycle
    1. **Initialization**: Set up with rules and configuration
       - Register validation rules
       - Configure validation behavior
       - Create validator instance

    2. **Operation**: Handle validation requests
       - Execute validation
       - Generate error messages
       - Manage rules dynamically

    ## Error Handling
    - ValueError: Raised for invalid rules or rule operations
    - ValidationError: Raised when validation fails
    - TypeError: Raised for type validation failures

    ## Examples
    ```python
    from sifaka.chain.managers.validation import ValidationManager
    from sifaka.rules import create_length_rule

    # Create manager
    manager = ValidationManager(
        rules=[create_length_rule(min_length=10)],
        prioritize_by_cost=True,
        fail_fast=True
    )

    # Validate output
    result = manager.validate("Some text")
    if not result.all_passed:
        errors = manager.get_error_messages(result)
        print("Validation failed:", errors)

    # Add new rule
    new_rule = create_length_rule(max_length=1000)
    manager.add_rule(new_rule)
    ```

    Type parameters:
        OutputType: The type of output being validated
    """

    def __init__(
        self, rules: List[Rule], prioritize_by_cost: bool = False, fail_fast: bool = False
    ):
        """
        Initialize a ValidationManager instance.

        ## Overview
        This method sets up the validation manager with the provided rules
        and configuration options. It creates a validator instance with the
        specified behavior.

        ## Lifecycle
        1. **Rule Setup**: Register initial rules
           - Store rule list
           - Configure rule ordering

        2. **Configuration**: Set validation behavior
           - Configure prioritization
           - Configure fail-fast behavior
           - Create validator instance

        Args:
            rules: The rules to validate against
            prioritize_by_cost: If True, rules will be sorted by cost (lowest first)
            fail_fast: If True, validation will stop after the first failure
        """
        self._rules = rules

        # Create validator config
        validator_config = ValidatorConfig(
            prioritize_by_cost=prioritize_by_cost, fail_fast=fail_fast
        )

        # Log configuration
        if prioritize_by_cost:
            logger.info(f"Validation will prioritize rules by cost (lowest first)")
        if fail_fast:
            logger.info(f"Validation will stop after the first failure")

        # Create validator with config
        self._validator = Validator[OutputType](rules=self._rules, config=validator_config)

    @property
    def rules(self) -> List[Rule]:
        """
        Get the rules.

        Returns:
            The rules
        """
        return self._rules

    @property
    def validator(self) -> Validator[OutputType]:
        """
        Get the validator.

        Returns:
            The validator
        """
        return self._validator

    def validate(self, output: OutputType) -> ValidationResult[OutputType]:
        """
        Validate the output against rules.

        Args:
            output: The output to validate

        Returns:
            The validation result
        """
        return self._validator.validate(output)

    def get_error_messages(self, validation_result: ValidationResult[OutputType]) -> List[str]:
        """
        Get error messages from a validation result.

        Args:
            validation_result: The validation result

        Returns:
            The error messages
        """
        return self._validator.get_error_messages(validation_result)

    def add_rule(self, rule: Any) -> None:
        """
        Add a rule for validation.

        This method implements the ValidationManagerProtocol.add_rule method.

        Args:
            rule: The rule to add

        Raises:
            ValueError: If the rule is invalid
        """
        if not isinstance(rule, Rule):
            raise ValueError(f"Expected Rule instance, got {type(rule)}")

        # Get current validator config
        current_config = self._validator.config

        # Add rule to the list
        self._rules.append(rule)

        # Recreate validator with the same config
        self._validator = Validator[OutputType](rules=self._rules, config=current_config)

        logger.info(f"Added rule {rule.name if hasattr(rule, 'name') else str(rule)}")

    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from validation.

        This method implements the ValidationManagerProtocol.remove_rule method.

        Args:
            rule_name: The name of the rule to remove

        Raises:
            ValueError: If the rule is not found
        """
        # Get current validator config
        current_config = self._validator.config

        # Find and remove the rule
        for i, rule in enumerate(self._rules):
            if getattr(rule, "name", str(rule)) == rule_name:
                self._rules.pop(i)

                # Recreate validator with the same config
                self._validator = Validator[OutputType](rules=self._rules, config=current_config)

                logger.info(f"Removed rule {rule_name}")
                return

        raise ValueError(f"Rule {rule_name} not found")

    def get_rules(self) -> List[Any]:
        """
        Get all registered rules.

        This method implements the ValidationManagerProtocol.get_rules method.

        Returns:
            A list of registered rules
        """
        return self._rules
