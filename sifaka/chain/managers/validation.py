"""
Validation manager module for Sifaka.

This module provides the ValidationManager class which is responsible for
validating outputs against rules.
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
    Manages validation for chains.

    This class is responsible for validating outputs against rules and
    managing rule-related functionality. It implements the ValidationManagerProtocol interface.
    """

    def __init__(
        self, rules: List[Rule], prioritize_by_cost: bool = False, fail_fast: bool = False
    ):
        """
        Initialize a ValidationManager instance.

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
