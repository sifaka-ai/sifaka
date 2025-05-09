"""
Validation manager module for Sifaka.

This module provides the ValidationManager class which is responsible for
validating outputs against rules.
"""

from typing import Any, Generic, List, TypeVar

from ..interfaces.manager import ValidationManagerProtocol
from ...rules import Rule
from ...validation import ValidationResult, Validator
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

    def __init__(self, rules: List[Rule], prioritize_by_cost: bool = False):
        """
        Initialize a ValidationManager instance.

        Args:
            rules: The rules to validate against
            prioritize_by_cost: If True, rules will be sorted by cost (lowest first)
        """
        self._rules = rules

        # Sort rules by cost if prioritization is enabled
        if prioritize_by_cost:
            self._rules = sorted(rules, key=lambda rule: getattr(rule.config, "cost", float("inf")))
            logger.info(f"Sorted {len(rules)} rules by cost")

        self._validator = Validator[OutputType](self._rules)

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

        self._rules.append(rule)
        self._validator = Validator[OutputType](self._rules)
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
        for i, rule in enumerate(self._rules):
            if getattr(rule, "name", str(rule)) == rule_name:
                self._rules.pop(i)
                self._validator = Validator[OutputType](self._rules)
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
