"""
Validation management for Sifaka rules.

This module provides the ValidationManager class for managing rule validation.
It handles rule execution, result processing, and validation flow control.
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict

from ..interfaces.rule import RuleProtocol
from ..result import RuleResult


class ValidationConfig(BaseModel):
    """
    Configuration for validation management.

    This class provides configuration options for the ValidationManager,
    controlling how rules are executed and results are processed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize rules by cost",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Manager-specific configuration parameters",
    )

    def with_options(self, **kwargs: Any) -> "ValidationConfig":
        """
        Create a new config with updated options.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options
        """
        return ValidationConfig(**{**self.model_dump(), **kwargs})


class ValidationManager:
    """
    Manager for rule validation.

    This class manages the execution of rules and processing of validation results.
    It provides methods for validating text against multiple rules and controlling
    the validation flow.
    """

    def __init__(
        self,
        rules: Optional[List[RuleProtocol]] = None,
        config: Optional[ValidationConfig] = None,
    ):
        """
        Initialize the validation manager.

        Args:
            rules: List of rules to manage
            config: Configuration for the manager
        """
        self.rules = rules or []
        self.config = config or ValidationConfig()

    def validate(self, text: str, **kwargs: Any) -> List[RuleResult]:
        """
        Validate text against all managed rules.

        Args:
            text: Text to validate
            **kwargs: Additional validation options

        Returns:
            List of validation results
        """
        results = []
        for rule in self._get_prioritized_rules():
            result = rule.validate(text, **kwargs)
            results.append(result)
        return results

    def _get_prioritized_rules(self) -> List[RuleProtocol]:
        """
        Get rules in priority order.

        Returns:
            List of rules in priority order
        """
        if self.config.prioritize_by_cost:
            return sorted(self.rules, key=lambda r: r.config.cost)
        return sorted(self.rules, key=lambda r: r.config.priority.value, reverse=True)

    def add_rule(self, rule: RuleProtocol) -> None:
        """
        Add a rule to the manager.

        Args:
            rule: Rule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from the manager.

        Args:
            rule_name: Name of the rule to remove
        """
        self.rules = [r for r in self.rules if r.name != rule_name]
