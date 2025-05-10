"""
Validation manager for Sifaka rules.

This module provides a manager for handling rule validation and state management.
"""

from typing import Any, Dict, List, Optional, Type

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.rules.base import BaseRule, RuleConfig, RuleResult
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from ..interfaces.rule import RuleProtocol
from ...utils.state import StateManager


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

    # Add state manager
    _state = PrivateAttr(default_factory=StateManager)

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

        # Initialize state
        self._state.update("initialized", True)
        self._state.update("rules", self.rules)
        self._state.update("config", self.config)
        self._state.update("validation_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "validation_manager")
        self._state.set_metadata("rule_count", len(self.rules))

    def validate(self, text: str, **kwargs: Any) -> List[RuleResult]:
        """
        Validate text against all managed rules.

        Args:
            text: Text to validate
            **kwargs: Additional validation options

        Returns:
            List of validation results
        """
        # Update validation count
        validation_count = self._state.get("validation_count", 0)
        self._state.update("validation_count", validation_count + 1)

        # Check cache if we've validated this text before
        cache = self._state.get("result_cache", {})
        cache_key = f"{text[:100]}:{str(kwargs)}"

        if cache_key in cache:
            self._state.set_metadata("cache_hit", True)
            return cache[cache_key]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Start timing
        import time

        start_time = time.time()

        results = []
        for rule in self._get_prioritized_rules():
            result = rule.validate(text, **kwargs)
            results.append(result)

            # Track pass/fail statistics
            if result.passed:
                pass_count = self._state.get_metadata("pass_count", 0)
                self._state.set_metadata("pass_count", pass_count + 1)
            else:
                fail_count = self._state.get_metadata("fail_count", 0)
                self._state.set_metadata("fail_count", fail_count + 1)

                # Track failing rules
                failing_rules = self._state.get_metadata("failing_rules", {})
                failing_rules[rule.name] = failing_rules.get(rule.name, 0) + 1
                self._state.set_metadata("failing_rules", failing_rules)

        # Record execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Update average execution time
        avg_time = self._state.get_metadata("avg_execution_time", 0)
        count = self._state.get("validation_count", 1)
        new_avg = ((avg_time * (count - 1)) + execution_time) / count
        self._state.set_metadata("avg_execution_time", new_avg)

        # Update max execution time if needed
        max_time = self._state.get_metadata("max_execution_time", 0)
        if execution_time > max_time:
            self._state.set_metadata("max_execution_time", execution_time)

        # Cache results (limit cache size)
        cache_size = 100  # Could make this configurable
        if len(cache) >= cache_size:
            # Simple strategy: just clear the cache when it gets full
            cache = {}

        cache[cache_key] = results
        self._state.update("result_cache", cache)

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

        # Update state
        self._state.update("rules", self.rules)
        self._state.set_metadata("rule_count", len(self.rules))

    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from the manager.

        Args:
            rule_name: Name of the rule to remove
        """
        self.rules = [r for r in self.rules if r.name != rule_name]

        # Update state
        self._state.update("rules", self.rules)
        self._state.set_metadata("rule_count", len(self.rules))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation statistics
        """
        return {
            "validation_count": self._state.get("validation_count", 0),
            "pass_count": self._state.get_metadata("pass_count", 0),
            "fail_count": self._state.get_metadata("fail_count", 0),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "failing_rules": self._state.get_metadata("failing_rules", {}),
            "rule_count": self._state.get_metadata("rule_count", 0),
            "cache_size": len(self._state.get("result_cache", {})),
        }

    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self._state.update("validation_count", 0)
        self._state.set_metadata("pass_count", 0)
        self._state.set_metadata("fail_count", 0)
        self._state.set_metadata("failing_rules", {})
        self._state.set_metadata("avg_execution_time", 0)
        self._state.set_metadata("max_execution_time", 0)

    def clear_cache(self) -> None:
        """Clear the validation result cache."""
        self._state.update("result_cache", {})
