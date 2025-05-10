"""
Validation manager for Sifaka rules.

This module provides a manager for handling rule validation and state management.

Usage Example:
    ```python
    from sifaka.rules.managers.validation import ValidationManager
    from sifaka.rules.formatting.length import create_length_rule
    from sifaka.rules.formatting.format import create_markdown_rule

    # Create rules
    length_rule = create_length_rule(min_length=10, max_length=1000)
    markdown_rule = create_markdown_rule(required_elements=["#", "*", "`"])

    # Create validation manager
    manager = ValidationManager(rules=[length_rule, markdown_rule])

    # Validate text
    results = manager.validate("# Heading\n\n* List item")

    # Check results
    for result in results:
        print(f"Rule: {result.rule_name}, Passed: {result.passed}, Message: {result.message}")

    # Get validation statistics
    stats = manager.get_statistics()
    print(f"Validation count: {stats['validation_count']}")
    ```
"""

import time
from typing import Any, Dict, List, Optional, Type, Union

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.rules.base import Rule as BaseRule, RuleConfig, RuleResult
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from sifaka.interfaces import RuleProtocol


class ValidationConfig(BaseModel):
    """
    Configuration for validation management.

    This class provides configuration options for the ValidationManager,
    controlling how rules are executed and results are processed.

    Attributes:
        prioritize_by_cost: Whether to prioritize rules by cost
        cache_size: Maximum number of results to cache
        params: Manager-specific configuration parameters

    Examples:
        ```python
        from sifaka.rules.managers.validation import ValidationConfig

        # Create a basic configuration
        config = ValidationConfig(prioritize_by_cost=True)

        # Create a configuration with custom parameters
        config = ValidationConfig(
            prioritize_by_cost=True,
            cache_size=200,
            params={"fail_fast": True, "timeout": 5.0}
        )
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize rules by cost",
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Maximum number of results to cache",
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

        Examples:
            ```python
            # Update configuration options
            updated_config = config.with_options(
                prioritize_by_cost=False,
                cache_size=500
            )
            ```
        """
        return ValidationConfig(**{**self.model_dump(), **kwargs})


class ValidationManager:
    """
    Manager for rule validation.

    This class manages the execution of rules and processing of validation results.
    It provides methods for validating text against multiple rules and controlling
    the validation flow.

    Lifecycle:
        1. Initialization: Set up with rules and configuration
        2. Rule Management: Add/remove rules as needed
        3. Validation: Validate text against rules
        4. Result Processing: Process validation results
        5. Statistics: Track and report validation statistics

    Examples:
        ```python
        from sifaka.rules.managers.validation import ValidationManager
        from sifaka.rules.formatting.length import create_length_rule

        # Create rules
        length_rule = create_length_rule(min_length=10, max_length=1000)

        # Create validation manager
        manager = ValidationManager(rules=[length_rule])

        # Validate text
        results = manager.validate("This is a test.")

        # Add more rules
        from sifaka.rules.formatting.format import create_markdown_rule
        markdown_rule = create_markdown_rule(required_elements=["#", "*", "`"])
        manager.add_rule(markdown_rule)

        # Get validation statistics
        stats = manager.get_statistics()
        print(f"Validation count: {stats['validation_count']}")
        ```
    """

    # Add state manager
    _state_manager = PrivateAttr(default_factory=StateManager)

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

        Examples:
            ```python
            # Create with default configuration
            manager = ValidationManager(rules=[rule1, rule2])

            # Create with custom configuration
            from sifaka.rules.managers.validation import ValidationConfig
            config = ValidationConfig(
                prioritize_by_cost=True,
                cache_size=200
            )
            manager = ValidationManager(
                rules=[rule1, rule2],
                config=config
            )
            ```
        """
        self.rules = rules or []
        self.config = config or ValidationConfig()

        # Initialize state
        self._state_manager.update("initialized", True)
        self._state_manager.update("rules", self.rules)
        self._state_manager.update("config", self.config)
        self._state_manager.update("validation_count", 0)
        self._state_manager.update("result_cache", {})
        self._state_manager.update("start_time", time.time())

        # Set metadata
        self._state_manager.set_metadata("component_type", "validation_manager")
        self._state_manager.set_metadata("rule_count", len(self.rules))
        self._state_manager.set_metadata("pass_count", 0)
        self._state_manager.set_metadata("fail_count", 0)
        self._state_manager.set_metadata("failing_rules", {})
        self._state_manager.set_metadata("avg_execution_time", 0)
        self._state_manager.set_metadata("max_execution_time", 0)

        logger.debug(f"Initialized ValidationManager with {len(self.rules)} rules")

    def validate(self, text: str, **kwargs: Any) -> List[RuleResult]:
        """
        Validate text against all managed rules.

        Args:
            text: Text to validate
            **kwargs: Additional validation options including:
                - fail_fast: Stop validation after first failure
                - timeout: Maximum time to spend on validation (seconds)
                - include_rules: List of rule names to include
                - exclude_rules: List of rule names to exclude

        Returns:
            List of validation results

        Examples:
            ```python
            # Basic validation
            results = manager.validate("This is a test.")

            # Validation with options
            results = manager.validate(
                "This is a test.",
                fail_fast=True,
                timeout=2.0,
                include_rules=["length_rule"]
            )
            ```
        """
        # Update validation count
        validation_count = self._state_manager.get("validation_count", 0)
        self._state_manager.update("validation_count", validation_count + 1)

        # Check cache if we've validated this text before
        cache = self._state_manager.get("result_cache", {})
        cache_key = f"{text[:100]}:{str(kwargs)}"

        if cache_key in cache:
            self._state_manager.set_metadata("cache_hit", True)
            logger.debug(f"Cache hit for validation: {cache_key[:30]}...")
            return cache[cache_key]

        # Mark as cache miss
        self._state_manager.set_metadata("cache_hit", False)
        logger.debug(f"Cache miss for validation: {cache_key[:30]}...")

        # Start timing
        start_time = time.time()

        # Extract validation options
        fail_fast = kwargs.pop("fail_fast", False)
        timeout = kwargs.pop("timeout", None)
        include_rules = kwargs.pop("include_rules", None)
        exclude_rules = kwargs.pop("exclude_rules", None)

        # Filter rules if needed
        rules_to_validate = self._get_prioritized_rules()
        if include_rules:
            rules_to_validate = [r for r in rules_to_validate if r.name in include_rules]
        if exclude_rules:
            rules_to_validate = [r for r in rules_to_validate if r.name not in exclude_rules]

        # Log validation start
        logger.debug(f"Starting validation with {len(rules_to_validate)} rules")

        results = []
        for rule in rules_to_validate:
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                logger.warning(f"Validation timeout after {timeout} seconds")
                break

            try:
                # Validate with the rule
                result = rule.validate(text, **kwargs)
                results.append(result)

                # Track pass/fail statistics
                if result.passed:
                    pass_count = self._state_manager.get_metadata("pass_count", 0)
                    self._state_manager.set_metadata("pass_count", pass_count + 1)
                else:
                    fail_count = self._state_manager.get_metadata("fail_count", 0)
                    self._state_manager.set_metadata("fail_count", fail_count + 1)

                    # Track failing rules
                    failing_rules = self._state_manager.get_metadata("failing_rules", {})
                    failing_rules[rule.name] = failing_rules.get(rule.name, 0) + 1
                    self._state_manager.set_metadata("failing_rules", failing_rules)

                    # Stop if fail_fast is enabled
                    if fail_fast:
                        logger.debug(f"Stopping validation early due to fail_fast option")
                        break

            except Exception as e:
                # Handle rule validation errors
                logger.error(f"Error validating with rule {rule.name}: {e}")

                # Create error result
                from ..utils import create_error_result

                error_result = create_error_result(e, rule.name)
                results.append(error_result)

                # Stop if fail_fast is enabled
                if fail_fast:
                    logger.debug(f"Stopping validation early due to error and fail_fast option")
                    break

        # Record execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Log validation completion
        logger.debug(f"Validation completed in {execution_time:.2f}s with {len(results)} results")

        # Update average execution time
        avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
        count = self._state_manager.get("validation_count", 1)
        new_avg = ((avg_time * (count - 1)) + execution_time) / count
        self._state_manager.set_metadata("avg_execution_time", new_avg)

        # Update max execution time if needed
        max_time = self._state_manager.get_metadata("max_execution_time", 0)
        if execution_time > max_time:
            self._state_manager.set_metadata("max_execution_time", execution_time)

        # Cache results (limit cache size)
        cache_size = self.config.cache_size
        if len(cache) >= cache_size:
            # Simple strategy: just clear the cache when it gets full
            logger.debug(f"Clearing validation cache (size: {len(cache)})")
            cache = {}

        cache[cache_key] = results
        self._state_manager.update("result_cache", cache)

        return results

    def _get_prioritized_rules(self) -> List[RuleProtocol]:
        """
        Get rules in priority order.

        Returns:
            List of rules in priority order

        Notes:
            If prioritize_by_cost is True, rules are sorted by cost (ascending).
            Otherwise, rules are sorted by priority (descending).
        """
        if self.config.prioritize_by_cost:
            return sorted(self.rules, key=lambda r: r.config.cost)
        return sorted(self.rules, key=lambda r: r.config.priority.value, reverse=True)

    def add_rule(self, rule: RuleProtocol) -> None:
        """
        Add a rule to the manager.

        Args:
            rule: Rule to add

        Examples:
            ```python
            from sifaka.rules.formatting.length import create_length_rule

            # Create a rule
            rule = create_length_rule(min_length=10, max_length=1000)

            # Add the rule to the manager
            manager.add_rule(rule)
            ```
        """
        # Check if rule already exists
        if any(r.name == rule.name for r in self.rules):
            logger.warning(f"Rule with name '{rule.name}' already exists, replacing")
            self.remove_rule(rule.name)

        # Add the rule
        self.rules.append(rule)
        logger.debug(f"Added rule '{rule.name}' to validation manager")

        # Update state
        self._state_manager.update("rules", self.rules)
        self._state_manager.set_metadata("rule_count", len(self.rules))

    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from the manager.

        Args:
            rule_name: Name of the rule to remove

        Examples:
            ```python
            # Remove a rule by name
            manager.remove_rule("length_rule")
            ```
        """
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]

        # Check if rule was removed
        if len(self.rules) < original_count:
            logger.debug(f"Removed rule '{rule_name}' from validation manager")
        else:
            logger.warning(f"Rule '{rule_name}' not found in validation manager")

        # Update state
        self._state_manager.update("rules", self.rules)
        self._state_manager.set_metadata("rule_count", len(self.rules))

    def get_rule(self, rule_name: str) -> Optional[RuleProtocol]:
        """
        Get a rule by name.

        Args:
            rule_name: Name of the rule to get

        Returns:
            The rule if found, None otherwise

        Examples:
            ```python
            # Get a rule by name
            rule = manager.get_rule("length_rule")
            if rule:
                print(f"Rule description: {rule.description}")
            ```
        """
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation statistics

        Examples:
            ```python
            # Get validation statistics
            stats = manager.get_statistics()
            print(f"Validation count: {stats['validation_count']}")
            print(f"Pass rate: {stats['pass_count'] / (stats['validation_count'] or 1):.2%}")
            ```
        """
        # Calculate uptime
        start_time = self._state_manager.get("start_time", time.time())
        uptime_seconds = time.time() - start_time

        # Calculate pass rate
        validation_count = self._state_manager.get("validation_count", 0)
        pass_count = self._state_manager.get_metadata("pass_count", 0)
        pass_rate = pass_count / validation_count if validation_count > 0 else 0

        return {
            "validation_count": validation_count,
            "pass_count": pass_count,
            "fail_count": self._state_manager.get_metadata("fail_count", 0),
            "pass_rate": pass_rate,
            "avg_execution_time": self._state_manager.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state_manager.get_metadata("max_execution_time", 0),
            "failing_rules": self._state_manager.get_metadata("failing_rules", {}),
            "rule_count": self._state_manager.get_metadata("rule_count", 0),
            "cache_size": len(self._state_manager.get("result_cache", {})),
            "cache_hit_rate": (
                self._state_manager.get_metadata("cache_hit_count", 0) / validation_count
                if validation_count > 0
                else 0
            ),
            "uptime_seconds": uptime_seconds,
            "rules": [r.name for r in self.rules],
        }

    def reset_statistics(self) -> None:
        """
        Reset validation statistics.

        This method resets all validation statistics but keeps the rules and configuration.

        Examples:
            ```python
            # Reset statistics
            manager.reset_statistics()
            ```
        """
        logger.debug("Resetting validation statistics")
        self._state_manager.update("validation_count", 0)
        self._state_manager.update("start_time", time.time())
        self._state_manager.set_metadata("pass_count", 0)
        self._state_manager.set_metadata("fail_count", 0)
        self._state_manager.set_metadata("failing_rules", {})
        self._state_manager.set_metadata("avg_execution_time", 0)
        self._state_manager.set_metadata("max_execution_time", 0)
        self._state_manager.set_metadata("cache_hit_count", 0)

    def clear_cache(self) -> None:
        """
        Clear the validation result cache.

        This method removes all cached validation results.

        Examples:
            ```python
            # Clear cache
            manager.clear_cache()
            ```
        """
        logger.debug("Clearing validation cache")
        self._state_manager.update("result_cache", {})

    def validate_all(self, texts: List[str], **kwargs: Any) -> Dict[str, List[RuleResult]]:
        """
        Validate multiple texts against all managed rules.

        Args:
            texts: List of texts to validate
            **kwargs: Additional validation options

        Returns:
            Dictionary mapping texts to validation results

        Examples:
            ```python
            # Validate multiple texts
            texts = ["Text 1", "Text 2", "Text 3"]
            results = manager.validate_all(texts)

            # Check results
            for text, text_results in results.items():
                print(f"Text: {text}")
                for result in text_results:
                    print(f"  Rule: {result.rule_name}, Passed: {result.passed}")
            ```
        """
        logger.debug(f"Validating {len(texts)} texts")
        results = {}

        for text in texts:
            results[text] = self.validate(text, **kwargs)

        return results
