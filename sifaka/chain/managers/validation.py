"""
Validation Manager Module

## Overview
This module provides the ValidationManager class which handles validation of outputs
against rules. It manages rule registration, validation execution, and error message
generation, providing a centralized way to handle validation in the chain system.

## Components
1. **ValidationManager**: Main validation management class
   - Rule management
   - Validation execution
   - Error message generation
   - State management
   - Performance tracking

2. **ValidationConfig**: Configuration for validation manager
   - Rule prioritization options
   - Fail-fast behavior
   - Cache settings

3. **ValidationResult**: Result of validation operation
   - Overall validation status
   - Individual rule results
   - Issues and suggestions
   - Performance metrics

## Architecture
The ValidationManager follows a component-based architecture:
- Inherits from BaseComponent for consistent behavior
- Uses StateManager for state management
- Implements caching for performance
- Tracks statistics for monitoring

## Usage Examples
```python
from sifaka.chain.managers.validation import create_validation_manager
from sifaka.rules import create_length_rule, create_toxicity_rule

# Create rules
rules = [
    create_length_rule(min_length=10, max_length=1000),
    create_toxicity_rule(threshold=0.7)
]

# Create validation manager using factory function
manager = create_validation_manager(
    rules=rules,
    name="content_validator",
    description="Validates content length and toxicity",
    prioritize_by_cost=True,
    fail_fast=True,
    cache_size=100
)

# Validate output
result = manager.validate("Some output text")

# Check validation result
if result.passed:
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

# Get statistics
stats = manager.get_statistics()
print(f"Validation count: {stats['validation_count']}")
print(f"Success rate: {stats['success_rate']:.2f}")
```

## Error Handling
- ValueError: Raised for invalid rules or rule operations
- TypeError: Raised for type validation failures

## Configuration
- rules: List of validation rules to apply
- prioritize_by_cost: Whether to sort rules by cost (lowest first)
- fail_fast: Whether to stop after first failure
- cache_size: Maximum number of cached validation results
"""

from typing import Any, Dict, List, Optional, TypeVar
import time

from pydantic import Field, ConfigDict

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult
from sifaka.rules.base import Rule as BaseRule, RuleResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
OutputType = TypeVar("OutputType")


class ValidationConfig(BaseConfig):
    """Configuration for validation manager."""

    prioritize_by_cost: bool = Field(
        default=False, description="Whether to prioritize rules by cost (lowest first)"
    )
    fail_fast: bool = Field(
        default=False, description="Whether to stop validation after first failure"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class ValidationResult(BaseResult):
    """Result of validation operation."""

    rule_results: List[RuleResult] = Field(default_factory=list)
    all_passed: bool = Field(default=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class ValidationManager(BaseComponent[str, ValidationResult]):
    """
    Validation manager for Sifaka chains.

    This class provides rule-based validation of generated text,
    coordinating between multiple rules and tracking validation results.

    ## Architecture
    The ValidationManager follows a component-based architecture:
    - Inherits from BaseComponent for consistent behavior
    - Uses StateManager for state management
    - Implements caching for performance
    - Tracks statistics for monitoring

    ## Lifecycle
    1. Initialization: Set up with rules and configuration
    2. Validation: Validate text against rules
    3. Rule Management: Add/remove rules as needed
    4. Statistics: Track validation performance
    """

    def __init__(
        self,
        rules: List[BaseRule[OutputType]],
        name: str = "validation_manager",
        description: str = "Validation manager for Sifaka chains",
        prioritize_by_cost: bool = False,
        fail_fast: bool = False,
        config: Optional[ValidationConfig] = None,
        **kwargs: Any,
    ):
        """Initialize the validation manager.

        Args:
            rules: List of rules to use for validation
            name: Name of the manager
            description: Description of the manager
            prioritize_by_cost: Whether to prioritize rules by cost
            fail_fast: Whether to stop after first failure
            config: Additional configuration
            **kwargs: Additional keyword arguments for configuration
        """
        # Create config if not provided
        if config is None:
            config = ValidationConfig(
                name=name,
                description=description,
                prioritize_by_cost=prioritize_by_cost,
                fail_fast=fail_fast,
                **kwargs,
            )

        # Initialize base component
        super().__init__(name, description, config)

        # Store rules in state
        self._state.update("rules", rules)
        self._state.update("result_cache", {})
        self._state.update("initialized", True)

        # Set metadata
        self._state.set_metadata("component_type", "validation_manager")
        self._state.set_metadata("creation_time", time.time())
        self._state.set_metadata("rule_count", len(rules))

    def process(self, input: str) -> ValidationResult:
        """
        Process the input text and return a validation result.

        This is the implementation of the abstract method from BaseComponent.

        Args:
            input: The text to validate

        Returns:
            ValidationResult with validation details

        Raises:
            ValueError: If input is invalid
        """
        return self.validate(input)

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against all rules.

        Args:
            text: The text to validate

        Returns:
            ValidationResult with rule results and overall status

        Raises:
            ValueError: If text is invalid
        """
        # Handle empty input
        empty_result = self.handle_empty_input(text)
        if empty_result:
            return ValidationResult(
                passed=False,
                message="Empty input",
                metadata={"error_type": "empty_input"},
                score=0.0,
                issues=["Input is empty"],
                suggestions=["Provide non-empty input"],
                rule_results=[],
                all_passed=False,
            )

        # Record start time
        start_time = time.time()

        try:
            # Check cache
            cache_key = text[:100]  # Use first 100 chars as key
            cache = self._state.get("result_cache", {})

            if cache_key in cache and self.config.cache_size > 0:
                self._state.set_metadata("cache_hit", True)
                return cache[cache_key]

            # Mark as cache miss
            self._state.set_metadata("cache_hit", False)

            # Get rules from state
            rules = self._state.get("rules", [])

            # Sort rules by cost if configured
            if self.config.prioritize_by_cost:
                rules = sorted(rules, key=lambda r: getattr(r, "cost", 1.0))

            # Run validation
            rule_results = []
            all_passed = True
            issues = []
            suggestions = []

            for rule in rules:
                result = rule.validate(text)
                rule_results.append(result)

                # Track if all rules passed
                if not result.passed:
                    all_passed = False
                    issues.extend(result.issues)
                    suggestions.extend(result.suggestions)

                    # Stop after first failure if configured
                    if self.config.fail_fast:
                        break

            # Calculate processing time
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            # Create validation result
            validation_result = ValidationResult(
                passed=all_passed,
                message="Validation passed" if all_passed else "Validation failed",
                metadata={
                    "rule_count": len(rules),
                    "rules_executed": len(rule_results),
                    "cache_hit": False,
                },
                score=1.0 if all_passed else 0.0,
                issues=issues,
                suggestions=suggestions,
                rule_results=rule_results,
                all_passed=all_passed,
                processing_time_ms=processing_time_ms,
            )

            # Update statistics
            self.update_statistics(validation_result)

            # Cache result if caching is enabled
            if self.config.cache_size > 0:
                # Manage cache size
                if len(cache) >= self.config.cache_size:
                    # Remove oldest entry (simple approach)
                    if cache:
                        oldest_key = next(iter(cache))
                        del cache[oldest_key]

                cache[cache_key] = validation_result
                self._state.update("result_cache", cache)

            return validation_result

        except Exception as e:
            # Record error
            self.record_error(e)
            logger.error(f"Validation error: {str(e)}")

            # Create error result
            return ValidationResult(
                passed=False,
                message=f"Validation error: {str(e)}",
                metadata={"error_type": str(type(e).__name__)},
                score=0.0,
                issues=[str(e)],
                suggestions=["Check input format and try again"],
                rule_results=[],
                all_passed=False,
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about validation usage.

        Returns:
            Dictionary with usage statistics
        """
        # Get base statistics from parent class
        stats = super().get_statistics()

        # Add validation-specific statistics
        stats.update(
            {
                "cache_size": len(self._state.get("result_cache", {})),
                "rule_count": len(self._state.get("rules", [])),
                "prioritize_by_cost": self.config.prioritize_by_cost,
                "fail_fast": self.config.fail_fast,
                "cache_enabled": self.config.cache_size > 0,
                "cache_limit": self.config.cache_size,
            }
        )

        return stats

    def clear_cache(self) -> None:
        """Clear the validation result cache."""
        self._state.update("result_cache", {})
        logger.debug(f"Validation cache cleared for {self.name}")

    def add_rule(self, rule: Any) -> None:
        """
        Add a rule for validation.

        ## Overview
        This method adds a new validation rule to the manager's rule list.
        The rule will be used in subsequent validations.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate rule type
           - Check rule validity

        2. **Rule Addition**: Add rule
           - Add to rule list
           - Update state
           - Clear cache

        Args:
            rule: The rule to add

        Raises:
            ValueError: If the rule is invalid
            TypeError: If the input type is incorrect

        Examples:
            ```python
            manager = ValidationManager(rules=[create_length_rule(min_length=10)])
            new_rule = create_length_rule(max_length=1000)
            manager.add_rule(new_rule)
            ```
        """
        # Validate rule type
        if not isinstance(rule, BaseRule):
            raise ValueError(f"Expected BaseRule instance, got {type(rule)}")

        # Check for duplicate rule names
        rules = self._state.get("rules", [])
        if any(r.name == rule.name for r in rules):
            logger.warning(f"Rule with name '{rule.name}' already exists, it will be replaced")
            # Remove existing rule with same name
            self.remove_rule(rule.name)
            # Get updated rules list
            rules = self._state.get("rules", [])

        # Add rule to the list
        rules.append(rule)
        self._state.update("rules", rules)

        # Update metadata
        self._state.set_metadata("rule_count", len(rules))

        # Clear cache since validation results may change
        self.clear_cache()

        logger.debug(f"Added rule '{rule.name}' to validation manager '{self.name}'")

    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule by name.

        ## Overview
        This method removes a validation rule from the manager's rule list
        based on its name. The rule will no longer be used in subsequent
        validations.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate rule name
           - Find rule to remove

        2. **Rule Removal**: Remove rule
           - Remove from rule list
           - Update state
           - Clear cache

        Args:
            rule_name: The name of the rule to remove

        Raises:
            ValueError: If the rule name is invalid or rule not found

        Examples:
            ```python
            manager = ValidationManager(rules=[create_length_rule(min_length=10)])
            manager.remove_rule("length_rule")
            ```
        """
        # Validate input
        if not rule_name or not isinstance(rule_name, str):
            raise ValueError(f"Invalid rule name: {rule_name}")

        # Find rule by name
        rule_to_remove = None
        rules = self._state.get("rules", [])
        for rule in rules:
            if rule.name == rule_name:
                rule_to_remove = rule
                break

        if rule_to_remove is None:
            raise ValueError(f"Rule not found: {rule_name}")

        # Remove rule from list
        rules.remove(rule_to_remove)
        self._state.update("rules", rules)

        # Update metadata
        self._state.set_metadata("rule_count", len(rules))

        # Clear cache since validation results may change
        self.clear_cache()

        logger.debug(f"Removed rule '{rule_name}' from validation manager '{self.name}'")

    def get_rules(self) -> List[BaseRule]:
        """
        Get all registered rules.

        ## Overview
        This method returns a list of all validation rules currently
        registered with the manager.

        ## Lifecycle
        1. **Rule Retrieval**: Get rules
           - Access rule list
           - Return rules

        Returns:
            The list of registered rules

        Examples:
            ```python
            manager = ValidationManager(rules=[create_length_rule(min_length=10)])
            rules = manager.get_rules()
            print(f"Number of rules: {len(rules)}")
            ```
        """
        return self._state.get("rules", [])

    def get_error_messages(self, validation_result: ValidationResult) -> List[str]:
        """
        Get error messages from validation result.

        Args:
            validation_result: The validation result to extract messages from

        Returns:
            List of error messages
        """
        messages = []

        # Add overall issues
        messages.extend(validation_result.issues)

        # Add rule-specific issues
        for result in validation_result.rule_results:
            if not result.passed:
                messages.extend(result.issues)

        return messages

    def warm_up(self) -> None:
        """Prepare the validation manager for use."""
        super().warm_up()

        # Pre-validate rules
        rules = self._state.get("rules", [])
        for rule in rules:
            if hasattr(rule, "warm_up"):
                rule.warm_up()

        logger.debug(f"Validation manager '{self.name}' warmed up with {len(rules)} rules")


def create_validation_manager(
    rules: List[BaseRule],
    name: str = "validation_manager",
    description: str = "Validation manager for Sifaka chains",
    prioritize_by_cost: bool = False,
    fail_fast: bool = False,
    cache_size: int = 100,
    **kwargs: Any,
) -> ValidationManager:
    """
    Create a validation manager.

    Args:
        rules: List of rules to use for validation
        name: Name of the manager
        description: Description of the manager
        prioritize_by_cost: Whether to prioritize rules by cost
        fail_fast: Whether to stop after first failure
        cache_size: Size of the validation cache
        **kwargs: Additional configuration parameters

    Returns:
        Configured ValidationManager instance
    """
    config = ValidationConfig(
        name=name,
        description=description,
        prioritize_by_cost=prioritize_by_cost,
        fail_fast=fail_fast,
        cache_size=cache_size,
        **kwargs,
    )

    return ValidationManager(
        rules=rules,
        name=name,
        description=description,
        config=config,
    )
