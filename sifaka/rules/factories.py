"""
Factory Functions for Sifaka Rules

This module provides factory functions for creating rules and related components,
ensuring a consistent way to instantiate and configure rules across the framework.

## Overview
Factory functions simplify the creation of complex objects by encapsulating
initialization logic and providing sensible defaults. The rule factories in this
module make it easy to create rules, validators, and validation managers with
consistent configuration and error handling.

## Components
- create_rule: Creates a rule with the specified validator and configuration
- create_validation_manager: Creates a validation manager with the specified rules

## Usage Examples
```python
from sifaka.rules.factories import create_rule, create_validation_manager
from sifaka.rules.validators import FunctionValidator

# Define a validation function
def validate_length(text: str) -> bool:
    return len(text) >= 10 and len(text) <= 1000

# Create a validator
validator = FunctionValidator(
    func=validate_length,
    validation_type=str
)

# Create a rule with the validator
rule = create_rule(
    name="length_rule",
    description="Validates text length",
    validator=validator,
    severity="warning",
    category="formatting",
    tags=["length", "formatting"]
)

# Create a validation manager with the rule
manager = create_validation_manager(
    rules=[rule],
    prioritize_by_cost=True
)

# Validate text
results = manager.validate("This is a test.") if manager else ""
print(f"Validation passed: {results.passed}")
```

## Error Handling
The factory functions include comprehensive error handling:
- Input validation to ensure required parameters are provided
- Type checking to ensure parameters are of the correct type
- Exception handling to provide clear error messages
- Logging of errors for debugging

## Configuration
Factory functions accept configuration parameters directly or as configuration objects:
- Direct parameters: Pass configuration options as keyword arguments
- Configuration objects: Pass pre-configured RuleConfig or ValidationConfig objects
- Mixed approach: Pass some options directly and others via configuration objects
"""

from typing import Any, List, Optional, Type

from .base import Rule
from .validators import BaseValidator
from sifaka.utils.config.rules import RuleConfig
from .managers.validation import ValidationManager, ValidationConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


def create_rule(
    name: str,
    validator: BaseValidator,
    description: Optional[Optional[str]] = None,
    config: Optional[Optional[RuleConfig]] = None,
    rule_type: Type[Rule] = Rule,
    rule_id: Optional[Optional[str]] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a rule with the given validator and configuration.

    This factory function provides a consistent way to create rules
    across the Sifaka framework. It handles configuration creation,
    parameter validation, and error handling.

    Args:
        name: Name of the rule
        validator: Validator to use for validation
        description: Description of the rule
        config: Configuration for the rule
        rule_type: Type of rule to create
        rule_id: Unique identifier for the rule
        **kwargs: Additional arguments for the rule constructor including:
            - severity: Severity level for rule violations (error, warning, info)
            - category: Category of the rule (formatting, content, etc.)
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation (LOW, MEDIUM, HIGH, CRITICAL)
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        A new rule instance

    Raises:
        ValueError: If the rule cannot be created due to invalid parameters
        TypeError: If the validator is not compatible with the rule type

    Examples:
        ```python
        from sifaka.rules.factories import create_rule
        from sifaka.rules.validators import FunctionValidator

        # Define a validation function
        def validate_length(text: str) -> bool:
            return len(text) >= 10 and len(text) <= 1000

        # Create a validator
        validator = FunctionValidator(
            func=validate_length,
            validation_type=str
        )

        # Create a basic rule
        rule = create_rule(
            name="length_rule",
            description="Validates text length",
            validator=validator
        )

        # Create a rule with metadata
        rule = create_rule(
            name="length_rule",
            description="Validates text length",
            validator=validator,
            rule_id="text_length_validator",
            severity="warning",
            category="formatting",
            tags=["length", "formatting", "validation"],
            priority="HIGH",
            cache_size=100,
            cost=2
        )

        # Create a rule with a custom rule type
        from sifaka.rules.content.safety import SafetyRule
        safety_rule = create_rule(
            name="safety_rule",
            description="Validates content safety",
            validator=safety_validator,
            rule_type=SafetyRule
        )
        ```
    """
    try:
        # Set default description if not provided
        description = description or f"Rule for {name}"

        # Determine rule ID
        rule_id = rule_id or name

        # Create config if not provided
        if config is None:
            config = RuleConfig(
                name=name,
                description=description,
                rule_id=rule_id,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ["priority", "cache_size", "cost", "severity", "category", "tags"]
                },
            )

        # Create and return rule
        return rule_type(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["priority", "cache_size", "cost", "severity", "category", "tags"]
            },
        )

    except Exception as e:
        if logger:
            logger.error(f"Error creating rule: {e}")
        raise ValueError(f"Error creating rule: {str(e)}")


def create_validation_manager(
    rules: Optional[Optional[List[Rule]]] = None,
    prioritize_by_cost: bool = False,
    **kwargs: Any,
) -> ValidationManager:
    """
    Create a validation manager with the given rules and configuration.

    This factory function provides a consistent way to create validation
    managers across the Sifaka framework. The validation manager orchestrates
    the execution of multiple rules, handles rule prioritization, and
    aggregates validation results.

    Args:
        rules: List of rules to manage
        prioritize_by_cost: Whether to prioritize rules by cost
        **kwargs: Additional configuration parameters including:
            - max_cache_size: Maximum size of the validation cache
            - parallel: Whether to run validations in parallel
            - timeout: Timeout for validation operations in seconds
            - fail_fast: Whether to stop validation after the first failure

    Returns:
        A new validation manager instance

    Raises:
        ValueError: If the validation manager cannot be created due to invalid parameters

    Examples:
        ```python
        from sifaka.rules.factories import create_validation_manager, create_rule
        from sifaka.rules.validators import FunctionValidator

        # Define validation functions
        def validate_length(text: str) -> bool:
            return len(text) >= 10 and len(text) <= 1000

        def validate_markdown(text: str) -> bool:
            return "#" in text and "*" in text

        # Create validators
        length_validator = FunctionValidator(func=validate_length, validation_type=str)
        markdown_validator = FunctionValidator(func=validate_markdown, validation_type=str)

        # Create rules
        length_rule = create_rule(
            name="length_rule",
            description="Validates text length",
            validator=length_validator,
            severity="warning"
        )

        markdown_rule = create_rule(
            name="markdown_rule",
            description="Validates markdown formatting",
            validator=markdown_validator,
            severity="info"
        )

        # Create a validation manager with rules
        manager = create_validation_manager(
            rules=[length_rule, markdown_rule],
            prioritize_by_cost=True,
            parallel=True,
            timeout=5.0,
            fail_fast=False
        )

        # Validate text
        results = manager.validate("# Heading\n\n* List item")

        # Check if all validations passed
        if results.all_passed():
            print("All validations passed!")
        else:
            print("Some validations failed:")
            for result in results.failed_results():
                print(f"- {result.rule_id}: {result.message}")

        # Get validation statistics
        stats = manager.get_statistics()
        print(f"Validation count: {stats['validation_count']}")
        print(f"Average processing time: {stats['avg_processing_time_ms']} ms")
        ```
    """
    try:
        # Create configuration
        config = ValidationConfig(prioritize_by_cost=prioritize_by_cost, params=kwargs)

        # Create and return manager
        manager = ValidationManager(rules=rules, config=config)

        # Log creation
        if rules:
            logger.debug(f"Created validation manager with {len(rules)} rules")

        return manager

    except Exception as e:
        if logger:
            logger.error(f"Error creating validation manager: {e}")
        raise ValueError(f"Error creating validation manager: {str(e)}")
