"""
Factory functions for Sifaka rules.

This module provides factory functions for creating rules and related components.
These functions provide a consistent way to create rules across the framework.

Usage Example:
    ```python
    from sifaka.rules.factories import create_rule, create_validation_manager
    from sifaka.rules.formatting.length import create_length_validator

    # Create a validator
    validator = create_length_validator(min_length=10, max_length=1000)

    # Create a rule with the validator
    rule = create_rule(
        name="length_rule",
        description="Validates text length",
        validator=validator
    )

    # Create a validation manager with the rule
    manager = create_validation_manager(rules=[rule])

    # Validate text
    results = manager.validate("This is a test.")
    ```
"""

from typing import Any, List, Optional, Type

from .base import Rule
from .validators import BaseValidator
from .config import RuleConfig
from .managers.validation import ValidationManager, ValidationConfig
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


def create_rule(
    name: str,
    validator: BaseValidator,
    description: Optional[str] = None,
    config: Optional[RuleConfig] = None,
    rule_type: Type[Rule] = Rule,
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a rule with the given validator and configuration.

    This factory function provides a consistent way to create rules
    across the Sifaka framework.

    Args:
        name: Name of the rule
        validator: Validator to use for validation
        description: Description of the rule
        config: Configuration for the rule
        rule_type: Type of rule to create
        rule_id: Unique identifier for the rule
        **kwargs: Additional arguments for the rule constructor including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        A new rule instance

    Examples:
        ```python
        from sifaka.rules.factories import create_rule
        from sifaka.rules.formatting.length import create_length_validator

        # Create a validator
        validator = create_length_validator(min_length=10, max_length=1000)

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
            tags=["length", "formatting", "validation"]
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
        logger.error(f"Error creating rule: {e}")
        raise ValueError(f"Error creating rule: {str(e)}")


def create_validation_manager(
    rules: Optional[List[Rule]] = None,
    prioritize_by_cost: bool = False,
    **kwargs: Any,
) -> ValidationManager:
    """
    Create a validation manager with the given rules and configuration.

    This factory function provides a consistent way to create validation
    managers across the Sifaka framework.

    Args:
        rules: List of rules to manage
        prioritize_by_cost: Whether to prioritize rules by cost
        **kwargs: Additional configuration parameters

    Returns:
        A new validation manager instance

    Examples:
        ```python
        from sifaka.rules.factories import create_validation_manager
        from sifaka.rules.formatting.length import create_length_rule
        from sifaka.rules.formatting.format import create_markdown_rule

        # Create rules
        length_rule = create_length_rule(min_length=10, max_length=1000)
        markdown_rule = create_markdown_rule(required_elements=["#", "*", "`"])

        # Create a validation manager with rules
        manager = create_validation_manager(
            rules=[length_rule, markdown_rule],
            prioritize_by_cost=True
        )

        # Validate text
        results = manager.validate("# Heading\n\n* List item")

        # Get validation statistics
        stats = manager.get_statistics()
        print(f"Validation count: {stats['validation_count']}")
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
        logger.error(f"Error creating validation manager: {e}")
        raise ValueError(f"Error creating validation manager: {str(e)}")
