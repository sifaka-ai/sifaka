from typing import Any, List

"""
Sifaka Rules Package

This package provides a collection of validation rules for content validation in the
Sifaka framework. Each rule implements specific validation logic and follows a
consistent interface for easy integration and extension.

Architecture Overview:
- Each rule inherits from the base Rule class
- Rules follow the single responsibility principle
- All rules return RuleResult objects with consistent structure
- Rules can be combined in a Chain for comprehensive validation

Module Structure:
- base.py: Base rule implementations
- config.py: Rule configuration
- result.py: Rule result models
- factories.py: Factory functions for creating rules
- interfaces/: Protocol interfaces for rules
- managers/: Component managers for validation
- content/: Content-based rules
- formatting/: Format-based rules

Usage Example:
    from sifaka.chain import ChainOrchestrator
    from sifaka.models import OpenAIProvider
    from sifaka.rules import create_length_rule, create_prohibited_content_rule
    from sifaka.critics.prompt import create_prompt_critic

    # Configure provider
    provider = OpenAIProvider(
        model_name="gpt-4-turbo-preview",
        config={"api_key": "your-api-key"}
    )

    # Create rules with factory functions
    length_rule = create_length_rule(
        min_length=10,
        max_length=1000,
        priority="MEDIUM"
    )

    content_rule = create_prohibited_content_rule(
        prohibited_terms=["bad", "inappropriate"],
        case_sensitive=False,
        priority="HIGH"
    )

    # Create critic for content improvement
    critic = create_prompt_critic(
        model=provider,
        system_prompt="You are an expert editor that improves text while maintaining its original meaning."
    )

    # Create chain with rules and critic
    chain = ChainOrchestrator(
        model=provider,
        rules=[length_rule, content_rule],
        critic=critic,
        max_attempts=3
    )

    # Validate and improve content
    result = chain.run("Your content here") if chain else ""

    # Access validation results
    print(f"Content: {result.content}")
    print(f"Validation passed: {result.passed_validation}")
    if not result.passed_validation:
        print("Failed rules:")
        for rule_name, rule_result in result.rule_results.items() if rule_results else "":
            if not rule_result.passed:
                print(f"- {rule_name}: {rule_result.message}")
"""
from .base import Rule, RuleConfig, FunctionRule
from ..core.results import RuleResult
from sifaka.utils.config.rules import RulePriority
from .factories import create_rule, create_validation_manager
from sifaka.interfaces import (
    RuleProtocol,
    SimpleRuleProtocol,
    RuleResultHandlerProtocol,
    ValidatableProtocol,
)
from .validators import BaseValidator, FunctionValidator, RuleValidator
from .managers.validation import ValidationManager
from .utils import try_validate
from sifaka.utils.results import create_rule_result, create_error_result
from .content.prohibited import create_prohibited_content_rule
from .content.safety import create_toxicity_rule, create_bias_rule, create_harmful_content_rule
from .content.sentiment import create_sentiment_rule
from .formatting.length import create_length_rule
from .formatting.structure import create_structure_rule

__all__: List[Any] = [
    "Rule",
    "RuleConfig",
    "RuleResult",
    "RulePriority",
    "FunctionRule",
    "BaseValidator",
    "FunctionValidator",
    "RuleValidator",
    "RuleProtocol",
    "SimpleRuleProtocol",
    "RuleResultHandlerProtocol",
    "ValidatableProtocol",
    "ValidationManager",
    "create_rule",
    "create_validation_manager",
    "create_rule_result",
    "create_error_result",
    "try_validate",
    "create_prohibited_content_rule",
    "create_toxicity_rule",
    "create_bias_rule",
    "create_harmful_content_rule",
    "create_sentiment_rule",
    "create_length_rule",
    "create_structure_rule",
]
