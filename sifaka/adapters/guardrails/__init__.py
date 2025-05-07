"""
Adapter for using Guardrails validators as rules.

This module provides adapters for using validators from the Guardrails library as Sifaka rules.
It enables seamless integration between Guardrails' validation capabilities and Sifaka's
rule system, allowing for sophisticated content validation.

Usage Example:
    from guardrails.hub import RegexMatch
    from sifaka.adapters.guardrails import create_guardrails_rule

    # Create a Guardrails validator
    regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

    # Create a Sifaka rule using the Guardrails validator
    phone_rule = create_guardrails_rule(
        guardrails_validator=regex_validator,
        rule_id="phone_number_format"
    )

    # Use it in a Sifaka chain
    result = chain.run("What's a good phone number format?")
"""

from sifaka.adapters.guardrails.adapter import (
    GuardrailsValidatable,
    GuardrailsValidatorAdapter,
    GuardrailsRule,
    create_guardrails_rule,
)

# Export public classes and functions
__all__ = [
    # Protocols
    "GuardrailsValidatable",
    # Core components
    "GuardrailsValidatorAdapter",
    "GuardrailsRule",
    # Factory functions
    "create_guardrails_rule",
]
