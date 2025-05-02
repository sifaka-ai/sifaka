"""
Adapter for using Guardrails validators as rules.

This module provides adapters for using validators from the Guardrails library as Sifaka rules.

Usage Example:
    from guardrails.hub import RegexMatch
    from sifaka.adapters.rules import create_guardrails_rule

    # Create a Guardrails validator
    regex_validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")

    # Create a Sifaka rule using the Guardrails validator
    phone_rule = create_guardrails_rule(
        guardrails_validator=regex_validator,
        rule_id="phone_number_format"
    )

    # Use it in a Sifaka chain
    result = chain.run("What's a good phone number format?")
"""

from typing import Any, Dict, Optional, Union, Protocol, runtime_checkable

try:
    from guardrails.validator_base import Validator as GuardrailsValidator
    from guardrails.classes import ValidationResult, PassResult, FailResult

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

    # Create placeholder classes for type hints when Guardrails isn't installed
    class GuardrailsValidator:
        pass

    class ValidationResult:
        pass

    class PassResult(ValidationResult):
        pass

    class FailResult(ValidationResult):
        pass


from sifaka.rules.base import BaseValidator, RuleResult, Rule, FunctionRule
from sifaka.adapters.rules.base import Adaptable, BaseAdapter


class GuardrailsValidatorAdapter(BaseValidator[str]):
    """Adapter that allows Guardrails validators to be used within Sifaka."""

    def __init__(self, guardrails_validator: GuardrailsValidator):
        """Initialize with a Guardrails validator.

        Args:
            guardrails_validator: The Guardrails validator to adapt
        """
        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )
        self._guardrails_validator = guardrails_validator

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate text using the Guardrails validator.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult: Sifaka validation result
        """
        # Handle empty text (Sifaka's convention)
        empty_result = self.handle_empty_text(output)
        if empty_result:
            return empty_result

        # Call the Guardrails validator
        gr_result = self._guardrails_validator.validate(output, metadata=kwargs)

        # Convert Guardrails result to Sifaka RuleResult
        if isinstance(gr_result, PassResult):
            return RuleResult(
                passed=True,
                message="Validation passed",
                metadata={"guardrails_metadata": gr_result.__dict__},
            )
        else:
            # It's a FailResult
            error_message = "Validation failed"
            if hasattr(gr_result, "get_failure_reason"):
                error_message = gr_result.get_failure_reason()
            elif hasattr(gr_result, "error_message"):
                error_message = gr_result.error_message

            return RuleResult(
                passed=False,
                message=error_message,
                metadata={"guardrails_metadata": gr_result.__dict__, "errors": [error_message]},
            )

    @property
    def validation_type(self) -> type:
        """Get the type this validator can validate."""
        return str


def create_guardrails_rule(
    guardrails_validator: GuardrailsValidator, rule_id: Optional[str] = None, **kwargs
) -> Rule:
    """Create a Sifaka rule that uses a Guardrails validator.

    Args:
        guardrails_validator: The Guardrails validator to adapt
        rule_id: Identifier for the rule
        **kwargs: Additional configuration for the rule

    Returns:
        Configured Rule using the Guardrails validator
    """
    if not GUARDRAILS_AVAILABLE:
        raise ImportError(
            "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
        )

    # Create the adapter
    adapter = GuardrailsValidatorAdapter(guardrails_validator)

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Use validator name or rule_id as name
    name = rule_id if rule_id else f"guardrails_{guardrails_validator.__class__.__name__}"

    # Create a rule using the adapter
    return FunctionRule(
        func=lambda text, **kw: adapter.validate(text, **kw),
        name=name,
        description=f"Rule using Guardrails validator: {guardrails_validator.__class__.__name__}",
        **rule_kwargs,
    )
