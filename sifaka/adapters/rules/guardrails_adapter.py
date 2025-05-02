"""
Adapter for using Guardrails validators as rules.

This module provides adapters for using validators from the Guardrails library as Sifaka rules.

Usage Example:
    from guardrails.hub import RegexMatch
    from sifaka.adapters.rules import create_guardrails_rule

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


from sifaka.rules.base import BaseValidator, RuleResult, Rule, RuleConfig
from sifaka.adapters.rules.base import Adaptable, BaseAdapter


@runtime_checkable
class GuardrailsValidatable(Protocol):
    """
    Protocol for components that can be validated by Guardrails.

    This defines the minimum interface needed for a component
    to be used with the Guardrails adapter.

    Examples:
        ```python
        class CustomValidator(GuardrailsValidatable):
            def validate(self, value, metadata=None):
                # Custom validation logic
                if valid:
                    return PassResult(value)
                else:
                    return FailResult(value, "Validation failed")
        ```
    """

    def validate(self, value: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a value.

        Args:
            value: The value to validate
            metadata: Additional context for validation

        Returns:
            Validation result
        """
        ...


class GuardrailsValidatorAdapter(BaseValidator[str]):
    """
    Adapter that allows Guardrails validators to be used within Sifaka.

    This adapter converts a Guardrails validator into a format compatible
    with Sifaka's validation system.

    Lifecycle:
    1. Initialization: Set up with a Guardrails validator
    2. Validation: Call Guardrails validator and convert result format
    3. Result: Return standardized RuleResult

    Examples:
        ```python
        from guardrails.hub import RegexMatch
        from sifaka.adapters.rules import GuardrailsValidatorAdapter

        regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")
        adapter = GuardrailsValidatorAdapter(regex_validator)

        result = adapter.validate("123-456-7890")
        ```
    """

    def __init__(self, guardrails_validator: GuardrailsValidator):
        """
        Initialize with a Guardrails validator.

        Args:
            guardrails_validator: The Guardrails validator to adapt

        Raises:
            ImportError: If Guardrails is not installed
        """
        super().__init__()

        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )
        self._guardrails_validator = guardrails_validator

    @property
    def guardrails_validator(self) -> GuardrailsValidator:
        """Get the Guardrails validator being adapted."""
        return self._guardrails_validator

    def _convert_guardrails_result(self, gr_result: ValidationResult) -> RuleResult:
        """
        Convert a Guardrails validation result to a Sifaka RuleResult.

        Args:
            gr_result: The Guardrails validation result

        Returns:
            Converted RuleResult
        """
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

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate text using the Guardrails validator.

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

        try:
            # Call the Guardrails validator
            gr_result = self._guardrails_validator.validate(output, metadata=kwargs)

            # Convert Guardrails result to Sifaka RuleResult
            return self._convert_guardrails_result(gr_result)
        except Exception as e:
            # Handle any exceptions from the Guardrails validator
            return RuleResult(
                passed=False,
                message=f"Guardrails validation error: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "validator": self._guardrails_validator.__class__.__name__,
                }
            )

    @property
    def validation_type(self) -> type:
        """Get the type this validator can validate."""
        return str


class GuardrailsRule(Rule):
    """
    Rule that uses a Guardrails validator for validation.

    This rule provides a bridge between Guardrails validators
    and the Sifaka rule system.

    Lifecycle:
    1. Initialization: Set up with a Guardrails validator
    2. Validation: Delegate to GuardrailsValidatorAdapter
    3. Result: Return standardized validation result

    Examples:
        ```python
        from guardrails.hub import RegexMatch
        from sifaka.adapters.rules import GuardrailsRule

        regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")
        rule = GuardrailsRule(
            guardrails_validator=regex_validator,
            name="phone_rule",
            description="Validates phone number format"
        )

        result = rule.validate("123-456-7890")
        ```
    """

    def __init__(
        self,
        guardrails_validator: GuardrailsValidator,
        rule_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize with a Guardrails validator.

        Args:
            guardrails_validator: The Guardrails validator to use
            rule_id: Unique identifier for the rule
            name: Name of the rule
            description: Description of the rule
            **kwargs: Additional configuration for the rule

        Raises:
            ImportError: If Guardrails is not installed
        """
        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )

        # Set default name and description if not provided
        if name is None:
            name = f"guardrails_{guardrails_validator.__class__.__name__}"
        if description is None:
            description = f"Rule using Guardrails validator: {guardrails_validator.__class__.__name__}"

        # Create rule configuration
        rule_config = RuleConfig(
            params={
                "guardrails_validator_type": guardrails_validator.__class__.__name__,
            }
        )

        self._guardrails_validator = guardrails_validator
        self._rule_id = rule_id or name
        self._adapter = GuardrailsValidatorAdapter(guardrails_validator)

        super().__init__(
            name=name,
            description=description,
            config=rule_config,
            **kwargs
        )

    def _create_default_validator(self) -> BaseValidator[str]:
        """
        Create a default validator for this rule.

        Returns:
            The GuardrailsValidatorAdapter
        """
        return self._adapter

    @property
    def guardrails_validator(self) -> GuardrailsValidator:
        """Get the Guardrails validator being used."""
        return self._guardrails_validator

    @property
    def rule_id(self) -> str:
        """Get the rule ID."""
        return self._rule_id

    def validate(self, text: str, **kwargs) -> RuleResult:
        """
        Validate text using the Guardrails validator.

        Args:
            text: Text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        result = self._adapter.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._rule_id)


def create_guardrails_rule(
    guardrails_validator: GuardrailsValidator,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
) -> GuardrailsRule:
    """
    Create a Sifaka rule that uses a Guardrails validator.

    Args:
        guardrails_validator: The Guardrails validator to adapt
        rule_id: Identifier for the rule
        name: Name for the rule
        description: Description of the rule
        **kwargs: Additional configuration for the rule

    Returns:
        Configured Rule using the Guardrails validator

    Examples:
        ```python
        from guardrails.hub import RegexMatch
        from sifaka.adapters.rules import create_guardrails_rule

        regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")
        rule = create_guardrails_rule(
            guardrails_validator=regex_validator,
            rule_id="phone_format_rule"
        )
        ```
    """
    if not GUARDRAILS_AVAILABLE:
        raise ImportError(
            "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
        )

    return GuardrailsRule(
        guardrails_validator=guardrails_validator,
        rule_id=rule_id,
        name=name,
        description=description,
        **kwargs
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
