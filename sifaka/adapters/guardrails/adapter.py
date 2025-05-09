"""
Guardrails Adapter

Adapter for using Guardrails validators as rules in the Sifaka framework.

## Overview
This module provides adapters for using validators from the Guardrails library as Sifaka rules.
It enables seamless integration between Guardrails' validation capabilities and Sifaka's
rule system, allowing for sophisticated content validation.

## Components
1. **GuardrailsValidatable Protocol**: Defines the expected interface for Guardrails validators
2. **GuardrailsValidatorAdapter**: Adapts Guardrails validators to work as Sifaka validators
3. **GuardrailsRule**: Rule that uses a Guardrails validator for validation
4. **Factory Functions**: Simple creation patterns for Guardrails-based rules

## Usage Examples
```python
from guardrails.hub import RegexMatch
from sifaka.adapters.guardrails import create_guardrails_rule

# Create a Guardrails validator
regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

# Create a Sifaka rule using the Guardrails validator
phone_rule = create_guardrails_rule(
    guardrails_validator=regex_validator,
    rule_id="phone_number_format",
    name="Phone Number Format",
    description="Validates that text contains a properly formatted phone number"
)

# Use it in a Sifaka chain
result = chain.run("What's a good phone number format?")
```

## Error Handling
- ImportError: Raised when Guardrails is not installed
- ValidationError: Raised when validation fails
- TypeError: Raised when input types are incompatible

## Configuration
- guardrails_validator: The Guardrails validator to use
- rule_id: Unique identifier for the rule
- name: Human-readable name of the rule
- description: Description of the rule's purpose
"""

from typing import Any, Dict, Optional, Union, Protocol, runtime_checkable

from pydantic import PrivateAttr

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
from sifaka.adapters.base import Adaptable, BaseAdapter
from sifaka.utils.state import AdapterState, StateManager, create_adapter_state, create_rule_state


@runtime_checkable
class GuardrailsValidatable(Protocol):
    """
    Protocol for components that can be validated by Guardrails.

    ## Overview
    This defines the minimum interface needed for a component
    to be used with the Guardrails adapter.

    ## Architecture
    The protocol defines a minimal interface that validators must implement
    to be compatible with Sifaka's adapter system.

    ## Lifecycle
    1. Initialization: Set up with validation configuration
    2. Validation: Process input value and apply validation logic
    3. Result: Return standardized validation result

    ## Error Handling
    - ValueError: Raised when input value is invalid
    - RuntimeError: Raised when validation fails
    - TypeError: Raised when input types are incompatible

    ## Examples
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
            value (str): The value to validate
            metadata (Optional[Dict[str, Any]]): Additional context for validation

        Returns:
            ValidationResult: Validation result with pass/fail status and metadata

        Raises:
            ValueError: If input value is invalid
            RuntimeError: If validation fails
        """
        ...


class GuardrailsValidatorAdapter(BaseValidator[str]):
    """
    Adapter for using Guardrails validators as Sifaka validators.

    ## Overview
    This adapter converts Guardrails validators into a format compatible
    with Sifaka's validation system, handling the conversion between
    Guardrails' validation results and Sifaka's RuleResult format.

    ## Architecture
    The adapter follows a standard pattern:
    1. Wraps a Guardrails validator instance
    2. Provides a validate() method conforming to BaseValidator interface
    3. Converts Guardrails ValidationResult to Sifaka RuleResult
    4. Handles empty text according to Sifaka conventions

    ## Lifecycle
    1. Initialization: Set up with a Guardrails validator
    2. Validation: Call Guardrails validator and convert result format
    3. Result: Return standardized RuleResult

    ## Error Handling
    - ImportError: Raised when Guardrails is not installed
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible

    ## Examples
    ```python
    from guardrails.hub import RegexMatch
    from sifaka.adapters.guardrails import GuardrailsValidatorAdapter

    # Create a Guardrails validator for phone numbers
    regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

    # Create the adapter
    adapter = GuardrailsValidatorAdapter(regex_validator)

    # Validate text
    result = adapter.validate("123-456-7890")
    print(f"Validation {'passed' if result.passed else 'failed'}")
    ```

    Attributes:
        guardrails_validator (GuardrailsValidator): The Guardrails validator being adapted
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
            gr_result = self.guardrails_validator.validate(output, metadata=kwargs)

            # Convert Guardrails result to Sifaka RuleResult
            return self._convert_guardrails_result(gr_result)

        except Exception as e:
            # Handle any exceptions from the Guardrails validator
            return RuleResult(
                passed=False,
                message=f"Guardrails validation error: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "validator": self.guardrails_validator.__class__.__name__,
                    "errors": [str(e)],
                },
            )

    @property
    def validation_type(self) -> type:
        """Get the type this validator can validate."""
        return str


class GuardrailsRule(Rule):
    """
    Rule that uses a Guardrails validator for validation.

    ## Overview
    This rule adapts a Guardrails validator to function as a Sifaka rule,
    allowing for sophisticated content validation using Guardrails' capabilities.

    ## Architecture
    The rule follows a standard pattern:
    1. Wraps a Guardrails validator
    2. Provides rule-specific configuration
    3. Delegates validation to GuardrailsValidatorAdapter
    4. Adds rule metadata to validation results

    ## Lifecycle
    1. Initialization: Set up with validator and configuration
    2. Validation: Delegate to GuardrailsValidatorAdapter
    3. Result Processing: Add rule-specific metadata

    ## Error Handling
    - ImportError: Raised when Guardrails is not installed
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible

    ## Examples
    ```python
    from guardrails.hub import RegexMatch
    from sifaka.adapters.guardrails import create_guardrails_rule

    # Create a rule
    validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")
    rule = create_guardrails_rule(
        guardrails_validator=validator,
        name="phone_format",
        description="Validates phone number format"
    )

    # Use the rule
    result = rule.validate("123-456-7890")
    ```

    Attributes:
        guardrails_validator (GuardrailsValidator): The Guardrails validator to use
        rule_id (str): Unique identifier for the rule
        name (str): Human-readable name of the rule
        description (str): Description of the rule's purpose
    """

    def __init__(
        self,
        guardrails_validator: GuardrailsValidator,
        rule_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the rule with a Guardrails validator.

        This method sets up the rule with a Guardrails validator and configuration options.
        It generates default values for name, description, and rule_id if not provided.

        Args:
            guardrails_validator: The Guardrails validator to use for validation
            rule_id: Optional unique identifier for the rule (default: generated from validator name)
            name: Optional display name for the rule (default: generated from validator name)
            description: Optional description of what the rule validates (default: generated from validator name)
            **kwargs: Additional configuration options passed to the base Rule class

        Raises:
            ImportError: If Guardrails is not installed

        Example:
            ```python
            from guardrails.hub import RegexMatch
            from sifaka.adapters.guardrails import GuardrailsRule

            # Create a rule with minimal configuration
            rule = GuardrailsRule(
                guardrails_validator=RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")
            )

            # Create a rule with custom configuration
            custom_rule = GuardrailsRule(
                guardrails_validator=RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}"),
                rule_id="phone_format",
                name="Phone Number Format",
                description="Validates that text contains a properly formatted phone number",
                severity="warning"
            )
            ```
        """
        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )

        # Generate default name and description if not provided
        validator_name = guardrails_validator.__class__.__name__
        rule_name = name or f"Guardrails {validator_name}"
        rule_description = description or f"Validates text using {validator_name}"

        # Store the Guardrails validator
        self._guardrails_validator = guardrails_validator
        self._rule_id = rule_id or f"guardrails_{validator_name.lower()}"

        # Initialize base rule
        super().__init__(
            name=rule_name,
            description=rule_description,
            **kwargs,
        )

        # Create the validator adapter
        self._validator = self._create_default_validator()

    def _create_default_validator(self) -> BaseValidator[str]:
        """Create the default validator adapter."""
        return GuardrailsValidatorAdapter(self._guardrails_validator)

    @property
    def guardrails_validator(self) -> GuardrailsValidator:
        """
        Get the underlying Guardrails validator.

        Returns:
            The Guardrails validator instance that this rule uses for validation

        Example:
            ```python
            rule = GuardrailsRule(guardrails_validator=regex_validator)
            validator = rule.guardrails_validator
            print(f"Using validator: {validator.__class__.__name__}")
            ```
        """
        return self._guardrails_validator

    @property
    def rule_id(self) -> str:
        """
        Get the rule ID.

        Returns:
            The unique identifier for this rule

        Example:
            ```python
            rule = GuardrailsRule(
                guardrails_validator=regex_validator,
                rule_id="phone_format"
            )
            print(f"Rule ID: {rule.rule_id}")
            ```
        """
        return self._rule_id

    def validate(self, text: str, **kwargs) -> RuleResult:
        """
        Validate text using the Guardrails validator.

        This method delegates validation to the GuardrailsValidatorAdapter
        and adds rule-specific metadata to the result.

        Args:
            text: The text to validate
            **kwargs: Additional validation context passed to the Guardrails validator

        Returns:
            RuleResult indicating if validation passed, with metadata including:
                - rule_id: The ID of this rule
                - guardrails_metadata: Original metadata from the Guardrails validator
                - errors: List of error messages if validation failed

        Example:
            ```python
            from guardrails.hub import RegexMatch
            from sifaka.adapters.guardrails import GuardrailsRule

            # Create a rule with a Guardrails validator
            rule = GuardrailsRule(
                guardrails_validator=RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}"),
                rule_id="phone_format"
            )

            # Validate text
            result = rule.validate("123-456-7890")
            print(f"Rule ID: {result.metadata['rule_id']}")
            print(f"Passed: {result.passed}")
            ```
        """
        result = self._validator.validate(text, **kwargs)
        return result.with_metadata(rule_id=self.rule_id)


def create_guardrails_rule(
    guardrails_validator: GuardrailsValidator,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> GuardrailsRule:
    """
    Factory function to create a Guardrails rule.

    ## Overview
    This function simplifies the creation of Guardrails rules by providing a
    consistent interface.

    Args:
        guardrails_validator (GuardrailsValidator): The Guardrails validator to use
        rule_id (Optional[str]): Unique identifier for the rule
        name (Optional[str]): Human-readable name of the rule
        description (Optional[str]): Description of the rule's purpose
        **kwargs: Additional keyword arguments

    Returns:
        GuardrailsRule: A configured Guardrails rule

    Raises:
        ImportError: If Guardrails is not installed
        ValueError: If validator is invalid
        TypeError: If input types are incompatible

    ## Examples
    ```python
    from guardrails.hub import RegexMatch
    from sifaka.adapters.guardrails import create_guardrails_rule

    # Create a rule
    validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")
    rule = create_guardrails_rule(
        guardrails_validator=validator,
        name="phone_format",
        description="Validates phone number format"
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
        **kwargs,
    )
