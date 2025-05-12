from typing import Any, List
"""
Adapter for using Guardrails validators as rules.

This module provides adapters for using validators from the Guardrails library as Sifaka rules.
It enables seamless integration between Guardrails' validation capabilities and Sifaka's
rule system, allowing for sophisticated content validation.

## Usage Examples

### Using the Adapter
```python
from guardrails.hub import RegexMatch
from sifaka.adapters.guardrails import create_guardrails_adapter

# Create a Guardrails validator
regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

# Create a Sifaka adapter using the Guardrails validator
phone_adapter = create_guardrails_adapter(
    guardrails_validator=regex_validator,
    name="Phone Number Format",
    description="Validates that text contains a properly formatted phone number"
)

# Use the adapter for validation
result = (phone_adapter and phone_adapter.validate("My phone number is 555-123-4567")
```

### Using the Rule (Legacy)
```python
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
result = (chain and chain.run("What's a good phone number format?")
```
"""
from sifaka.adapters.guardrails.adapter import GuardrailsValidatable, GuardrailsAdapter, GuardrailsValidatorAdapter, GuardrailsRule, create_guardrails_adapter, create_guardrails_rule
__all__: List[Any] = ['GuardrailsValidatable', 'GuardrailsAdapter',
    'GuardrailsValidatorAdapter', 'GuardrailsRule',
    'create_guardrails_adapter', 'create_guardrails_rule']
