# Guardrails Integration

Sifaka provides seamless integration with [Guardrails AI](https://www.guardrailsai.com/), allowing you to leverage Guardrails' validation capabilities within the Sifaka framework.

## Installation

To use the Guardrails integration, you need to install both Sifaka and Guardrails:

```bash
pip install sifaka
pip install guardrails-ai
```

## Basic Usage

```python
from sifaka.adapters.rules.guardrails import create_guardrails_rule
from sifaka.chain import create_simple_chain
from sifaka.models import create_openai_provider

# Create a guardrails rule using a rail specification
guardrails_rule = create_guardrails_rule(
    name="guardrails_validator",
    description="Validates text using Guardrails",
    rail_spec="""
    <rail version="0.1">
    <output>
        <string name="text" format="no-profanity" />
    </output>
    </rail>
    """
)

# Create a chain with the guardrails rule
model = create_openai_provider("gpt-4")
chain = create_simple_chain(
    model=model,
    rules=[guardrails_rule],
    max_attempts=3
)

# Run the chain
result = chain.run("Write a short story")
```

## Using Guardrails Hub Validators

Guardrails provides a collection of pre-built validators through the Guardrails Hub. You can use these validators with Sifaka:

```python
from guardrails.hub import RegexMatch, ValidChoices
from sifaka.adapters.rules.guardrails import create_guardrails_rule
from sifaka.chain import create_simple_chain
from sifaka.models import create_openai_provider

# Create a rule with a RegexMatch validator
phone_rule = create_guardrails_rule(
    guardrails_validator=RegexMatch(regex=r"\d{3}-\d{3}-\d{4}"),
    name="phone_format_rule",
    description="Validates phone number format (XXX-XXX-XXXX)"
)

# Create a rule with a ValidChoices validator
color_rule = create_guardrails_rule(
    guardrails_validator=ValidChoices(choices=["red", "green", "blue", "yellow"]),
    name="color_choice_rule",
    description="Validates that the text contains a valid color"
)

# Create a chain with both rules
model = create_openai_provider("gpt-4")
chain = create_simple_chain(
    model=model,
    rules=[phone_rule, color_rule],
    max_attempts=3
)

# Run the chain
result = chain.run("Write a sentence with a phone number and a color")
```

## Advanced Usage

### Custom Guardrails Validators

You can create custom Guardrails validators and use them with Sifaka:

```python
from guardrails.validators import Validator
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from sifaka.adapters.rules.guardrails import create_guardrails_rule

# Create a custom Guardrails validator
class WordCountValidator(Validator):
    """Validates that the text has a specific word count range."""
    
    min_words: int
    max_words: int
    
    def validate(self, value: str, metadata: Dict[str, Any]) -> List[str]:
        """Validate the word count."""
        word_count = len(value.split())
        
        if word_count < self.min_words:
            return [f"Text has {word_count} words, which is less than the minimum of {self.min_words} words"]
        
        if word_count > self.max_words:
            return [f"Text has {word_count} words, which exceeds the maximum of {self.max_words} words"]
        
        return []

# Create a Sifaka rule with the custom validator
word_count_rule = create_guardrails_rule(
    guardrails_validator=WordCountValidator(min_words=10, max_words=50),
    name="word_count_rule",
    description="Validates that the text has between 10 and 50 words"
)
```

### Combining Multiple Validators

You can combine multiple Guardrails validators using a rail specification:

```python
from sifaka.adapters.rules.guardrails import create_guardrails_rule

# Create a rule with multiple validators
multi_validator_rule = create_guardrails_rule(
    name="multi_validator_rule",
    description="Validates text using multiple Guardrails validators",
    rail_spec="""
    <rail version="0.1">
    <output>
        <string name="text"
                format="no-profanity"
                on-fail-length="reask"
                min-length="50"
                max-length="500" />
    </output>
    </rail>
    """
)
```

### Error Handling

The Guardrails adapter provides detailed error information when validation fails:

```python
from guardrails.hub import RegexMatch
from sifaka.adapters.rules.guardrails import create_guardrails_rule

# Create a rule with a RegexMatch validator
email_rule = create_guardrails_rule(
    guardrails_validator=RegexMatch(regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    name="email_format_rule",
    description="Validates email format"
)

# Validate text
result = email_rule.validate("This is not an email")

# Check the result
if not result.passed:
    print(f"Validation failed: {result.message}")
    
    # Access detailed error information
    if "errors" in result.metadata:
        for error in result.metadata["errors"]:
            print(f"Error: {error}")
```

## State Management

The Guardrails adapter uses Sifaka's standardized state management approach:

```python
from guardrails.hub import RegexMatch
from sifaka.adapters.rules.guardrails import GuardrailsValidatorAdapter

# Create a Guardrails validator adapter
regex_validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")
adapter = GuardrailsValidatorAdapter(regex_validator)

# State is managed internally using StateManager
# The state includes:
# - The Guardrails validator (adaptee)
# - The validation cache
# - Initialization status

# Access the adaptee through the state manager
validator = adapter._state_manager.get_state().adaptee
```

## Performance Considerations

- Guardrails validators are generally lightweight and fast
- Consider using caching for repeated validations of the same text
- Complex rail specifications with multiple validators may have higher computational cost

## Compatibility Notes

- Sifaka's Guardrails integration is compatible with Guardrails AI version 0.6.6 and above
- The integration is designed to work with both Guardrails Hub validators and custom validators
- Rail specifications can be provided as strings or loaded from files
