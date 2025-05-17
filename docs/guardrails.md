# GuardrailsAI Integration in Sifaka

Sifaka provides integration with [GuardrailsAI](https://www.guardrailsai.com/docs), a framework for validating and structuring data from language models. This integration allows you to use GuardrailsAI validators within the Sifaka validation chain system.

## Overview

The GuardrailsAI integration in Sifaka consists of:

1. **GuardrailsValidator**: A validator that adapts GuardrailsAI's Guard to the Sifaka validator interface
2. **Factory Functions**: Functions for creating GuardrailsValidator instances
3. **Integration with Sifaka Chains**: Ability to use GuardrailsAI validators in Sifaka validation chains

## Installation

To use GuardrailsAI with Sifaka, you need to install the `guardrails-ai` package:

```bash
pip install guardrails-ai
```

### API Key Configuration

GuardrailsAI requires an API key for authentication. You can obtain one from [Guardrails Hub](https://hub.guardrailsai.com/keys).

There are two ways to provide the API key:

1. **Environment Variable**: Set the `GUARDRAILS_API_KEY` environment variable:
   ```bash
   export GUARDRAILS_API_KEY="your-api-key-here"
   ```

2. **Direct Parameter**: Pass the API key directly to the validator:
   ```python
   validator = guardrails_validator(
       validators=["toxic_language"],
       api_key="your-api-key-here"
   )
   ```

You can also configure the GuardrailsAI CLI, which will store the API key in a configuration file:

```bash
guardrails configure
```

The configuration process will ask you three questions:
1. Whether you want to enable metrics reporting
2. Whether you want to use hosted remote inference endpoints
3. To enter your API key

## Installing GuardrailsAI Validators

Before using GuardrailsAI validators, you need to install them from the [Guardrails Hub](https://hub.guardrailsai.com):

```bash
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/detect_pii
# Install other validators as needed
```

## Using GuardrailsValidator

### Approach 1: Using Pre-defined Validators

You can create a GuardrailsValidator by specifying the validators you want to use:

```python
import os
from sifaka.validators import guardrails_validator

# Get API key from environment variable
guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY")

# Create a GuardrailsAI validator for toxic language detection
toxic_validator = guardrails_validator(
    validators=["toxic_language"],
    validator_args={
        "toxic_language": {
            "threshold": 0.5,
            "validation_method": "sentence"
        }
    },
    api_key=guardrails_api_key,  # Pass the API key to the validator
    name="Toxic Language Validator"
)

# Validate text
result = toxic_validator.validate("This is a test message.")
print(f"Passed: {result.passed}, Message: {result.message}")
```

### Approach 2: Using a Custom GuardrailsAI Guard

You can also create a GuardrailsValidator with a custom GuardrailsAI Guard:

```python
import os
import guardrails
from sifaka.validators import guardrails_validator

# Get API key from environment variable
guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY")

# Set API key in environment if provided
if guardrails_api_key:
    os.environ["GUARDRAILS_API_KEY"] = guardrails_api_key

# Create a custom GuardrailsAI Guard
guard = guardrails.Guard().use_many(
    guardrails.hub.ToxicLanguage(threshold=0.7, validation_method="sentence"),
    guardrails.hub.DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"])
)

# Create a GuardrailsAI validator with the custom guard
custom_validator = guardrails_validator(
    guard=guard,
    api_key=guardrails_api_key,  # Pass the API key to the validator
    name="Custom GuardrailsAI Validator"
)

# Validate text
result = custom_validator.validate("This is a test message.")
print(f"Passed: {result.passed}, Message: {result.message}")
```

## Using GuardrailsAI Validators in Chains

GuardrailsAI validators can be used in Sifaka chains just like any other validator:

```python
import os
from sifaka.validators import guardrails_validator
from sifaka.chain import Chain
from sifaka.factories import create_model

# Get API key from environment variable
guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY")

# Create a GuardrailsAI validator
pii_validator = guardrails_validator(
    validators=["detect_pii"],
    validator_args={
        "detect_pii": {
            "pii_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"]
        }
    },
    api_key=guardrails_api_key,  # Pass the API key to the validator
    name="PII Detector"
)

# Create a model
model = create_model("openai:gpt-3.5-turbo")

# Create a chain with GuardrailsAI validators
chain = (
    Chain()
    .with_model(model)
    .with_prompt("Write a short paragraph about privacy and data security.")
    .validate_with(pii_validator)
)

# Run the chain
result = chain.run()

print(f"Chain result passed all validations: {result.passed}")
print(f"Generated text: {result.text}")
```

## Available GuardrailsAI Validators

GuardrailsAI provides a wide range of validators through the [Guardrails Hub](https://hub.guardrailsai.com). Some popular validators include:

- **ToxicLanguage**: Detects toxic content in text
- **DetectPII**: Identifies personally identifiable information
- **CompetitorCheck**: Checks for mentions of competitors
- **RegexMatch**: Validates text against regular expressions
- **ValidLength**: Ensures text is within a specified length range
- **ProvenanceLLM**: Verifies that LLM output is grounded in provided sources

Visit the [Guardrails Hub](https://hub.guardrailsai.com) for a complete list of available validators and their documentation.

## Passing Metadata to GuardrailsAI Validators

Some GuardrailsAI validators require additional metadata. You can pass this metadata when calling `validate`:

```python
import os
from sifaka.validators import guardrails_validator

# Get API key from environment variable
guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY")

# Create a GuardrailsAI validator
provenance_validator = guardrails_validator(
    validators=["provenance_llm"],
    validator_args={
        "provenance_llm": {
            "validation_method": "sentence"
        }
    },
    api_key=guardrails_api_key,  # Pass the API key to the validator
    name="Provenance Validator"
)

# Define sources and embedding function
sources = [
    "The sun is a star.",
    "The sun rises in the east and sets in the west."
]

# Create metadata
metadata = {
    'sources': sources,
    'embed_function': lambda x: model.encode(x)  # Using a sentence transformer model
}

# Validate text with metadata
result = provenance_validator.validate(
    "The sun is a star that rises in the east and sets in the west.",
    metadata=metadata
)
```

## Comparison with Sifaka Validators

GuardrailsAI validators are more similar to Sifaka validators than classifiers. Both are focused on validating whether text meets specific criteria and providing pass/fail results.

The key differences are:

1. **Ecosystem**: GuardrailsAI has a large ecosystem of pre-built validators available through Guardrails Hub
2. **Implementation**: GuardrailsAI validators often use more sophisticated techniques and models
3. **Configuration**: GuardrailsAI validators have a different configuration approach with their own CLI

## Best Practices

1. **Install Required Validators**: Make sure to install the required validators from Guardrails Hub before using them
2. **Configure API Key**: Set up your GuardrailsAI API key using the `guardrails configure` command
3. **Check Documentation**: Refer to the [GuardrailsAI documentation](https://www.guardrailsai.com/docs) for details on specific validators
4. **Handle Errors**: Properly handle potential errors when using GuardrailsAI validators, especially when they require external services
5. **Combine Validators**: Consider combining multiple validators for comprehensive validation
