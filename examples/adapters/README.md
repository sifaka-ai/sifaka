# Sifaka Adapters Examples

This directory contains examples demonstrating how to use various adapters in Sifaka.

## Available Examples

### 1. Standalone Adapter Example

**File:** `standalone_adapter_example.py`

A simplified example demonstrating the adapter pattern by:
- Converting classifiers to rules using an adapter
- Composing multiple rules with logical operations

This example doesn't require any external dependencies and is a good starting point for understanding the adapter pattern.

### 2. GuardRails and Pydantic Example

**File:** `guardrails_pydantic_example.py`

Demonstrates how to use both GuardRails and Pydantic adapters together:
- Creates a GuardRails validator for email validation
- Creates a Pydantic model for structured data validation
- Shows how both adapters can be used together for comprehensive validation

**Requirements:**
- guardrails-ai (`pip install guardrails-ai`)
- pydantic (`pip install pydantic`)
- pydantic-ai (`pip install pydantic-ai`)

### 3. Advanced GuardRails and Pydantic Example

**File:** `advanced_guardrails_pydantic_example.py`

A more realistic example that:
- Uses a language model to generate content
- Validates the content with GuardRails rules
- Extracts structured data with Pydantic models
- Refines the content if validation fails

**Requirements:**
- guardrails-ai (`pip install guardrails-ai`)
- pydantic (`pip install pydantic`)
- pydantic-ai (`pip install pydantic-ai`)
- An OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Running the Examples

To run any example:

```bash
python examples/adapters/example_file_name.py
```

For example:

```bash
python examples/adapters/standalone_adapter_example.py
```

## Key Concepts

### Adapter Pattern

The adapter pattern allows components with incompatible interfaces to work together. In Sifaka, adapters are used to:

1. Convert classifiers to rules
2. Integrate external validation libraries like GuardRails
3. Connect with agent frameworks like PydanticAI

### GuardRails Adapters

GuardRails adapters allow you to use GuardRails validators as Sifaka rules. This enables:
- Leveraging GuardRails' extensive validator library
- Combining GuardRails validators with Sifaka's rule system
- Using GuardRails validators in Sifaka chains

### Pydantic Adapters

Pydantic adapters enable integration between Sifaka and PydanticAI agents. This allows:
- Validating structured data with Pydantic models
- Using Sifaka rules to validate PydanticAI agent outputs
- Refining PydanticAI agent outputs with Sifaka critics
