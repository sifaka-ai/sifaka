# Adapters Component

## Overview

The Adapters component in Sifaka provides a standardized way to integrate external libraries and components with the Sifaka ecosystem. Adapters follow the Adapter design pattern, wrapping external components (adaptees) to make them compatible with Sifaka's interfaces and patterns.

## Architecture

The Adapters component follows a standardized architecture:

1. **Base Adapter**: The `BaseAdapter` class in `sifaka/adapters/base.py` provides the foundation for all adapters, implementing standardized state management and lifecycle methods.

2. **Specialized Adapters**: Specific adapter implementations extend the base adapter to support different types of external components:
   - `GuardrailsAdapter`: Adapts Guardrails validators to Sifaka's validation system
   - `ClassifierAdapter`: Adapts external classifiers to Sifaka's classification system
   - `SifakaPydanticAdapter`: Integrates Sifaka's validation with PydanticAI agents

3. **Factory Functions**: Each adapter type has factory functions that simplify creation and configuration:
   - `create_adapter`: Core factory function in `sifaka/core/factories.py`
   - `create_pydantic_adapter`: Specialized factory for PydanticAI adapters
   - `create_pydantic_adapter_with_critic`: Factory for PydanticAI adapters with critic support

## State Management

All adapters use the standardized state management pattern:

1. **State Manager**: Each adapter has a `_state_manager` attribute created with `create_adapter_state()` from `utils/state.py`.

2. **State Structure**: The adapter state includes:
   - `adaptee`: The external component being adapted
   - `initialized`: Whether the adapter has been initialized
   - `execution_count`: Number of executions
   - `error_count`: Number of errors encountered
   - `last_execution_time`: Timestamp of last execution
   - `avg_execution_time`: Average execution time
   - `cache`: Cache for results
   - `config_cache`: Cache for configuration

3. **Metadata**: Additional information stored in the state manager's metadata:
   - `component_type`: Type of component (e.g., "adapter")
   - `adapter_type`: Type of adapter (e.g., "guardrails", "classifier")
   - `creation_time`: When the adapter was created
   - `name`: Human-readable name for the adapter
   - `description`: Description of the adapter's purpose

## Lifecycle Management

Adapters follow a standardized lifecycle:

1. **Initialization**: During construction, the adapter initializes its state and sets up the adaptee.

2. **Warm-up**: The `warm_up()` method ensures the adapter is properly initialized before use.

3. **Execution**: The adapter's main functionality is executed through type-specific methods (e.g., `validate()`, `classify()`).

4. **Cleanup**: Resources are released when the adapter is no longer needed.

## Error Handling

Adapters use standardized error handling:

1. **Error Types**: Errors are categorized using standard error classes from `utils/errors.py`.

2. **Error Recording**: Errors are recorded in the state manager for tracking and analysis.

3. **Error Conversion**: External errors are converted to Sifaka-compatible errors with appropriate context.

## Available Adapters

### GuardrailsAdapter

Adapts Guardrails validators to Sifaka's validation system.

```python
from guardrails.validators import RegexMatch
from sifaka.adapters.guardrails import GuardrailsAdapter

# Create a Guardrails validator
validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")

# Create an adapter
adapter = GuardrailsAdapter(validator)

# Use the adapter
result = adapter.validate("123-456-7890")
```

### GuardrailsRule

Wraps a Guardrails validator as a Sifaka rule.

```python
from guardrails.validators import RegexMatch
from sifaka.adapters.guardrails import GuardrailsRule

# Create a Guardrails validator
validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")

# Create a rule
rule = GuardrailsRule(validator)

# Use the rule
result = rule.validate("123-456-7890")
```

### SifakaPydanticAdapter

Integrates Sifaka's validation with PydanticAI agents.

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from sifaka.adapters.pydantic_ai import create_pydantic_adapter
from sifaka.rules.formatting.length import create_length_rule

# Define a Pydantic model
class Response(BaseModel):
    content: str

# Create rules and adapter
rules = [create_length_rule(min_chars=10, max_chars=100)]
adapter = create_pydantic_adapter(
    rules=rules,
    output_model=Response,
    max_refine=2
)

# Use as a PydanticAI output validator
@agent.output_validator
def validate_with_sifaka(ctx: RunContext, output: Response) -> Response:
    return adapter(ctx, output)
```

## Factory Functions

### create_adapter

Core factory function for creating adapters of any type.

```python
from sifaka.core.factories import create_adapter
from guardrails.validators import RegexMatch

# Create a Guardrails validator
validator = RegexMatch(regex=r"\d{3}-\d{3}-\d{4}")

# Create an adapter
adapter = create_adapter(
    adapter_type="guardrails",
    adaptee=validator,
    name="phone_validator",
    description="Validates phone numbers"
)
```

### create_pydantic_adapter

Factory function for creating PydanticAI adapters.

```python
from pydantic import BaseModel
from sifaka.adapters.pydantic_ai import create_pydantic_adapter
from sifaka.rules.formatting.length import create_length_rule

# Define a Pydantic model
class Response(BaseModel):
    content: str

# Create rules and adapter
rules = [create_length_rule(min_chars=10, max_chars=100)]
adapter = create_pydantic_adapter(
    rules=rules,
    output_model=Response,
    max_refine=2
)
```

### create_pydantic_adapter_with_critic

Factory function for creating PydanticAI adapters with critic support.

```python
from pydantic import BaseModel
from sifaka.adapters.pydantic_ai import create_pydantic_adapter_with_critic
from sifaka.rules.formatting.length import create_length_rule
from sifaka.models.factories import create_openai_provider

# Define a Pydantic model
class Response(BaseModel):
    content: str

# Create a model provider
provider = create_openai_provider(model_name="gpt-4")

# Create rules and adapter
rules = [create_length_rule(min_chars=10, max_chars=100)]
adapter = create_pydantic_adapter_with_critic(
    rules=rules,
    output_model=Response,
    model_provider=provider,
    max_refine=2
)
```
