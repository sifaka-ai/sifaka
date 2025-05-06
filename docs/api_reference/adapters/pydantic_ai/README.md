# PydanticAI Adapter

The PydanticAI adapter integrates Sifaka's validation and refinement capabilities with PydanticAI agents. It enables PydanticAI agents to benefit from Sifaka's rule-based validation and critic-based refinement to improve the semantic quality of outputs beyond just structural validation.

## Overview

While PydanticAI already enforces structured outputs via Pydantic models (ensuring type correctness and JSON structure), structural validation alone is not enough to guarantee semantic correctness or safety. An output might formally match the schema but still contain factual errors, policy violations, or missing details.

The Sifaka PydanticAI adapter adds a post-generation validation and refinement loop that catches semantic issues and fixes them before the agent's answer is returned.

## Core Components

### SifakaPydanticAdapter

`SifakaPydanticAdapter` is the main adapter class that bridges between PydanticAI and Sifaka.

```python
from sifaka.adapters.pydantic_ai import SifakaPydanticAdapter
from sifaka.rules.formatting.length import create_length_rule
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# Define a Pydantic model
class Response(BaseModel):
    content: str

# Create rules and adapter
rules = [create_length_rule(min_chars=10, max_chars=100)]
adapter = SifakaPydanticAdapter(
    rules=rules,
    output_model=Response,
    max_refine=2
)

# Use as a PydanticAI output validator
@agent.output_validator
def validate_with_sifaka(ctx: RunContext, output: Response) -> Response:
    return adapter(ctx, output)
```

### SifakaPydanticConfig

`SifakaPydanticConfig` provides configuration options for the adapter.

```python
from sifaka.adapters.pydantic_ai import SifakaPydanticConfig

config = SifakaPydanticConfig(
    max_refine=3,
    prioritize_by_cost=True,
    serialize_method="model_dump",  # For Pydantic v2
    deserialize_method="model_validate"  # For Pydantic v2
)
```

## Factory Functions

### create_pydantic_adapter

`create_pydantic_adapter` creates a basic adapter with the specified rules.

```python
from sifaka.adapters.pydantic_ai import create_pydantic_adapter
from sifaka.rules.formatting.length import create_length_rule
from pydantic import BaseModel

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

`create_pydantic_adapter_with_critic` creates an adapter with rules and a critic for refinement.

```python
from sifaka.adapters.pydantic_ai import create_pydantic_adapter_with_critic
from sifaka.rules.formatting.length import create_length_rule
from sifaka.models.openai import create_openai_provider
from pydantic import BaseModel

# Define a Pydantic model
class Response(BaseModel):
    content: str

# Create model provider
provider = create_openai_provider(model_name="gpt-4")

# Create rules and adapter
rules = [create_length_rule(min_chars=10, max_chars=100)]
adapter = create_pydantic_adapter_with_critic(
    rules=rules,
    output_model=Response,
    model_provider=provider,
    system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
    max_refine=2
)
```

## How It Works

The PydanticAI adapter works as follows:

1. **Initialization**: The adapter is initialized with Sifaka rules, an optional critic, and configuration options.

2. **Integration**: The adapter is registered as a PydanticAI output validator using the `@agent.output_validator` decorator.

3. **Validation**: When the PydanticAI agent generates an output, the adapter validates it against the Sifaka rules.

4. **Refinement**: If validation fails, the adapter raises a `ModelRetry` exception with the validation errors, which triggers PydanticAI to retry the generation with the error feedback.

5. **Result**: Once validation passes or the maximum number of refinement attempts is reached, the adapter returns the final output.

## Usage Examples

### Basic Usage

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sifaka.adapters.pydantic_ai import create_pydantic_adapter
from sifaka.rules.formatting.length import create_length_rule

# Define a Pydantic model for the agent output
class OrderSummary(BaseModel):
    order_id: int
    customer: str
    items: list[str]
    notes: str = Field(..., description="Summary notes for the order")

# Create a PydanticAI agent
agent = Agent("openai:gpt-4", output_type=OrderSummary)

# Create Sifaka rules
rules = [
    create_length_rule(min_chars=50, max_chars=500),
    # Add more rules as needed
]

# Create a Sifaka adapter for PydanticAI
sifaka_adapter = create_pydantic_adapter(
    rules=rules,
    output_model=OrderSummary,
    max_refine=2
)

# Register the adapter as an output validator
@agent.output_validator
def validate_with_sifaka(ctx: RunContext, output: OrderSummary) -> OrderSummary:
    return sifaka_adapter(ctx, output)

# Run the agent
result = agent.run_sync("Create an order summary for customer John Doe")
print(result.output)
```

### Advanced Usage with Critic

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sifaka.adapters.pydantic_ai import create_pydantic_adapter_with_critic
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule
from sifaka.models.openai import create_openai_provider

# Define a Pydantic model
class OrderSummary(BaseModel):
    order_id: int
    customer: str
    items: list[str]
    notes: str = Field(..., description="Summary notes for the order")

# Create a PydanticAI agent
agent = Agent("openai:gpt-4", output_type=OrderSummary)

# Create a model provider for the critic
model_provider = create_openai_provider(model_name="gpt-4")

# Create Sifaka rules
rules = [
    create_length_rule(
        min_chars=50,
        max_chars=500,
        field_path="notes"  # Only validate the notes field
    ),
    create_prohibited_content_rule(
        terms=["harmful", "offensive", "inappropriate"]
    )
]

# Create a Sifaka adapter with a critic
sifaka_adapter = create_pydantic_adapter_with_critic(
    rules=rules,
    output_model=OrderSummary,
    model_provider=model_provider,
    system_prompt=(
        "You are an expert editor that improves order summaries. "
        "Ensure the notes field is detailed and between 50-500 characters. "
        "Remove any prohibited terms and maintain a professional tone."
    ),
    max_refine=2
)

# Register the adapter as an output validator
@agent.output_validator
def validate_with_sifaka(ctx: RunContext, output: OrderSummary) -> OrderSummary:
    return sifaka_adapter(ctx, output)

# Run the agent
result = agent.run_sync("Create an order summary for customer John Doe")
print(result.output)
```

## Best Practices

1. **Choose Appropriate Rules**: Select Sifaka rules that complement PydanticAI's structural validation.

2. **Set Reasonable Refinement Limits**: Configure `max_refine` based on your use case to avoid excessive retries.

3. **Provide Clear System Prompts**: When using critics, provide clear system prompts that explain how to fix validation issues.

4. **Field-Specific Validation**: Use field paths in rules to validate specific fields in complex models.

5. **Error Handling**: Implement proper error handling for edge cases where validation or refinement might fail.

## Limitations and Considerations

1. **Performance Impact**: Each refinement attempt requires an additional model call, which can increase latency and costs.

2. **Serialization Challenges**: Complex nested Pydantic models might require custom serialization/deserialization logic.

3. **Critic Effectiveness**: The effectiveness of refinement depends on the quality of the critic and the clarity of validation error messages.

4. **Compatibility**: The adapter is designed for PydanticAI's current output validation system and may need updates if that system changes.
