"""
PydanticAI adapter for Sifaka.

This module provides adapters for integrating Sifaka's validation and refinement
capabilities with PydanticAI agents. It enables PydanticAI agents to benefit from
Sifaka's rule-based validation and critic-based refinement to improve the semantic
quality of outputs beyond just structural validation.

## Architecture Overview

The PydanticAI adapter follows an adapter pattern to enable Sifaka to work with
PydanticAI agents:

1. **Adapter Components**: Implement wrappers that translate between systems
2. **Factory Functions**: Provide simple creation patterns for common use cases
3. **Integration Points**: Connect with PydanticAI's output validation system

## Usage Example

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sifaka.adapters.pydantic_ai import create_pydantic_adapter
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule

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
    create_prohibited_content_rule(terms=["harmful", "offensive"])
]

# Create a Sifaka adapter for PydanticAI
sifaka_adapter = create_pydantic_adapter(rules=rules, max_refine=2)

# Register the adapter as an output validator
@agent.output_validator
def validate_with_sifaka(ctx: RunContext, output: OrderSummary) -> OrderSummary:
    return sifaka_adapter(ctx, output)

# Run the agent
result = agent.run_sync("Create an order summary for customer John Doe")
print(result.output)
```

For more advanced usage, see the examples directory.
"""

from sifaka.adapters.pydantic_ai.adapter import (
    SifakaPydanticAdapter,
    SifakaPydanticConfig,
)
from sifaka.adapters.pydantic_ai.factory import (
    create_pydantic_adapter,
    create_pydantic_adapter_with_critic,
)

__all__ = [
    # Core components
    "SifakaPydanticAdapter",
    "SifakaPydanticConfig",
    # Factory functions
    "create_pydantic_adapter",
    "create_pydantic_adapter_with_critic",
]
