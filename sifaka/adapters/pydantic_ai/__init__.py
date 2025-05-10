"""
PydanticAI Adapter

Adapter for integrating Sifaka's validation and refinement with PydanticAI agents.

## Overview
This module provides adapters for integrating Sifaka's validation and refinement
capabilities with PydanticAI agents. It enables PydanticAI agents to benefit from
Sifaka's rule-based validation and critic-based refinement to improve the semantic
quality of outputs beyond just structural validation.

## Components
1. **SifakaPydanticAdapter**: Core adapter class for PydanticAI integration
2. **SifakaPydanticConfig**: Configuration class for the adapter
3. **Factory Functions**: Simple creation patterns for common use cases

## Usage Examples
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

## Error Handling
- ImportError: Raised when PydanticAI is not installed
- ValueError: Raised when configuration is invalid
- ModelRetry: Raised when validation fails and refinement is needed
- AdapterError: Raised for adapter-specific errors

## State Management
The module uses a standardized state management approach:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state object
- Clear separation of configuration and state
- Execution tracking for monitoring and debugging

## Configuration
- max_refine: Maximum number of refinement attempts
- prioritize_by_cost: Whether to prioritize rules by cost
- serialize_method: Method to use for serializing Pydantic models
- deserialize_method: Method to use for deserializing Pydantic models
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
