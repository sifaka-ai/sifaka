"""
Chain Factories Module

This module provides factory functions for creating chains and components.
These factories simplify the creation of chains with sensible defaults.

## Factory Functions
1. **create_chain**: Creates a chain with the specified components
2. **create_model_adapter**: Creates a model adapter for existing model providers
3. **create_validator_adapter**: Creates a validator adapter for existing rules
4. **create_improver_adapter**: Creates an improver adapter for existing critics

## Usage Examples
```python
from sifaka.chain import create_chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
validators = [create_length_rule(min_chars=10, max_chars=1000)]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain using factory
chain = create_chain(
    model=model,
    validators=validators,
    improver=critic,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story")
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")
```
"""

from typing import Any, Dict, List, Optional

from .chain import Chain
from .interfaces import Model, Validator, Improver, Formatter
from .config import ChainConfig
from .adapters import ModelAdapter, ValidatorAdapter, ImproverAdapter, FormatterAdapter


def create_chain(
    model: Any,
    validators: List[Any] = None,
    improver: Optional[Any] = None,
    formatter: Optional[Any] = None,
    max_attempts: int = 3,
    config: Optional[ChainConfig] = None,
    name: str = "chain",
    description: str = "Sifaka chain for text generation and validation",
) -> Chain:
    """
    Create a chain with the specified components.

    This factory function creates a chain with the specified components,
    automatically adapting them to the required interfaces if needed.

    Args:
        model: The model to use for generation
        validators: The validators to use for validation
        improver: Optional improver for output improvement
        formatter: Optional formatter for result formatting
        max_attempts: Maximum number of generation attempts
        config: Chain configuration
        name: Chain name
        description: Chain description

    Returns:
        A chain instance
    """
    # Adapt model if needed
    adapted_model = model if isinstance(model, Model) else ModelAdapter(model)

    # Adapt validators if needed
    adapted_validators = []
    if validators:
        for validator in validators:
            if isinstance(validator, Validator):
                adapted_validators.append(validator)
            else:
                adapted_validators.append(ValidatorAdapter(validator))

    # Adapt improver if needed
    adapted_improver = None
    if improver:
        if isinstance(improver, Improver):
            adapted_improver = improver
        else:
            adapted_improver = ImproverAdapter(improver)

    # Adapt formatter if needed
    adapted_formatter = None
    if formatter:
        if isinstance(formatter, Formatter):
            adapted_formatter = formatter
        else:
            adapted_formatter = FormatterAdapter(formatter)

    # Create chain
    return Chain(
        model=adapted_model,
        validators=adapted_validators,
        improver=adapted_improver,
        formatter=adapted_formatter,
        max_attempts=max_attempts,
        config=config,
        name=name,
        description=description,
    )
