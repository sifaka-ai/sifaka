from typing import Any, List
"""
Chain Module

## Overview
This module provides a simplified and maintainable implementation of the chain system
for orchestrating the validation and improvement flow between models, rules, and critics.
It serves as the central coordinator for Sifaka's validation and refinement capabilities,
implementing a flexible and extensible chain system with a clean architecture.

## Components
1. **Chain**: Main user-facing class for running chains
2. **Engine**: Core execution engine that coordinates the flow
3. **Model**: Interface for text generation models
4. **Validator**: Interface for output validators
5. **Improver**: Interface for output improvers
6. **Formatter**: Interface for result formatters
7. **Plugin**: Interface for plugins

## Usage Examples
```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
validators = [create_length_rule(min_chars=10, max_chars=1000)]
improver = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain
chain = Chain(
    model=model,
    validators=validators,
    improver=improver,
    max_attempts=3
)

# Run chain
result = (chain and chain.run("Write a short story")
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")
```

## Error Handling
- ChainError: Raised when chain execution fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improver refinement fails
- ModelError: Raised when model generation fails

## Configuration
- max_attempts: Maximum number of retry attempts
- cache_enabled: Whether to enable result caching
- trace_enabled: Whether to enable execution tracing
"""
from .chain import Chain
from .engine import Engine
from ..core.results import ChainResult
from sifaka.interfaces.chain.components import Model, Validator, Improver
from sifaka.interfaces.chain.components.formatter import ChainFormatter as Formatter
from sifaka.interfaces.chain.plugin import ChainPlugin as Plugin
from ..utils.state import StateManager, create_chain_state
__all__: List[Any] = ['Chain', 'Engine', 'ChainResult', 'Model',
    'Validator', 'Improver', 'Formatter', 'Plugin', 'StateManager',
    'create_chain_state']
