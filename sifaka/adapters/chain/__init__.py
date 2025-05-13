from typing import Any, List
"""
Chain Adapters

This module provides adapter classes for integrating existing Sifaka components
with the chain system. These adapters implement the chain interfaces while
delegating to the existing components.

## Adapter Classes
1. **ModelAdapter**: Adapts existing model providers to the Model interface
2. **ValidatorAdapter**: Adapts existing rules to the Validator interface
3. **ImproverAdapter**: Adapts existing critics to the Improver interface
4. **FormatterAdapter**: Adapts existing formatters to the Formatter interface

## Usage Examples
```python
from sifaka.adapters.chain import ModelAdapter, ValidatorAdapter, ImproverAdapter
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model_provider = OpenAIProvider("gpt-3.5-turbo")
rule = create_length_rule(min_chars=10, max_chars=1000)
critic = create_prompt_critic(
    llm_provider=model_provider,
    system_prompt="You are an expert editor that improves text."
)

# Create adapters
model = ModelAdapter(model_provider)
validator = ValidatorAdapter(rule)
improver = ImproverAdapter(critic)

# Use adapters
output = model.generate("Write a short story") if model else ""
validation_result = validator.validate(output) if validator else ""
if not validation_result.passed:
    improved_output = improver.improve(output, [validation_result]) if improver else ""
```

## Error Handling
- ModelError: Raised when model generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improvement fails
- FormatterError: Raised when formatting fails

## State Management
All adapters use the standardized state management pattern:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state manager
- Clear separation of configuration and state
- Execution tracking for monitoring and debugging
"""
from .model import ModelAdapter
from .validator import ValidatorAdapter
from .improver import ImproverAdapter
from .formatter import FormatterAdapter
__all__: List[Any] = ['ModelAdapter', 'ValidatorAdapter', 'ImproverAdapter',
    'FormatterAdapter']
