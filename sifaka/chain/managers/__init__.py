"""
Chain Managers Package

Management components for Sifaka's chain system.

## Overview
This package provides specialized managers for different aspects of chain
functionality, including prompt creation and validation management. These
managers handle the core operational aspects of the chain system.

## Components
1. **PromptManager**: Manages prompt operations
   - Prompt creation and modification
   - Feedback incorporation
   - History management
   - Context and example handling

2. **ValidationManager**: Manages validation operations
   - Rule management
   - Validation execution
   - Error message generation
   - Rule prioritization

## Usage Examples
```python
from sifaka.chain.managers import PromptManager, ValidationManager
from sifaka.rules import create_length_rule

# Create prompt manager
prompt_manager = PromptManager()
prompt = prompt_manager.create_prompt(
    "Write a story",
    feedback="Make it longer",
    context="Set in future"
)

# Create validation manager
validation_manager = ValidationManager(
    rules=[create_length_rule(min_length=10)],
    prioritize_by_cost=True,
    fail_fast=True
)

# Validate output
result = validation_manager.validate("Some output text")
if not result.all_passed:
    errors = validation_manager.get_error_messages(result)
    print("Validation failed:", errors)
```

## Error Handling
- ValueError: Raised for invalid inputs or operations
- ValidationError: Raised when validation fails
- TypeError: Raised for type validation failures

## Configuration
- PromptManager:
  - feedback: Optional feedback for prompts
  - history: Optional execution history
  - context: Optional contextual information
  - examples: Optional example outputs

- ValidationManager:
  - rules: List of validation rules
  - prioritize_by_cost: Whether to sort rules by cost
  - fail_fast: Whether to stop on first failure
"""

from .prompt import PromptManager
from .validation import ValidationManager

__all__ = [
    "PromptManager",
    "ValidationManager",
]
