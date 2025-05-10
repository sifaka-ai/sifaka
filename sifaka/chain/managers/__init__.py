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
   - State management
   - Performance tracking
   - Caching

2. **ValidationManager**: Manages validation operations
   - Rule management
   - Validation execution
   - Error message generation
   - Rule prioritization
   - State management
   - Performance tracking
   - Caching

3. **MemoryManager**: Manages memory operations
   - Memory storage and retrieval
   - Memory store management
   - Caching of memory operations
   - State management
   - Performance tracking
   - Error handling

## Usage Examples
```python
from sifaka.chain.managers import (
    BasePrompt,
    create_prompt_manager,
    create_validation_manager,
    create_memory_manager,
    BaseMemory
)
from sifaka.rules import create_length_rule

# Create prompt manager using factory function
prompt_manager = create_prompt_manager(
    name="story_prompt_manager",
    description="Generates story prompts",
    template_format="text",
    add_timestamps=True,
    max_history_items=5,
    max_examples=3,
    cache_size=100
)

# Create a prompt template
story_prompt = BasePrompt(
    name="story_prompt",
    description="Generates story prompts",
    template="Write a story about {topic}"
)

# Add prompt to manager
prompt_manager.add_prompt(story_prompt)

# Generate prompt
context = {"topic": "space exploration"}
results = prompt_manager.generate(context)
prompt = results[0].prompt

# Create validation manager using factory function
validation_manager = create_validation_manager(
    rules=[create_length_rule(min_length=10)],
    name="content_validator",
    description="Validates content length",
    prioritize_by_cost=True,
    fail_fast=True,
    cache_size=100
)

# Validate output
result = validation_manager.validate("Some output text")
if not result.passed:
    errors = validation_manager.get_error_messages(result)
    print("Validation failed:", errors)

# Get validation statistics
stats = validation_manager.get_statistics()
print(f"Validation count: {stats['validation_count']}")
print(f"Success rate: {stats['success_rate']:.2f}")

# Create memory manager using factory function
memory_manager = create_memory_manager(
    name="story_memory",
    description="Stores story generation history",
    cache_enabled=True,
    max_items=50
)

# Store a value in memory
memory_manager.store("last_prompt", prompt)

# Retrieve a value from memory
memory_input = {"operation": "retrieve", "key": "last_prompt"}
memory_result = memory_manager.process(memory_input)
if memory_result.found:
    print(f"Retrieved value: {memory_result.value}")

# Get memory statistics
memory_stats = memory_manager.get_statistics()
print(f"Memory count: {memory_stats['memory_count']}")
print(f"Cache size: {memory_stats['cache_size']}")
```

## Error Handling
- ValueError: Raised for invalid inputs or operations
- ValidationError: Raised when validation fails
- TypeError: Raised for type validation failures

## Configuration
- PromptManager:
  - name: Name of the manager
  - description: Description of the manager
  - template_format: Format of prompt templates
  - add_timestamps: Whether to add timestamps to prompts
  - max_history_items: Maximum number of history items to include
  - max_examples: Maximum number of examples to include
  - cache_size: Size of the prompt cache

- ValidationManager:
  - rules: List of validation rules
  - name: Name of the manager
  - description: Description of the manager
  - prioritize_by_cost: Whether to sort rules by cost
  - fail_fast: Whether to stop on first failure
  - cache_size: Size of the validation cache

- MemoryManager:
  - memories: List of memory stores
  - name: Name of the manager
  - description: Description of the manager
  - cache_enabled: Whether to enable caching
  - max_items: Maximum number of items to store in memory
"""

from .prompt import (
    PromptManager,
    PromptConfig,
    PromptResult,
    BasePrompt,
    create_prompt_manager,
)
from .validation import (
    ValidationManager,
    ValidationConfig,
    ValidationResult,
    create_validation_manager,
)
from sifaka.core.managers.memory import (
    KeyValueMemoryManager as MemoryManager,
    MemoryConfig,
    MemoryResult,
    BaseMemory,
    create_key_value_memory_manager as create_memory_manager,
)

__all__ = [
    # Prompt management
    "PromptManager",
    "PromptConfig",
    "PromptResult",
    "BasePrompt",
    "create_prompt_manager",
    # Validation management
    "ValidationManager",
    "ValidationConfig",
    "ValidationResult",
    "create_validation_manager",
    # Memory management
    "MemoryManager",
    "MemoryConfig",
    "MemoryResult",
    "BaseMemory",
    "create_memory_manager",
]
