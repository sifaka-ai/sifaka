from typing import Any, List
"""
Managers for critics.

This package provides specialized managers for different aspects of critics:

## Core Managers

1. **PromptManager**
   - Manages prompt creation and formatting
   - Handles template management
   - Provides prompt validation
   - Supports different prompt types (validation, critique, improvement, reflection)

2. **ResponseParser**
   - Parses responses from language models
   - Validates response formats
   - Extracts structured data
   - Handles different response types
   - Note: This is now imported from sifaka.core.managers.response

3. **MemoryManager**
   - Manages memory for critics
   - Implements circular buffer
   - Handles memory retrieval
   - Supports memory clearing
   - Note: This is now imported from sifaka.core.managers.memory

## Component Lifecycle

### Manager Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Resource allocation
   - Error handling setup
   - State initialization

2. **Usage Phase**
   - Component coordination
   - Data processing
   - Error handling
   - State management

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery
   - Final logging

### Component Interactions

1. **Critic Core**
   - Coordinates managers
   - Handles errors
   - Manages state
   - Provides configuration

2. **Model Provider**
   - Receives prompts
   - Sends responses
   - Handles errors
   - Manages resources

3. **Validation System**
   - Validates prompts
   - Processes responses
   - Handles errors
   - Manages state

## Error Handling

1. **Input Validation Errors**
   - Invalid configurations
   - Malformed data
   - Missing resources
   - Recovery: Skip invalid operations

2. **Processing Errors**
   - Format mismatches
   - Parsing failures
   - Memory issues
   - Recovery: Use fallback values

3. **Resource Errors**
   - Allocation failures
   - Cleanup issues
   - State corruption
   - Recovery: Reset state

## Examples

```python
from sifaka.critics.managers import (
    MemoryManager,
    PromptManager,
    ResponseParser,
    create_response_parser,
    DefaultPromptManager
)

# Create managers
memory_manager = MemoryManager(buffer_size=5)
prompt_manager = DefaultPromptManager()
response_parser = create_response_parser()  # Use factory function

# Use managers together
text = "Sample text to process"
prompt = prompt_manager.create_critique_prompt(text) if prompt_manager else ""
response = model_provider.generate(prompt) if model_provider else ""
critique = response_parser.parse_critique_response(response) if response_parser else ""
memory_manager.add_to_memory(critique["feedback"]) if memory_manager else ""

# Get memory items
recent_items = memory_manager.get_memory(max_items=2) if memory_manager else ""
print(f"Recent feedback: {recent_items}")
```
"""
from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
from sifaka.core.managers.prompt import DefaultPromptManager, CriticPromptManager as PromptManager, PromptCriticPromptManager, ReflexionCriticPromptManager
from sifaka.core.managers.response import ResponseParser, create_response_parser
__all__: List[Any] = ['MemoryManager', 'PromptManager',
    'DefaultPromptManager', 'PromptCriticPromptManager',
    'ReflexionCriticPromptManager', 'ResponseParser', 'create_response_parser']
