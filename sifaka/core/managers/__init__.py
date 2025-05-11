"""
Core managers for Sifaka.

This package provides core manager implementations that can be used across
different components of the Sifaka library.

## Core Managers

1. **Memory Managers**
   - KeyValueMemoryManager: Manages key-value based memory storage and retrieval
   - BufferMemoryManager: Manages a circular buffer of string items
   - Supports multiple memory stores
   - Implements caching for performance
   - Tracks statistics for monitoring
   - Provides consistent state management

2. **Prompt Managers**
   - PromptManager: Base class for prompt management
   - DefaultPromptManager: Default implementation for critics
   - CriticPromptManager: Interface for critic prompt managers
   - PromptCriticPromptManager: Specialized implementation for prompt critics
   - ReflexionCriticPromptManager: Specialized implementation for reflexion critics
   - SelfRefineCriticPromptManager: Specialized implementation for self-refine critics
   - ConstitutionalCriticPromptManager: Specialized implementation for constitutional critics

3. **Response Parsers**
   - ResponseParser: Parses responses from language models
   - Handles validation, critique, improvement, and reflection responses
   - Provides consistent error handling
   - Tracks statistics for monitoring
"""

from .memory import (
    KeyValueMemoryManager,
    MemoryConfig,
    MemoryResult,
    BaseMemory,
    create_key_value_memory_manager,
    BufferMemoryManager,
    create_buffer_memory_manager,
)

from .prompt import (
    PromptManager,
    PromptConfig,
    PromptResult,
    BasePrompt,
    create_prompt_manager,
    DefaultPromptManager,
    CriticPromptManager,
    PromptCriticPromptManager,
    ReflexionCriticPromptManager,
)

from .prompt_factories import (
    SelfRefineCriticPromptManager,
    ConstitutionalCriticPromptManager,
)

from .response import (
    ResponseParser,
    create_response_parser,
)

__all__ = [
    # Key-Value Memory Management
    "KeyValueMemoryManager",
    "MemoryConfig",
    "MemoryResult",
    "BaseMemory",
    "create_key_value_memory_manager",
    # Buffer Memory Management
    "BufferMemoryManager",
    "create_buffer_memory_manager",
    # Prompt Management
    "PromptManager",
    "PromptConfig",
    "PromptResult",
    "BasePrompt",
    "create_prompt_manager",
    "DefaultPromptManager",
    "CriticPromptManager",
    "PromptCriticPromptManager",
    "ReflexionCriticPromptManager",
    # Prompt Factories
    "SelfRefineCriticPromptManager",
    "ConstitutionalCriticPromptManager",
    # Response Parsing
    "ResponseParser",
    "create_response_parser",
]
