"""
Interfaces for critics.

This package provides protocol interfaces for critics in the Sifaka framework.
These interfaces are imported from the main interfaces directory.

## Interface Hierarchy

1. **Critic**: Base interface for all critics
   - **TextValidator**: Interface for text validators
   - **TextImprover**: Interface for text improvers
   - **TextCritic**: Interface for text critics
   - **LLMProvider**: Interface for language model providers
   - **PromptFactory**: Interface for prompt factories
"""

# Import from the main interfaces directory
from sifaka.interfaces.critic import (
    # Core critic interfaces
    Critic,
    AsyncCritic,
    # Text validation protocols
    TextValidator,
    SyncTextValidator,
    AsyncTextValidator,
    # Text improvement protocols
    TextImprover,
    SyncTextImprover,
    AsyncTextImprover,
    # Text critiquing protocols
    TextCritic,
    SyncTextCritic,
    AsyncTextCritic,
    # Language model protocols
    LLMProvider,
    SyncLLMProvider,
    AsyncLLMProvider,
    # Prompt factory protocols
    PromptFactory,
    SyncPromptFactory,
    AsyncPromptFactory,
    # Type definitions
    CritiqueResult,
)

__all__ = [
    # Core critic interfaces
    "Critic",
    "AsyncCritic",
    # Synchronous protocols
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "LLMProvider",
    "PromptFactory",
    # Synchronous explicit protocols
    "SyncTextValidator",
    "SyncTextImprover",
    "SyncTextCritic",
    "SyncLLMProvider",
    "SyncPromptFactory",
    # Asynchronous protocols
    "AsyncTextValidator",
    "AsyncTextImprover",
    "AsyncTextCritic",
    "AsyncLLMProvider",
    "AsyncPromptFactory",
    # Type definitions
    "CritiqueResult",
]
