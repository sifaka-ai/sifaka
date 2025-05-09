"""
Interfaces for critics.

This package provides protocol interfaces for critics in the Sifaka framework.
"""

from .critic import (
    TextValidator,
    TextImprover,
    TextCritic,
    SyncTextValidator,
    SyncTextImprover,
    SyncTextCritic,
    AsyncTextValidator,
    AsyncTextImprover,
    AsyncTextCritic,
    LLMProvider,
    PromptFactory,
    SyncLLMProvider,
    SyncPromptFactory,
    AsyncLLMProvider,
    AsyncPromptFactory,
    CritiqueResult,
)

__all__ = [
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
