"""Thought module for Sifaka.

This module contains the thought state management components:
- Thought: Main thought state container
- ThoughtStorage: Thought persistence management
- ThoughtHistory: History and reference management
"""

# Import main classes for backward compatibility
from sifaka.core.thought.thought import (
    CriticFeedback,
    Document,
    Thought,
    ThoughtReference,
    ValidationResult,
)

__all__ = [
    "Thought",
    "Document",
    "ValidationResult",
    "CriticFeedback",
    "ThoughtReference",
]
