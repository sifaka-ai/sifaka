"""Thought module for Sifaka.

This module contains the thought state management components:
- Thought: Main thought state container
- ThoughtStorage: Thought persistence management
- ThoughtHistory: History and reference management
"""

# Import main classes for backward compatibility
from sifaka.core.thought.thought import (
    Thought,
    Document,
    ValidationResult,
    CriticFeedback,
    ThoughtReference,
)

__all__ = [
    "Thought",
    "Document",
    "ValidationResult",
    "CriticFeedback",
    "ThoughtReference",
]
