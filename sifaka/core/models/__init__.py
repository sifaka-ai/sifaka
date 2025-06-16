"""Core models for Sifaka.

This module contains the core data models that were extracted from the monolithic
thought.py file to improve maintainability and follow the Single Responsibility Principle.

Models:
- Generation: Text generation tracking with PydanticAI metadata
- ValidationResult: Validation operation results
- CritiqueResult: Critique feedback and suggestions
- ToolCall: Tool execution tracking
- SifakaThought: Main thought container (imports and composes the above)
"""

from sifaka.core.models.generation import Generation
from sifaka.core.models.validation import ValidationResult
from sifaka.core.models.critique import CritiqueResult
from sifaka.core.models.tool_call import ToolCall

# SifakaThought will be imported from the main thought.py file
# to maintain backward compatibility
from sifaka.core.thought import SifakaThought

__all__ = [
    "Generation",
    "ValidationResult", 
    "CritiqueResult",
    "ToolCall",
    "SifakaThought",
]
