"""Core components of the Sifaka framework.

This module contains the fundamental building blocks of Sifaka:
- Chain: Main orchestrator for text generation workflows
- Thought: Central state container for tracking generation process
- Interfaces: Core protocols and abstract base classes

Import these components directly from their modules to avoid circular imports:
- from sifaka.core.chain import Chain
- from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
- from sifaka.core.interfaces import Model, Validator, Critic, Retriever
"""

# No imports at package level to avoid circular import issues
__all__: list[str] = []
