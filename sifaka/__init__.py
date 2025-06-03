"""Sifaka - PydanticAI-native AI validation, improvement, and evaluation framework.

This is a complete rewrite of Sifaka built on PydanticAI's graph capabilities.
The new architecture provides:

- Graph-based workflow orchestration using pydantic_graph
- Pure async implementation throughout
- State persistence for resumable workflows
- Parallel execution of validators and critics
- Rich observability and analytics
- Type-safe operations with Pydantic models

Key Components:
- SifakaEngine: Main orchestration engine
- SifakaThought: Core state container with full audit trail
- Graph Nodes: Generate, Validate, and Critique operations
- Critics: Research-based improvement agents (Reflexion, Constitutional, Self-Refine)
- Validators: Content validation (length, coherence, factual accuracy)
- Storage: Pluggable storage backends (memory, file, Redis)

Example Usage:
    ```python
    from sifaka import SifakaEngine

    # Create engine with default configuration
    engine = SifakaEngine()

    # Process a single thought
    thought = await engine.think("Explain renewable energy")
    print(thought.final_text)

    # Continue conversation
    follow_up = await engine.continue_thought(thought, "Focus on solar panels")
    print(follow_up.final_text)
    ```

For more advanced usage, see the documentation and examples.
"""

from sifaka.core.engine import SifakaEngine
from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.utils import (
    SifakaError,
    ValidationError,
    CritiqueError,
    GraphExecutionError,
    ConfigurationError,
    SifakaConfig,
)

__version__ = "0.5.0-alpha"
__all__ = [
    # Core components
    "SifakaEngine",
    "SifakaThought",
    "SifakaDependencies",
    # Configuration and utilities
    "SifakaConfig",
    # Error types
    "SifakaError",
    "ValidationError",
    "CritiqueError",
    "GraphExecutionError",
    "ConfigurationError",
]
