"""Thought state management for Sifaka.

This module implements the central Thought container that serves as the core
state management system for Sifaka's text generation workflows. The Thought
follows an immutable design pattern where all operations return new instances,
providing complete audit trails and iteration history.

Components:
    Thought: Main immutable state container for text generation workflows
    Document: Container for retrieved context documents
    ValidationResult: Results from validator execution
    CriticFeedback: Feedback and suggestions from critics
    ThoughtReference: References to related thoughts for retrieval

The Thought container tracks:
- Original prompt and system messages
- Generated text across iterations
- Validation results and critic feedback
- Retrieved context documents
- Complete iteration history
- Exact prompts sent to models

Example:
    ```python
    from sifaka.core.thought import Thought, Document, ValidationResult

    # Create initial thought
    thought = Thought(
        prompt="Write about AI",
        system_prompt="You are a helpful assistant"
    )

    # Thoughts are immutable - operations return new instances
    updated_thought = thought.set_text("AI is artificial intelligence...")
    next_iteration = thought.next_iteration()

    # Add validation results
    validation = ValidationResult(
        validator_name="length",
        passed=True,
        message="Text meets length requirements"
    )
    validated_thought = thought.add_validation_result(validation)

    # Complete audit trail
    print(f"Current iteration: {thought.iteration}")
    print(f"History length: {len(thought.history)}")
    print(f"Validation results: {len(thought.validation_results)}")
    ```
"""

# Import utility classes
from sifaka.core.thought.history import ThoughtHistory
from sifaka.core.thought.storage import ThoughtStorage

# Import main classes for backward compatibility
from sifaka.core.thought.thought import (
    CriticFeedback,
    Document,
    Thought,
    ThoughtReference,
    ToolCall,
    ValidationResult,
)

__all__ = [
    "Thought",
    "Document",
    "ValidationResult",
    "CriticFeedback",
    "ThoughtReference",
    "ToolCall",
    "ThoughtHistory",
    "ThoughtStorage",
]
