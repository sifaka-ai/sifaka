"""Core components of the Sifaka framework.

This module contains the fundamental building blocks of Sifaka:
- Chain: Main orchestrator for text generation workflows
- Thought: Central state container for tracking generation process
- Interfaces: Core protocols and abstract base classes

The core architecture follows a thought-centric design where a central Thought
container flows through a chain of AI components (models, validators, critics)
with complete state tracking and iterative improvement capabilities.

Components:
    Chain: Main fluent API for building text generation workflows
    Thought: Immutable state container with complete audit trail
    Model: Protocol for language model implementations
    Validator: Protocol for text validation components
    Critic: Protocol for text improvement components
    Retriever: Protocol for context retrieval components

Example:
    ```python
    from sifaka.core.chain import Chain
    from sifaka.core.thought import Thought
    from sifaka.core.interfaces import Model, Validator, Critic

    # Create a basic chain
    chain = Chain()
    chain.with_model(model).with_prompt("Write about AI")
    chain.validate_with(validator).improve_with(critic)

    # Execute and get complete results
    thought = chain.run()
    print(f"Generated: {thought.text}")
    print(f"Iterations: {thought.iteration}")
    ```

Import Patterns:
    Direct imports (recommended for internal development):
    ```python
    from sifaka.core.chain import Chain
    from sifaka.core.thought import Thought, Document, ValidationResult
    from sifaka.core.interfaces import Model, Validator, Critic, Retriever
    ```

    Main package imports (recommended for external usage):
    ```python
    from sifaka import Chain, Thought, Model, Validator, Critic, Retriever
    ```
"""

# No imports at package level to avoid potential circular import issues
# Import these components directly from their modules:
# from sifaka.core.chain import Chain
# from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
# from sifaka.core.interfaces import Model, Validator, Critic, Retriever

__all__: list[str] = []
