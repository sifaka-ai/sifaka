"""Core components of the Sifaka framework.

This module contains the fundamental building blocks of Sifaka:
- Thought: Central state container for tracking generation process
- Interfaces: Core protocols and abstract base classes

The core architecture follows a thought-centric design where a central Thought
container flows through PydanticAI agents with Sifaka components (validators, critics)
providing validation and improvement capabilities.

Components:
    Thought: Immutable state container with complete audit trail
    Model: Protocol for language model implementations
    Validator: Protocol for text validation components
    Critic: Protocol for text improvement components
    Retriever: Protocol for context retrieval components

Example:
    ```python
    from pydantic_ai import Agent
    from sifaka.agents import create_pydantic_chain
    from sifaka.core.thought import Thought
    from sifaka.core.interfaces import Model, Validator, Critic

    # Create a PydanticAI-based chain
    agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant.")
    chain = create_pydantic_chain(
        agent=agent,
        validators=[validator],
        critics=[critic]
    )

    # Execute and get complete results
    thought = chain.run("Write about AI")
    print(f"Generated: {thought.text}")
    print(f"Iterations: {thought.iteration}")
    ```

Import Patterns:
    Direct imports (recommended for internal development):
    ```python
    from sifaka.core.thought import Thought, Document, ValidationResult
    from sifaka.core.interfaces import Model, Validator, Critic, Retriever
    ```

    Main package imports (recommended for external usage):
    ```python
    from sifaka import Thought, Model, Validator, Critic, Retriever
    from sifaka.agents import create_pydantic_chain
    ```
"""

# No imports at package level to avoid potential circular import issues
# Import these components directly from their modules:
# from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
# from sifaka.core.interfaces import Model, Validator, Critic, Retriever

__all__: list[str] = []
