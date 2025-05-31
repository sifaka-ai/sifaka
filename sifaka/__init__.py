"""Sifaka: A framework for building reliable AI text generation chains.

Sifaka provides a comprehensive framework for building AI text generation
pipelines with validation, criticism, and improvement capabilities using PydanticAI.

Key components:
- PydanticAI Chains: Modern agent-based text generation workflows
- Models: Interfaces to various language models (OpenAI, Anthropic, etc.)
- Validators: Components for validating generated text
- Critics: Components for providing feedback and improvement suggestions
- Classifiers: Text classification for content moderation
- Retrievers: Document retrieval for context-aware generation

Example:
    ```python
    from pydantic_ai import Agent
    from sifaka.agents import create_pydantic_chain
    from sifaka.validators import LengthValidator
    from sifaka.critics import ReflexionCritic
    from sifaka.models import create_model

    # Create PydanticAI agent
    agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant.")

    # Create Sifaka components
    validator = LengthValidator(min_length=10, max_length=1000)
    critic = ReflexionCritic(model=create_model("openai:gpt-3.5-turbo"))

    # Create chain
    chain = create_pydantic_chain(
        agent=agent,
        validators=[validator],
        critics=[critic]
    )

    # Run the chain
    result = chain.run("Write a friendly greeting.")
    print(result.text)
    ```
"""

# Standard library imports
from typing import List

# Sifaka imports (absolute paths only)
from sifaka.core.interfaces import Critic, Model, Retriever, Validator
from sifaka.core.thought import CriticFeedback, Document, Thought, ValidationResult

__all__: List[str] = [
    # Core components
    "Thought",
    "Document",
    "ValidationResult",
    "CriticFeedback",
    # Interfaces
    "Model",
    "Validator",
    "Critic",
    "Retriever",
]

# Version info
__version__ = "0.3.0"
