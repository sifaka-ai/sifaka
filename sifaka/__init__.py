"""Sifaka: A framework for building reliable AI text generation chains.

Sifaka provides a comprehensive framework for building AI text generation
pipelines with validation, criticism, and improvement capabilities.

Key components:
- Chain: The main orchestrator for text generation workflows
- Models: Interfaces to various language models (OpenAI, Anthropic, etc.)
- Validators: Components for validating generated text
- Critics: Components for providing feedback and improvement suggestions
- Classifiers: Text classification for content moderation
- Retrievers: Document retrieval for context-aware generation

Example:
    ```python
    from sifaka import Chain
    from sifaka.models import create_model
    from sifaka.validators import LengthValidator

    # Create a simple chain with validation
    model = create_model("mock:gpt-4")
    validator = LengthValidator(min_length=10, max_length=1000)

    result = (Chain()
        .with_model(model)
        .with_prompt("Write a friendly greeting.")
        .validate_with(validator)
        .run())

    print(result.text)
    ```
"""

# Standard library imports
from typing import List

# Sifaka imports (absolute paths only)
from sifaka.core.chain import Chain
from sifaka.quickstart import QuickStart
from sifaka.core.interfaces import Critic, Model, Retriever, Validator
from sifaka.core.thought import CriticFeedback, Document, Thought, ValidationResult

__all__: List[str] = [
    # Core components
    "Chain",
    "QuickStart",
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
__version__ = "0.1.0"
