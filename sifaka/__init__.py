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
    from sifaka.classifiers import create_toxicity_validator

    # Create a simple chain with toxicity checking
    result = (Chain()
        .with_model("mock:gpt-4")
        .with_prompt("Write a friendly greeting.")
        .validate_with(create_toxicity_validator())
        .run())

    print(result.text)
    ```
"""

from typing import List

# Import core components
from sifaka.chain import Chain
from sifaka.core.thought import Thought

# Import key interfaces
from sifaka.core.interfaces import Model, Validator, Critic

__all__: List[str] = [
    "Chain",
    "Thought",
    "Model",
    "Validator",
    "Critic",
]

# Version info
__version__ = "0.1.0"
