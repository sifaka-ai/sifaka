"""
Core interfaces for Sifaka.

This module defines the core interfaces that all Sifaka components must implement.
These interfaces ensure consistent behavior across different implementations and
enable components to work together seamlessly.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

# Use TYPE_CHECKING imports to avoid circular dependencies
if TYPE_CHECKING:
    from sifaka.core.thought import Thought, ValidationResult


@runtime_checkable
class Model(Protocol):
    """Protocol defining the interface for language model providers.

    This protocol defines the minimum interface that all language model
    implementations must follow. It requires methods for generating text
    from a prompt and counting tokens in text.

    All model implementations in Sifaka must implement this protocol.
    """

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            The generated text.
        """
        ...

    def generate_with_thought(self, thought: "Thought", **options: Any) -> tuple[str, str]:
        """Generate text using a Thought container.

        This method allows models to access the full context in the Thought container,
        including any retrieved documents, when generating text.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.
        """
        ...


@runtime_checkable
class Validator(Protocol):
    """Protocol defining the interface for validators.

    This protocol defines the minimum interface that all validator
    implementations must follow. It requires a validate method that
    checks if text meets certain criteria.
    """

    def validate(self, thought: "Thought") -> "ValidationResult":
        """Validate text against specific criteria.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with validation results.
        """
        ...


@runtime_checkable
class Critic(Protocol):
    """Protocol defining the interface for critics.

    This protocol defines the minimum interface that all critic
    implementations must follow. It requires methods for critiquing
    and improving text.
    """

    def critique(self, thought: "Thought") -> Dict[str, Any]:
        """Critique text and provide feedback.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results.
        """
        ...

    def improve(self, thought: "Thought") -> str:
        """Improve text based on critique.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol defining the interface for retrievers.

    This protocol defines the minimum interface that all retriever
    implementations must follow. It requires a retrieve method that
    finds relevant documents for a query.
    """

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        ...

    def retrieve_for_thought(self, thought: "Thought", is_pre_generation: bool = True) -> "Thought":
        """Retrieve documents and add them to a thought.

        Args:
            thought: The thought to add documents to.
            is_pre_generation: Whether this is pre-generation or post-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        ...


# Chain protocol removed to avoid circular imports with the concrete Chain class
