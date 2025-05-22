"""
Core interfaces for Sifaka.

This module defines the core interfaces that all Sifaka components must implement.
These interfaces ensure consistent behavior across different implementations and
enable components to work together seamlessly.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from sifaka.core.thought import Thought


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

    def generate_with_thought(self, thought: Thought, **options: Any) -> str:
        """Generate text using a Thought container.

        This method allows models to access the full context in the Thought container,
        including any retrieved documents, when generating text.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            The generated text.
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

    def validate(self, thought: Thought) -> Dict[str, Any]:
        """Validate text against specific criteria.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A dictionary with validation results.
        """
        ...


@runtime_checkable
class Critic(Protocol):
    """Protocol defining the interface for critics.

    This protocol defines the minimum interface that all critic
    implementations must follow. It requires methods for critiquing
    and improving text.
    """

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text and provide feedback.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results.
        """
        ...

    def improve(self, thought: Thought) -> str:
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


@runtime_checkable
class Chain(Protocol):
    """Protocol defining the interface for chains.

    This protocol defines the minimum interface that all chain
    implementations must follow. It requires methods for running
    the chain and configuring it.
    """

    def run(self) -> Thought:
        """Run the chain and return the result.

        Returns:
            The final Thought container with the result.
        """
        ...

    def with_model(self, model: Model) -> "Chain":
        """Set the model for the chain.

        Args:
            model: The model to use.

        Returns:
            The chain instance for method chaining.
        """
        ...

    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt for the chain.

        Args:
            prompt: The prompt to use.

        Returns:
            The chain instance for method chaining.
        """
        ...

    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain.

        Args:
            validator: The validator to add.

        Returns:
            The chain instance for method chaining.
        """
        ...

    def improve_with(self, critic: Critic) -> "Chain":
        """Add a critic to the chain.

        Args:
            critic: The critic to add.

        Returns:
            The chain instance for method chaining.
        """
        ...
