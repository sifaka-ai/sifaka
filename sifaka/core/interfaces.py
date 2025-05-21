"""
Core interfaces for Sifaka components.

This module defines the interfaces for models, validators, critics, and retrievers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought


class Model(ABC):
    """Interface for text generation models."""

    @abstractmethod
    def generate(self, thought: Thought) -> str:
        """
        Generate text based on the prompt and context in the thought.

        Args:
            thought: The thought containing the prompt and context.

        Returns:
            The generated text.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""
        pass


class Validator(ABC):
    """Interface for validators."""

    @abstractmethod
    def validate(self, thought: Thought) -> bool:
        """
        Validate the text in the thought.

        Args:
            thought: The thought containing the text to validate.

        Returns:
            True if the text passes validation, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the validator."""
        pass


class Critic(ABC):
    """Interface for critics."""

    @abstractmethod
    def critique(self, thought: Thought) -> str:
        """
        Critique the text in the thought and provide feedback.

        Args:
            thought: The thought containing the text to critique.

        Returns:
            Feedback on how to improve the text.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the critic."""
        pass


class Retriever(ABC):
    """Interface for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, thought: Optional[Thought] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information based on the query.

        Args:
            query: The query to search for.
            thought: Optional thought to add context to the query.

        Returns:
            A list of retrieved documents.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the retriever."""
        pass


class PersistenceProvider(ABC):
    """Interface for persistence providers."""

    @abstractmethod
    def save(self, thought: Thought) -> str:
        """
        Save a thought to the persistence store.

        Args:
            thought: The thought to save.

        Returns:
            A unique identifier for the saved thought.
        """
        pass

    @abstractmethod
    def load(self, thought_id: str) -> Thought:
        """
        Load a thought from the persistence store.

        Args:
            thought_id: The unique identifier of the thought to load.

        Returns:
            The loaded thought.
        """
        pass

    @abstractmethod
    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List thought IDs in the persistence store.

        Args:
            filter_criteria: Optional criteria to filter thoughts.

        Returns:
            A list of thought IDs.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the persistence provider."""
        pass
