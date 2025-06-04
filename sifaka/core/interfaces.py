"""Core interfaces and protocols for Sifaka.

This module defines the core protocols and interfaces that components must implement
to work with the Sifaka graph-based workflow system.

Key interfaces:
- Validator: For content validation
- Critic: For iterative improvement feedback
- Storage: For thought persistence
- Retriever: For context retrieval
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from sifaka.core.thought import SifakaThought


class Validator(Protocol):
    """Protocol for content validators.

    Validators check generated content against specific criteria and return
    structured results indicating whether the content passes validation.
    """

    async def validate_async(self, text: str) -> Dict[str, Any]:
        """Validate text content asynchronously.

        Args:
            text: The text to validate

        Returns:
            Dictionary containing:
            - passed: bool indicating if validation passed
            - details: Dict with validation details
            - score: Optional float score (0.0-1.0)
        """
        ...

    @property
    def name(self) -> str:
        """Get the validator name for identification."""
        ...


class Critic(Protocol):
    """Protocol for critics that provide improvement feedback.

    Critics analyze generated content and provide structured feedback
    for iterative improvement.
    """

    async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
        """Provide critique feedback for a thought.

        Args:
            thought: The thought to critique

        Returns:
            Dictionary containing:
            - feedback: str with critique feedback
            - suggestions: List[str] with improvement suggestions
            - needs_improvement: bool indicating if improvement is needed
        """
        ...

    async def improve_async(self, thought: SifakaThought) -> str:
        """Generate improved text based on critique.

        Args:
            thought: The thought to improve

        Returns:
            Improved text string
        """
        ...

    @property
    def name(self) -> str:
        """Get the critic name for identification."""
        ...


class Storage(Protocol):
    """Protocol for thought storage backends.

    Storage implementations persist thoughts for analytics, debugging,
    and resumable workflows.
    """

    async def store_thought(self, thought: SifakaThought) -> None:
        """Store a thought.

        Args:
            thought: The thought to store
        """
        ...

    async def retrieve_thought(self, thought_id: str) -> Optional[SifakaThought]:
        """Retrieve a thought by ID.

        Args:
            thought_id: The thought ID to retrieve

        Returns:
            The thought if found, None otherwise
        """
        ...

    async def list_thoughts(
        self, conversation_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[SifakaThought]:
        """List thoughts, optionally filtered by conversation.

        Args:
            conversation_id: Optional conversation ID to filter by
            limit: Optional limit on number of thoughts to return

        Returns:
            List of thoughts
        """
        ...


class Retriever(Protocol):
    """Protocol for context retrievers.

    Retrievers provide relevant context for generation and critique operations.
    """

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query.

        Args:
            query: The query to retrieve context for
            top_k: Maximum number of results to return

        Returns:
            List of context documents with metadata
        """
        ...

    @property
    def name(self) -> str:
        """Get the retriever name for identification."""
        ...


class BaseValidator(ABC):
    """Abstract base class for validators.

    Provides common functionality for validator implementations.
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get the validator name."""
        return self._name

    @abstractmethod
    async def validate_async(self, text: str) -> Dict[str, Any]:
        """Validate text content asynchronously.

        Must be implemented by subclasses.
        """
        pass

    def validate_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous validation (not recommended for new code).

        This is provided for backward compatibility but should not be used
        in new implementations. Use validate_async instead.
        """
        import asyncio

        return asyncio.run(self.validate_async(text))


class BaseCritic(ABC):
    """Abstract base class for critics.

    Provides common functionality for critic implementations.
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get the critic name."""
        return self._name

    @abstractmethod
    async def critique_async(self, thought: SifakaThought) -> Dict[str, Any]:
        """Provide critique feedback for a thought.

        Must be implemented by subclasses.
        """
        pass

    async def improve_async(self, thought: SifakaThought) -> str:
        """Generate improved text based on critique.

        Default implementation raises NotImplementedError.
        Subclasses can override to provide improvement functionality.
        """
        raise NotImplementedError(f"{self.name} does not support improvement")


class BaseStorage(ABC):
    """Abstract base class for storage implementations.

    Provides common functionality for storage backends.
    """

    @abstractmethod
    async def store_thought(self, thought: SifakaThought) -> None:
        """Store a thought."""
        pass

    @abstractmethod
    async def retrieve_thought(self, thought_id: str) -> Optional[SifakaThought]:
        """Retrieve a thought by ID."""
        pass

    @abstractmethod
    async def list_thoughts(
        self, conversation_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[SifakaThought]:
        """List thoughts."""
        pass


class BaseRetriever(ABC):
    """Abstract base class for retrievers.

    Provides common functionality for retriever implementations.
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get the retriever name."""
        return self._name

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query."""
        pass
