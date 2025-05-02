"""
Protocol definitions for critics in Sifaka.

This module defines the protocols that critics must implement to be compatible
with the Sifaka framework. These protocols define the interfaces for text validation,
improvement, and critiquing.
"""

from typing import Any, List, Protocol, TypedDict, runtime_checkable


@runtime_checkable
class SyncTextValidator(Protocol):
    """Protocol for synchronous text validation."""

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text passes validation, False otherwise
        """
        ...


@runtime_checkable
class AsyncTextValidator(Protocol):
    """Protocol for asynchronous text validation."""

    async def validate(self, text: str) -> bool:
        """
        Asynchronously validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text passes validation, False otherwise
        """
        ...


@runtime_checkable
class TextValidator(SyncTextValidator, Protocol):
    """Protocol for text validation (sync version)."""

    ...


@runtime_checkable
class SyncTextImprover(Protocol):
    """Protocol for synchronous text improvement."""

    def improve(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text
        """
        ...


@runtime_checkable
class AsyncTextImprover(Protocol):
    """Protocol for asynchronous text improvement."""

    async def improve(self, text: str, feedback: str) -> str:
        """
        Asynchronously improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text
        """
        ...


@runtime_checkable
class TextImprover(SyncTextImprover, Protocol):
    """Protocol for text improvement (sync version)."""

    ...


class CritiqueResult(TypedDict):
    """Type definition for critique results."""

    score: float
    feedback: str
    issues: List[str]
    suggestions: List[str]


@runtime_checkable
class SyncTextCritic(Protocol):
    """Protocol for synchronous text critiquing."""

    def critique(self, text: str) -> CritiqueResult:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult: A dictionary containing critique information
        """
        ...


@runtime_checkable
class AsyncTextCritic(Protocol):
    """Protocol for asynchronous text critiquing."""

    async def critique(self, text: str) -> CritiqueResult:
        """
        Asynchronously critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult: A dictionary containing critique information
        """
        ...


@runtime_checkable
class TextCritic(SyncTextCritic, Protocol):
    """Protocol for text critiquing (sync version)."""

    ...


@runtime_checkable
class SyncLLMProvider(Protocol):
    """Protocol for synchronous language model providers."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional arguments for the model

        Returns:
            str: The generated text
        """
        ...


@runtime_checkable
class AsyncLLMProvider(Protocol):
    """Protocol for asynchronous language model providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Asynchronously generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional arguments for the model

        Returns:
            str: The generated text
        """
        ...


@runtime_checkable
class LLMProvider(SyncLLMProvider, Protocol):
    """Protocol for language model providers (sync version)."""

    ...


@runtime_checkable
class SyncPromptFactory(Protocol):
    """Protocol for synchronous prompt factories."""

    def create_prompt(self, text: str, **kwargs: Any) -> str:
        """
        Create a prompt for a language model.

        Args:
            text: The text to create a prompt for
            **kwargs: Additional arguments for prompt creation

        Returns:
            str: The created prompt
        """
        ...


@runtime_checkable
class AsyncPromptFactory(Protocol):
    """Protocol for asynchronous prompt factories."""

    async def create_prompt(self, text: str, **kwargs: Any) -> str:
        """
        Asynchronously create a prompt for a language model.

        Args:
            text: The text to create a prompt for
            **kwargs: Additional arguments for prompt creation

        Returns:
            str: The created prompt
        """
        ...


@runtime_checkable
class PromptFactory(SyncPromptFactory, Protocol):
    """Protocol for prompt factories (sync version)."""

    ...


# Export public protocols
__all__ = [
    # Synchronous protocols
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "LLMProvider",
    "PromptFactory",
    # Synchronous explicit protocols
    "SyncTextValidator",
    "SyncTextImprover",
    "SyncTextCritic",
    "SyncLLMProvider",
    "SyncPromptFactory",
    # Asynchronous protocols
    "AsyncTextValidator",
    "AsyncTextImprover",
    "AsyncTextCritic",
    "AsyncLLMProvider",
    "AsyncPromptFactory",
    # Type definitions
    "CritiqueResult",
]
