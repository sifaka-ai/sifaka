"""
Protocol definitions for critics in Sifaka.

This module defines the protocols that critics must implement to be compatible
with the Sifaka framework. These protocols define the interfaces for text validation,
improvement, and critiquing.
"""

from typing import Any, Dict, List, Protocol, Union, runtime_checkable


@runtime_checkable
class TextValidator(Protocol):
    """Protocol for text validation."""

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text passes validation, False otherwise
        """
        ...

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
class TextImprover(Protocol):
    """Protocol for text improvement."""

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
class TextCritic(Protocol):
    """Protocol for text critiquing."""

    def critique(self, text: str) -> dict:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            dict: A dictionary containing critique information
        """
        ...

    async def critique(self, text: str) -> dict:
        """
        Asynchronously critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            dict: A dictionary containing critique information
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for language model providers."""

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional arguments for the model

        Returns:
            str: The generated text
        """
        ...

    async def generate(self, prompt: str, **kwargs) -> str:
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
class PromptFactory(Protocol):
    """Protocol for prompt factories."""

    def create_prompt(self, text: str, **kwargs) -> str:
        """
        Create a prompt for a language model.

        Args:
            text: The text to create a prompt for
            **kwargs: Additional arguments for prompt creation

        Returns:
            str: The created prompt
        """
        ...

    async def create_prompt(self, text: str, **kwargs) -> str:
        """
        Asynchronously create a prompt for a language model.

        Args:
            text: The text to create a prompt for
            **kwargs: Additional arguments for prompt creation

        Returns:
            str: The created prompt
        """
        ...


# Export public protocols
__all__ = [
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "LLMProvider",
    "PromptFactory",
]
