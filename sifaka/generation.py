"""
Text generation module for Sifaka.

This module provides the Generator class which is responsible for
generating text using model providers.
"""

from typing import Generic, TypeVar, Any

from sifaka.models.base import ModelProvider
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class Generator(Generic[OutputType]):
    """
    Handles text generation using model providers.

    This class is responsible for generating text using model providers.
    It provides a consistent interface for text generation across different
    model providers.
    """

    def __init__(self, model: ModelProvider):
        """
        Initialize a Generator instance.

        Args:
            model: The model provider to use for text generation
        """
        self._model = model

    def generate(self, prompt: str) -> OutputType:
        """
        Generate text using the model.

        Args:
            prompt: The prompt to generate from

        Returns:
            The generated text

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            RuntimeError: If generation fails
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        try:
            # Use the model provider to generate text
            return self._model.generate(prompt)  # type: ignore
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text: {error_msg}")
            raise RuntimeError(f"Error generating text: {error_msg}") from e
