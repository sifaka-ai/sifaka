"""
Generation module for Sifaka.

This module provides components for generating outputs using model providers.
"""

from typing import TypeVar, Generic

from .models.base import ModelProvider

OutputType = TypeVar("OutputType")


class Generator(Generic[OutputType]):
    """
    Generator class that handles output generation using model providers.

    This class is responsible for using model providers to generate outputs
    from prompts.
    """

    def __init__(self, model: ModelProvider):
        """
        Initialize a Generator instance.

        Args:
            model: The model provider to use for generation
        """
        self.model = model

    def generate(self, prompt: str) -> OutputType:
        """
        Generate an output from a prompt.

        Args:
            prompt: The prompt to generate from

        Returns:
            Generated output
        """
        return self.model.generate(prompt)