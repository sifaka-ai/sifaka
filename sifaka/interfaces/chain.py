"""
Chain Protocol Module

This module defines the protocol for chains in the Sifaka framework.
Chains are components that orchestrate the flow of text generation,
validation, and improvement.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .component import ComponentProtocol


@runtime_checkable
class ChainProtocol(ComponentProtocol, Protocol):
    """
    Protocol for chains in Sifaka.

    Chains are components that orchestrate the flow of text generation,
    validation, and improvement. They coordinate model providers, validators,
    and improvers to generate and refine text.
    """

    def run(self, prompt: str, **kwargs: Any) -> Any:
        """
        Run the chain with the given prompt.

        Args:
            prompt: The prompt for text generation
            **kwargs: Additional arguments for chain execution

        Returns:
            Chain execution results
        """
        ...

    def validate(self, text: str, **kwargs: Any) -> Any:
        """
        Validate the given text using the chain's validators.

        Args:
            text: The text to validate
            **kwargs: Additional arguments for validation

        Returns:
            Validation results
        """
        ...

    def improve(self, text: str, feedback: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Improve the given text using the chain's improvers.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement
            **kwargs: Additional arguments for improvement

        Returns:
            Improvement results
        """
        ...
