"""
Prompt manager interface for Sifaka.

This module defines the interface for prompt managers in the Sifaka framework.
These interfaces establish a common contract for prompt manager behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **PromptManager**: Interface for prompt managers

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

# Type variables
PromptType = TypeVar("PromptType")


@runtime_checkable
class PromptManager(Protocol[PromptType]):
    """
    Interface for prompt managers.

    This interface defines the contract for components that manage prompts.
    It ensures that prompt managers can create, format, and validate prompts.

    ## Lifecycle

    1. **Initialization**: Set up prompt management resources
    2. **Prompt Creation**: Create prompts from inputs
    3. **Prompt Formatting**: Format prompts for models
    4. **Prompt Validation**: Validate prompts
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a create_prompt method to create prompts from inputs
    - Provide a format_prompt method to format prompts for models
    - Provide a validate_prompt method to validate prompts
    """

    @abstractmethod
    def create_prompt(self, input_value: Any, **kwargs: Any) -> PromptType:
        """
        Create a prompt from an input value.

        Args:
            input_value: The input value to create a prompt from
            **kwargs: Additional prompt creation parameters

        Returns:
            A prompt

        Raises:
            ValueError: If the input value is invalid
        """

    @abstractmethod
    def format_prompt(self, prompt: PromptType, **kwargs: Any) -> Any:
        """
        Format a prompt for a model.

        Args:
            prompt: The prompt to format
            **kwargs: Additional prompt formatting parameters

        Returns:
            A formatted prompt

        Raises:
            ValueError: If the prompt is invalid
        """

    @abstractmethod
    def validate_prompt(self, prompt: PromptType) -> bool:
        """
        Validate a prompt.

        Args:
            prompt: The prompt to validate

        Returns:
            True if the prompt is valid, False otherwise

        Raises:
            ValueError: If the prompt is invalid
        """
