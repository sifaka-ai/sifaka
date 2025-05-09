"""
Prompt manager module for Sifaka.

This module provides the PromptManager class which is responsible for
creating, modifying, and managing prompts.
"""

from typing import Any, List, Optional

from ..interfaces.manager import PromptManagerProtocol
from ...utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager(PromptManagerProtocol[str]):
    """
    Manages prompts for chains.

    This class is responsible for creating, modifying, and managing prompts
    for use in chains. It implements the PromptManagerProtocol interface.
    """

    def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
        """
        Create a new prompt with feedback.

        Args:
            original_prompt: The original prompt
            feedback: The feedback to include

        Returns:
            A new prompt with feedback
        """
        return f"{original_prompt}\n\nPrevious attempt feedback:\n{feedback}"

    def create_prompt_with_history(self, original_prompt: str, history: List[str]) -> str:
        """
        Create a new prompt with history.

        Args:
            original_prompt: The original prompt
            history: The history to include

        Returns:
            A new prompt with history
        """
        history_text = "\n".join(history)
        return f"{original_prompt}\n\nPrevious attempts:\n{history_text}"

    def create_prompt_with_context(self, original_prompt: str, context: str) -> str:
        """
        Create a new prompt with context.

        Args:
            original_prompt: The original prompt
            context: The context to include

        Returns:
            A new prompt with context
        """
        return f"Context:\n{context}\n\nPrompt:\n{original_prompt}"

    def create_prompt_with_examples(self, original_prompt: str, examples: List[str]) -> str:
        """
        Create a new prompt with examples.

        Args:
            original_prompt: The original prompt
            examples: The examples to include

        Returns:
            A new prompt with examples
        """
        examples_text = "\n".join(
            [f"Example {i+1}: {example}" for i, example in enumerate(examples)]
        )
        return f"{original_prompt}\n\nExamples:\n{examples_text}"

    def create_prompt(self, input_value: Any, **kwargs: Any) -> str:
        """
        Create a prompt from an input value.

        This method implements the PromptManagerProtocol.create_prompt method.

        Args:
            input_value: The input value to create a prompt from
            **kwargs: Additional prompt creation parameters

        Returns:
            A prompt

        Raises:
            ValueError: If the input value is invalid
        """
        if not isinstance(input_value, str):
            raise ValueError(f"Expected string input, got {type(input_value)}")

        # Process additional parameters
        feedback = kwargs.get("feedback")
        history = kwargs.get("history")
        context = kwargs.get("context")
        examples = kwargs.get("examples")

        prompt = input_value

        # Apply transformations based on parameters
        if feedback:
            prompt = self.create_prompt_with_feedback(prompt, feedback)
        if history:
            prompt = self.create_prompt_with_history(prompt, history)
        if context:
            prompt = self.create_prompt_with_context(prompt, context)
        if examples:
            prompt = self.create_prompt_with_examples(prompt, examples)

        return prompt

    def format_prompt(self, prompt: str, **kwargs: Any) -> Any:
        """
        Format a prompt for a model.

        This method implements the PromptManagerProtocol.format_prompt method.

        Args:
            prompt: The prompt to format
            **kwargs: Additional prompt formatting parameters

        Returns:
            A formatted prompt

        Raises:
            ValueError: If the prompt is invalid
        """
        # For now, just return the prompt as is
        # In a real implementation, this would format the prompt for the model
        return prompt

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate a prompt.

        This method implements the PromptManagerProtocol.validate_prompt method.

        Args:
            prompt: The prompt to validate

        Returns:
            True if the prompt is valid, False otherwise

        Raises:
            ValueError: If the prompt is invalid
        """
        # For now, just check if the prompt is a non-empty string
        if not isinstance(prompt, str):
            raise ValueError(f"Expected string prompt, got {type(prompt)}")
        return bool(prompt.strip())
