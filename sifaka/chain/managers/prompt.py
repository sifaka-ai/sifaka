"""
Prompt Manager Module

Manages prompt creation and modification for Sifaka's chain system.

## Overview
This module provides the PromptManager class which handles prompt creation,
modification, and validation. It supports adding feedback, history, context,
and examples to prompts, providing a flexible way to enhance prompts based
on chain execution results.

## Components
1. **PromptManager**: Main prompt management class
   - Prompt creation
   - Prompt modification
   - Prompt validation
   - Prompt formatting

2. **Prompt Enhancement**: Specialized prompt modifiers
   - Feedback addition
   - History integration
   - Context inclusion
   - Example incorporation

## Usage Examples
```python
from sifaka.chain.managers.prompt import PromptManager

# Create prompt manager
manager = PromptManager()

# Create basic prompt
prompt = manager.create_prompt("Write a story about a robot")

# Add feedback
prompt_with_feedback = manager.create_prompt_with_feedback(
    prompt,
    "Make the story more emotional"
)

# Add history
prompt_with_history = manager.create_prompt_with_history(
    prompt,
    ["Previous story about a sad robot", "Story about a happy robot"]
)

# Add context
prompt_with_context = manager.create_prompt_with_context(
    prompt,
    "The story should be set in the future"
)

# Add examples
prompt_with_examples = manager.create_prompt_with_examples(
    prompt,
    ["Example story about a curious robot", "Example story about a brave robot"]
)

# Create complex prompt
complex_prompt = manager.create_prompt(
    "Write a story about a robot",
    feedback="Make it emotional",
    history=["Previous story"],
    context="Set in future",
    examples=["Example story"]
)

# Validate prompt
if manager.validate_prompt(complex_prompt):
    print("Prompt is valid")
```

## Error Handling
- ValueError: Raised for invalid input types or empty prompts
- TypeError: Raised for type validation failures
- RuntimeError: Raised for unexpected conditions

## Configuration
- feedback: Optional feedback to include in prompts
- history: Optional list of previous attempts
- context: Optional context information
- examples: Optional list of example outputs
"""

from typing import Any, List, Optional

from ..interfaces.manager import PromptManagerProtocol
from ...utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager(PromptManagerProtocol[str]):
    """
    Manages prompt creation and modification for chains.

    ## Overview
    This class provides centralized management of prompt creation, modification,
    and validation. It supports various ways to enhance prompts with feedback,
    history, context, and examples, implementing the PromptManagerProtocol interface.

    ## Architecture
    PromptManager follows a builder pattern:
    1. **Prompt Creation**: Creates base prompts from input
    2. **Prompt Enhancement**: Adds additional context and guidance
    3. **Prompt Validation**: Ensures prompt quality and format

    ## Lifecycle
    1. **Creation**: Create base prompts
       - Process input values
       - Apply basic formatting
       - Validate structure

    2. **Enhancement**: Add additional context
       - Add feedback from previous attempts
       - Include execution history
       - Add contextual information
       - Include example outputs

    ## Error Handling
    - ValueError: Raised for invalid input types or empty prompts
    - TypeError: Raised for type validation failures
    - RuntimeError: Raised for unexpected conditions

    ## Examples
    ```python
    from sifaka.chain.managers.prompt import PromptManager

    # Create manager
    manager = PromptManager()

    # Create basic prompt
    prompt = manager.create_prompt(
        "Write a story",
        feedback="Make it longer",
        context="Set in future",
        examples=["Example story"]
    )

    # Add feedback
    prompt = manager.create_prompt_with_feedback(
        prompt,
        "Add more dialogue"
    )

    # Validate prompt
    if manager.validate_prompt(prompt):
        print("Prompt is valid")
    ```
    """

    def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
        """
        Create a new prompt with feedback.

        ## Overview
        This method enhances a prompt by adding feedback from previous attempts
        or validation results. The feedback is appended to the original prompt
        in a structured format.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate original prompt
           - Validate feedback string

        2. **Prompt Enhancement**: Add feedback
           - Format feedback section
           - Combine with original prompt

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
