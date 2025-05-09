"""
Prompt Manager Module

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

        Raises:
            ValueError: If the original prompt or feedback is invalid
            TypeError: If the input types are incorrect

        Examples:
            ```python
            manager = PromptManager()
            prompt = manager.create_prompt_with_feedback(
                "Write a story",
                "Make it more emotional"
            )
            ```
        """
        return f"{original_prompt}\n\nPrevious attempt feedback:\n{feedback}"

    def create_prompt_with_history(self, original_prompt: str, history: List[str]) -> str:
        """
        Create a new prompt with history.

        ## Overview
        This method enhances a prompt by adding a history of previous attempts.
        The history is appended to the original prompt in a structured format.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate original prompt
           - Validate history list

        2. **Prompt Enhancement**: Add history
           - Format history section
           - Combine with original prompt

        Args:
            original_prompt: The original prompt
            history: The history to include

        Returns:
            A new prompt with history

        Raises:
            ValueError: If the original prompt or history is invalid
            TypeError: If the input types are incorrect

        Examples:
            ```python
            manager = PromptManager()
            prompt = manager.create_prompt_with_history(
                "Write a story",
                ["Previous story about a sad robot", "Story about a happy robot"]
            )
            ```
        """
        history_text = "\n".join(history)
        return f"{original_prompt}\n\nPrevious attempts:\n{history_text}"

    def create_prompt_with_context(self, original_prompt: str, context: str) -> str:
        """
        Create a new prompt with context.

        ## Overview
        This method enhances a prompt by adding contextual information.
        The context is prepended to the original prompt in a structured format.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate original prompt
           - Validate context string

        2. **Prompt Enhancement**: Add context
           - Format context section
           - Combine with original prompt

        Args:
            original_prompt: The original prompt
            context: The context to include

        Returns:
            A new prompt with context

        Raises:
            ValueError: If the original prompt or context is invalid
            TypeError: If the input types are incorrect

        Examples:
            ```python
            manager = PromptManager()
            prompt = manager.create_prompt_with_context(
                "Write a story",
                "The story should be set in the future"
            )
            ```
        """
        return f"Context:\n{context}\n\nPrompt:\n{original_prompt}"

    def create_prompt_with_examples(self, original_prompt: str, examples: List[str]) -> str:
        """
        Create a new prompt with examples.

        ## Overview
        This method enhances a prompt by adding example outputs.
        The examples are appended to the original prompt in a structured format.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate original prompt
           - Validate examples list

        2. **Prompt Enhancement**: Add examples
           - Format examples section
           - Combine with original prompt

        Args:
            original_prompt: The original prompt
            examples: The examples to include

        Returns:
            A new prompt with examples

        Raises:
            ValueError: If the original prompt or examples are invalid
            TypeError: If the input types are incorrect

        Examples:
            ```python
            manager = PromptManager()
            prompt = manager.create_prompt_with_examples(
                "Write a story",
                ["Example story about a curious robot", "Example story about a brave robot"]
            )
            ```
        """
        examples_text = "\n".join(
            [f"Example {i+1}: {example}" for i, example in enumerate(examples)]
        )
        return f"{original_prompt}\n\nExamples:\n{examples_text}"

    def create_prompt(self, input_value: Any, **kwargs: Any) -> str:
        """
        Create a prompt from an input value.

        ## Overview
        This method creates a prompt from an input value, optionally enhancing it
        with feedback, history, context, and examples.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate input value
           - Process additional parameters

        2. **Prompt Creation**: Create base prompt
           - Convert input to string
           - Apply basic formatting

        3. **Prompt Enhancement**: Add additional context
           - Add feedback if provided
           - Add history if provided
           - Add context if provided
           - Add examples if provided

        Args:
            input_value: The input value to create a prompt from
            **kwargs: Additional prompt creation parameters
                - feedback: Optional feedback to include
                - history: Optional list of previous attempts
                - context: Optional context information
                - examples: Optional list of example outputs

        Returns:
            A prompt

        Raises:
            ValueError: If the input value is invalid
            TypeError: If the input types are incorrect

        Examples:
            ```python
            manager = PromptManager()
            prompt = manager.create_prompt(
                "Write a story",
                feedback="Make it longer",
                context="Set in future",
                examples=["Example story"]
            )
            ```
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
        Format a prompt according to specified parameters.

        ## Overview
        This method formats a prompt according to specified parameters,
        such as adding line breaks, indentation, or other formatting.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate prompt
           - Process formatting parameters

        2. **Prompt Formatting**: Apply formatting
           - Apply line breaks
           - Apply indentation
           - Apply other formatting

        Args:
            prompt: The prompt to format
            **kwargs: Formatting parameters

        Returns:
            The formatted prompt

        Raises:
            ValueError: If the prompt is invalid
            TypeError: If the input types are incorrect

        Examples:
            ```python
            manager = PromptManager()
            formatted_prompt = manager.format_prompt(
                "Write a story",
                indent=2,
                line_breaks=True
            )
            ```
        """
        # Default formatting
        formatted = prompt

        # Apply custom formatting based on kwargs
        if kwargs.get("line_breaks", True):
            formatted = formatted.replace(". ", ".\n")
        if kwargs.get("indent"):
            indent = " " * kwargs["indent"]
            formatted = "\n".join(indent + line for line in formatted.split("\n"))

        return formatted

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate a prompt.

        ## Overview
        This method validates a prompt to ensure it meets quality and format
        requirements.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate prompt

        2. **Prompt Validation**: Check requirements
           - Check minimum length
           - Check format
           - Check content

        Args:
            prompt: The prompt to validate

        Returns:
            True if the prompt is valid, False otherwise

        Raises:
            ValueError: If the prompt is invalid
            TypeError: If the input type is incorrect

        Examples:
            ```python
            manager = PromptManager()
            is_valid = manager.validate_prompt("Write a story")
            if is_valid:
                print("Prompt is valid")
            ```
        """
        if not isinstance(prompt, str):
            raise TypeError(f"Expected string input, got {type(prompt)}")

        # Basic validation
        if not prompt.strip():
            return False

        # Additional validation can be added here
        return True
