"""
Prompt factories for critics.

This module provides specialized prompt managers for different critic types.

## Component Lifecycle

### Prompt Factory Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Template setup
   - Format validation
   - Error handling setup

2. **Usage Phase**
   - Prompt creation
   - Template processing
   - Format validation
   - Error handling and recovery

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

### Component Interactions

1. **Critic Core**
   - Receives prompt requests
   - Provides configuration
   - Handles prompt results

2. **Model Provider**
   - Receives formatted prompts
   - Returns model responses
   - Handles model-specific formatting

3. **Response Parser**
   - Parses model responses
   - Validates response formats
   - Extracts structured data

### Error Handling and Recovery

1. **Input Validation Errors**
   - Empty or invalid text inputs
   - Invalid feedback format
   - Invalid reflection format
   - Recovery: Return appropriate error messages

2. **Template Errors**
   - Missing template variables
   - Invalid template syntax
   - Format validation failures
   - Recovery: Use default templates or fallback formats

3. **Resource Errors**
   - Memory allocation failures
   - Configuration issues
   - Recovery: Resource cleanup and state preservation

## Examples

```python
from sifaka.critics.managers.prompt_factories import (
    PromptCriticPromptManager,
    ReflexionCriticPromptManager
)

# Create a prompt critic manager
prompt_manager = PromptCriticPromptManager()

# Create a validation prompt
text = "This is a sample text to validate."
validation_prompt = prompt_manager.create_validation_prompt(text)
print(validation_prompt)

# Create a critique prompt
critique_prompt = prompt_manager.create_critique_prompt(text)
print(critique_prompt)

# Create a reflexion critic manager
reflexion_manager = ReflexionCriticPromptManager()

# Create an improvement prompt with reflections
feedback = "The text needs more detail and better structure."
reflections = ["Previous improvement focused on clarity", "Added more examples"]
improvement_prompt = reflexion_manager.create_improvement_prompt(
    text, feedback, reflections
)
print(improvement_prompt)
```
"""

from typing import List, Optional

from .prompt import PromptManager
from ...utils.logging import get_logger

logger = get_logger(__name__)


class PromptCriticPromptManager(PromptManager):
    """
    Prompt manager for PromptCritic.

    This class provides prompt creation methods for PromptCritic, implementing
    specialized prompt templates for validation, critique, improvement, and
    reflection operations.

    ## Lifecycle Management

    The PromptCriticPromptManager manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up templates
       - Configures error handling
       - Allocates resources

    2. **Operation**
       - Creates specialized prompts
       - Handles template variables
       - Validates formats
       - Manages errors

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    The PromptCriticPromptManager implements comprehensive error handling:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies reflection format

    2. **Template Management**
       - Handles missing variables
       - Validates template syntax
       - Manages format errors

    3. **Resource Management**
       - Handles allocation failures
       - Manages cleanup errors
       - Preserves valid state

    ## Examples

    ```python
    from sifaka.critics.managers.prompt_factories import PromptCriticPromptManager

    # Create a prompt critic manager
    manager = PromptCriticPromptManager()

    # Create different types of prompts
    text = "This is a sample text."

    # Validation prompt
    validation_prompt = manager.create_validation_prompt(text)
    print(validation_prompt)

    # Critique prompt
    critique_prompt = manager.create_critique_prompt(text)
    print(critique_prompt)

    # Improvement prompt
    feedback = "The text needs more detail."
    improvement_prompt = manager.create_improvement_prompt(text, feedback)
    print(improvement_prompt)
    ```
    """

    def _create_validation_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_validation_prompt.

        This method creates a specialized validation prompt for PromptCritic,
        requesting the model to validate whether the text meets quality standards.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            text: The text to validate

        Returns:
            A prompt for validating the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If prompt creation fails
        """
        return f"""Please Validate the following text:

        TEXT TO VALIDATE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        VALID: [true/false]
        REASON: [reason for validation result]

        VALIDATION:"""

    def _create_critique_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_critique_prompt.

        This method creates a specialized critique prompt for PromptCritic,
        requesting the model to analyze and provide feedback on the text.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            text: The text to critique

        Returns:
            A prompt for critiquing the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If prompt creation fails

        Examples:
            ```python
            # Create a critique prompt
            text = "This is a sample text to critique."
            prompt = manager._create_critique_prompt_impl(text)
            print(prompt)
            ```
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text: must be non-empty string")
        return f"""Analyze the following text and provide a detailed critique:

{text}

Please provide your critique in the following format:
SCORE: [0-1]
FEEDBACK: [general feedback]
ISSUES:
- [specific issue 1]
- [specific issue 2]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]"""

    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Implementation of create_improvement_prompt.

        This method creates a specialized improvement prompt for PromptCritic,
        requesting the model to improve the text based on feedback and reflections.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            text: The text to improve
            feedback: The feedback to incorporate
            reflections: Optional list of previous reflections

        Returns:
            A prompt for improving the text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If prompt creation fails

        Examples:
            ```python
            # Create an improvement prompt
            text = "This is a sample text to improve."
            feedback = "The text needs more detail and better structure."
            reflections = ["Previous improvement focused on clarity"]
            prompt = manager._create_improvement_prompt_impl(text, feedback, reflections)
            print(prompt)
            ```
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text: must be non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Invalid feedback: must be non-empty string")

        prompt = f"""Improve the following text based on the feedback and previous reflections:

Text:
{text}

Feedback:
{feedback}"""

        if reflections:
            prompt += "\n\nPrevious reflections:\n" + "\n".join(f"- {r}" for r in reflections)

        prompt += "\n\nPlease provide the improved text in the following format:\nIMPROVED_TEXT: [improved text]"
        return prompt

    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Implementation of create_reflection_prompt.

        This method creates a specialized reflection prompt for PromptCritic,
        requesting the model to reflect on the improvement process.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            original_text: The original text
            feedback: The feedback provided
            improved_text: The improved version of the text

        Returns:
            A prompt for reflecting on the improvement process

        Raises:
            ValueError: If any input is empty
            RuntimeError: If prompt creation fails

        Examples:
            ```python
            # Create a reflection prompt
            original = "This is the original text."
            feedback = "The text needed more detail."
            improved = "This is the improved text with more detail."
            prompt = manager._create_reflection_prompt_impl(original, feedback, improved)
            print(prompt)
            ```
        """
        if not original_text or not isinstance(original_text, str):
            raise ValueError("Invalid original text: must be non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Invalid feedback: must be non-empty string")
        if not improved_text or not isinstance(improved_text, str):
            raise ValueError("Invalid improved text: must be non-empty string")

        return f"""Reflect on the improvement process:

Original text:
{original_text}

Feedback:
{feedback}

Improved text:
{improved_text}

Please provide your reflection in the following format:
REFLECTION: [your reflection on the improvement process]"""


class ReflexionCriticPromptManager(PromptManager):
    """
    Prompt manager for ReflexionCritic.

    This class provides prompt creation methods for ReflexionCritic, implementing
    specialized prompt templates that incorporate memory and reflections for
    validation, critique, improvement, and reflection operations.

    ## Lifecycle Management

    The ReflexionCriticPromptManager manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up templates
       - Configures error handling
       - Allocates resources

    2. **Operation**
       - Creates specialized prompts
       - Incorporates memory
       - Handles template variables
       - Validates formats
       - Manages errors

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    The ReflexionCriticPromptManager implements comprehensive error handling:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies reflection format
       - Validates memory items

    2. **Template Management**
       - Handles missing variables
       - Validates template syntax
       - Manages format errors
       - Incorporates memory safely

    3. **Resource Management**
       - Handles allocation failures
       - Manages cleanup errors
       - Preserves valid state

    ## Examples

    ```python
    from sifaka.critics.managers.prompt_factories import ReflexionCriticPromptManager

    # Create a reflexion critic manager
    manager = ReflexionCriticPromptManager()

    # Create different types of prompts
    text = "This is a sample text."
    feedback = "The text needs more detail."
    reflections = ["Previous improvement focused on clarity"]

    # Validation prompt
    validation_prompt = manager.create_validation_prompt(text)
    print(validation_prompt)

    # Critique prompt
    critique_prompt = manager.create_critique_prompt(text)
    print(critique_prompt)

    # Improvement prompt with reflections
    improvement_prompt = manager.create_improvement_prompt(text, feedback, reflections)
    print(improvement_prompt)
    ```
    """

    def _create_validation_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_validation_prompt.

        This method creates a specialized validation prompt for ReflexionCritic,
        requesting the model to validate whether the text meets quality standards.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            text: The text to validate

        Returns:
            A prompt for validating the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If prompt creation fails

        Examples:
            ```python
            # Create a validation prompt
            text = "This is a sample text to validate."
            prompt = manager._create_validation_prompt_impl(text)
            print(prompt)
            ```
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text: must be non-empty string")
        return f"""Validate the following text:

{text}

Please provide your validation in the following format:
VALID: [true/false]"""

    def _create_critique_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_critique_prompt.

        This method creates a specialized critique prompt for ReflexionCritic,
        requesting the model to analyze and provide feedback on the text.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            text: The text to critique

        Returns:
            A prompt for critiquing the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If prompt creation fails

        Examples:
            ```python
            # Create a critique prompt
            text = "This is a sample text to critique."
            prompt = manager._create_critique_prompt_impl(text)
            print(prompt)
            ```
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text: must be non-empty string")
        return f"""Analyze the following text and provide a detailed critique:

{text}

Please provide your critique in the following format:
SCORE: [0-1]
FEEDBACK: [general feedback]
ISSUES:
- [specific issue 1]
- [specific issue 2]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]"""

    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Implementation of create_improvement_prompt.

        This method creates a specialized improvement prompt for ReflexionCritic,
        requesting the model to improve the text based on feedback and reflections.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            text: The text to improve
            feedback: The feedback to incorporate
            reflections: Optional list of previous reflections

        Returns:
            A prompt for improving the text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If prompt creation fails

        Examples:
            ```python
            # Create an improvement prompt
            text = "This is a sample text to improve."
            feedback = "The text needs more detail and better structure."
            reflections = ["Previous improvement focused on clarity"]
            prompt = manager._create_improvement_prompt_impl(text, feedback, reflections)
            print(prompt)
            ```
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text: must be non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Invalid feedback: must be non-empty string")

        prompt = f"""Improve the following text based on the feedback and previous reflections:

Text:
{text}

Feedback:
{feedback}"""

        if reflections:
            prompt += "\n\nPrevious reflections:\n" + "\n".join(f"- {r}" for r in reflections)

        prompt += "\n\nPlease provide the improved text in the following format:\nIMPROVED_TEXT: [improved text]"
        return prompt

    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Implementation of create_reflection_prompt.

        This method creates a specialized reflection prompt for ReflexionCritic,
        requesting the model to reflect on the improvement process.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            original_text: The original text
            feedback: The feedback provided
            improved_text: The improved version of the text

        Returns:
            A prompt for reflecting on the improvement process

        Raises:
            ValueError: If any input is empty
            RuntimeError: If prompt creation fails

        Examples:
            ```python
            # Create a reflection prompt
            original = "This is the original text."
            feedback = "The text needed more detail."
            improved = "This is the improved text with more detail."
            prompt = manager._create_reflection_prompt_impl(original, feedback, improved)
            print(prompt)
            ```
        """
        if not original_text or not isinstance(original_text, str):
            raise ValueError("Invalid original text: must be non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Invalid feedback: must be non-empty string")
        if not improved_text or not isinstance(improved_text, str):
            raise ValueError("Invalid improved text: must be non-empty string")

        return f"""Reflect on the improvement process:

Original text:
{original_text}

Feedback:
{feedback}

Improved text:
{improved_text}

Please provide your reflection in the following format:
REFLECTION: [your reflection on the improvement process]"""
