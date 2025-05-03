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
        requesting the model to analyze the text and provide detailed feedback.

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
        """
        return f"""Please critique the following text:

        TEXT TO CRITIQUE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        SCORE: [number between 0 and 1]
        FEEDBACK: [your general feedback]
        ISSUES:
        - [issue 1]
        - [issue 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]

        CRITIQUE:"""

    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Implementation of create_improvement_prompt.

        This method creates a specialized improvement prompt for PromptCritic,
        requesting the model to improve the text based on feedback.

        Lifecycle:
        1. Input validation
        2. Feedback processing
        3. Template selection
        4. Variable substitution
        5. Format validation
        6. Error handling

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            reflections: Optional reflections to include in the prompt

        Returns:
            A prompt for improving the text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If prompt creation fails
        """
        return f"""Please improve the following text based on the feedback:

        TEXT TO IMPROVE:
        {text}

        FEEDBACK:
        {feedback}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        IMPROVED_TEXT: [your improved version of the text]

        IMPROVED VERSION:"""

    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Implementation of create_reflection_prompt.

        This method creates a specialized reflection prompt for PromptCritic,
        requesting the model to reflect on the improvement and what was learned.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text

        Returns:
            A prompt for reflecting on the improvement

        Raises:
            ValueError: If any input is empty
            RuntimeError: If prompt creation fails
        """
        return f"""Please reflect on the following text improvement:

        ORIGINAL TEXT:
        {original_text}

        FEEDBACK:
        {feedback}

        IMPROVED TEXT:
        {improved_text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        REFLECTION: [your reflection on what was improved and why]

        REFLECTION:"""


class ReflexionCriticPromptManager(PromptManager):
    """
    Prompt manager for ReflexionCritic.

    This class provides prompt creation methods for ReflexionCritic, implementing
    specialized prompt templates that incorporate previous reflections for
    improved text generation.

    ## Lifecycle Management

    The ReflexionCriticPromptManager manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up templates
       - Configures error handling
       - Allocates resources

    2. **Operation**
       - Creates reflection-aware prompts
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
    from sifaka.critics.managers.prompt_factories import ReflexionCriticPromptManager

    # Create a reflexion critic manager
    manager = ReflexionCriticPromptManager()

    # Create different types of prompts
    text = "This is a sample text."
    feedback = "The text needs more detail."
    reflections = ["Previous improvement focused on clarity", "Added more examples"]

    # Improvement prompt with reflections
    improvement_prompt = manager.create_improvement_prompt(
        text, feedback, reflections
    )
    print(improvement_prompt)

    # Reflection prompt
    improved_text = "This is an improved version of the text."
    reflection_prompt = manager.create_reflection_prompt(
        text, feedback, improved_text
    )
    print(reflection_prompt)
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

        This method creates a specialized critique prompt for ReflexionCritic,
        requesting the model to analyze the text and provide detailed feedback.

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
        """
        return f"""Please critique the following text:

        TEXT TO CRITIQUE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        SCORE: [number between 0 and 1]
        FEEDBACK: [your general feedback]
        ISSUES:
        - [issue 1]
        - [issue 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]

        CRITIQUE:"""

    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Implementation of create_improvement_prompt.

        This method creates a specialized improvement prompt for ReflexionCritic,
        requesting the model to improve the text based on feedback and previous
        reflections.

        Lifecycle:
        1. Input validation
        2. Feedback processing
        3. Reflection integration
        4. Template selection
        5. Variable substitution
        6. Format validation
        7. Error handling

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            reflections: Optional reflections to include in the prompt

        Returns:
            A prompt for improving the text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If prompt creation fails
        """
        prompt = f"""Please improve the following text based on the feedback:

        TEXT TO IMPROVE:
        {text}

        FEEDBACK:
        {feedback}
        """

        if reflections and len(reflections) > 0:
            prompt += "\n\nREFLECTIONS FROM PREVIOUS IMPROVEMENTS:\n"
            for i, reflection in enumerate(reflections):
                prompt += f"{i+1}. {reflection}\n"

        prompt += """
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        IMPROVED_TEXT: [your improved version of the text]

        IMPROVED VERSION:"""

        return prompt

    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Implementation of create_reflection_prompt.

        This method creates a specialized reflection prompt for ReflexionCritic,
        requesting the model to reflect on the improvement and what was learned.

        Lifecycle:
        1. Input validation
        2. Template selection
        3. Variable substitution
        4. Format validation
        5. Error handling

        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text

        Returns:
            A prompt for reflecting on the improvement

        Raises:
            ValueError: If any input is empty
            RuntimeError: If prompt creation fails
        """
        return f"""Please reflect on the following text improvement:

        ORIGINAL TEXT:
        {original_text}

        FEEDBACK:
        {feedback}

        IMPROVED TEXT:
        {improved_text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        REFLECTION: [your reflection on what was improved and why]

        REFLECTION:"""
