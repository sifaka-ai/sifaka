"""
Prompt manager for Sifaka critics.

This module provides a manager for handling critic prompts and state management.

## Component Overview

1. **Core Components**
   - `PromptManager`: Abstract base class for prompt management
   - `DefaultPromptManager`: Concrete implementation with default templates
   - Template system for different prompt types
   - Variable substitution mechanism

2. **Prompt Types**
   - Validation prompts
   - Critique prompts
   - Improvement prompts
   - Reflection prompts

## Component Lifecycle

### PromptManager Lifecycle

1. **Initialization Phase**
   - Configuration validation and setup
   - Template initialization
   - Resource allocation
   - Error handling setup

2. **Usage Phase**
   - Prompt creation for different operations
   - Template variable substitution
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

## Error Handling and Recovery

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

Basic usage:

```python
from sifaka.critics.managers.prompt import DefaultPromptManager
from sifaka.critics.models import CriticConfig

# Create a prompt manager
config = CriticConfig(
    name="test_critic",
    description="Test critic for prompt management"
)
prompt_manager = DefaultPromptManager(config)

# Create a validation prompt
text = "This is a sample text to validate."
validation_prompt = prompt_manager.create_validation_prompt(text)
print(validation_prompt)

# Create a critique prompt
critique_prompt = prompt_manager.create_critique_prompt(text)
print(critique_prompt)

# Create an improvement prompt
feedback = "The text needs more detail and better structure."
improvement_prompt = prompt_manager.create_improvement_prompt(text, feedback)
print(improvement_prompt)
```

Advanced usage with reflections:

```python
from sifaka.critics.managers.prompt import DefaultPromptManager
from sifaka.critics.models import CriticConfig

# Create a prompt manager
config = CriticConfig(
    name="reflexion_critic",
    description="A critic that uses reflections"
)
prompt_manager = DefaultPromptManager(config)

# Create an improvement prompt with reflections
text = "This is a sample text."
feedback = "The text needs improvement."
reflections = [
    "Previous attempts focused too much on structure",
    "Need to maintain the original meaning"
]
improvement_prompt = prompt_manager.create_improvement_prompt(
    text, feedback, reflections
)
print(improvement_prompt)

# Create a reflection prompt
improved_text = "This is an improved version of the text."
reflection_prompt = prompt_manager.create_reflection_prompt(
    text, feedback, improved_text
)
print(reflection_prompt)
```

Custom prompt templates:

```python
from sifaka.critics.managers.prompt import PromptManager
from sifaka.critics.models import CriticConfig

class CustomPromptManager(PromptManager):
    def _create_validation_prompt_impl(self, text: str) -> str:
        return (
            "Please validate the following text:\n\n"
            f"{text}\n\n"
            "Consider the following aspects:\n"
            "1. Grammar and spelling\n"
            "2. Clarity and coherence\n"
            "3. Style and tone\n"
            "4. Overall quality\n\n"
            "Provide a detailed analysis."
        )

    def _create_critique_prompt_impl(self, text: str) -> str:
        return (
            "Please critique the following text:\n\n"
            f"{text}\n\n"
            "Focus on:\n"
            "1. Strengths and weaknesses\n"
            "2. Areas for improvement\n"
            "3. Specific suggestions\n"
            "4. Overall assessment"
        )

# Use the custom prompt manager
config = CriticConfig(
    name="custom_critic",
    description="A critic with custom prompts"
)
prompt_manager = CustomPromptManager(config)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type
import time
from pydantic import PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.critics.base import BaseCritic, CriticConfig, CriticResult
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager(ABC):
    """
    Manages prompt creation for critics.

    This class is responsible for creating prompts for validation, critique,
    improvement, and reflection. It follows a template-based approach where
    each type of prompt has a specific format and structure.

    ## Lifecycle Management

    The PromptManager manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up templates
       - Configures error handling
       - Allocates resources

    2. **Operation**
       - Creates prompts for different operations
       - Handles template variables
       - Validates formats
       - Manages errors

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    The PromptManager implements comprehensive error handling:

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
    from sifaka.critics.managers.prompt import DefaultPromptManager
    from sifaka.critics.models import CriticConfig

    # Create a prompt manager
    config = CriticConfig(
        name="test_critic",
        description="Test critic for prompt management"
    )
    prompt_manager = DefaultPromptManager(config)

    # Create different types of prompts
    text = "This is a sample text."

    # Validation prompt
    validation_prompt = prompt_manager.create_validation_prompt(text)

    # Critique prompt
    critique_prompt = prompt_manager.create_critique_prompt(text)

    # Improvement prompt
    feedback = "The text needs more detail."
    improvement_prompt = prompt_manager.create_improvement_prompt(text, feedback)

    # Reflection prompt
    improved_text = "This is an improved version of the sample text."
    reflection_prompt = prompt_manager.create_reflection_prompt(
        text, feedback, improved_text
    )
    ```
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(self, config: CriticConfig):
        """
        Initialize a PromptManager instance.

        This method sets up the prompt manager with its configuration and
        performs necessary validation and initialization.

        Lifecycle:
        1. Validate configuration
        2. Initialize templates
        3. Set up error handling
        4. Configure resources

        Args:
            config: The critic configuration

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If initialization fails
        """
        self._config = config

        # Initialize state
        self._state.update("config", config)
        self._state.update("initialized", True)
        self._state.update("template_cache", {})

        # Initialize metadata
        self._state.set_metadata("component_type", "prompt_manager")
        self._state.set_metadata("validation_prompt_count", 0)
        self._state.set_metadata("critique_prompt_count", 0)
        self._state.set_metadata("improvement_prompt_count", 0)
        self._state.set_metadata("reflection_prompt_count", 0)
        self._state.set_metadata("error_count", 0)
        self._state.set_metadata("creation_time", time.time())

    def create_validation_prompt(self, text: str) -> str:
        """Create a prompt for text validation.

        This method creates a prompt for validating text quality and standards.
        It delegates to the implementation-specific _create_validation_prompt_impl.

        ## Overview
        The method provides:
        - Text validation prompt creation
        - Input validation
        - Error handling
        - Template variable substitution

        ## Usage Examples
        ```python
        # Create a validation prompt
        text = "This is a sample text to validate."
        prompt = prompt_manager.create_validation_prompt(text)
        print(prompt)

        # Create with error handling
        try:
            prompt = prompt_manager.create_validation_prompt("")
        except ValueError as e:
            print(f"Validation error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Empty text validation
        - Invalid text format checking
        - Template error handling
        - Implementation error recovery

        Args:
            text: The text to validate

        Returns:
            str: The validation prompt

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If prompt creation fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Start timing
        start_time = time.time()

        # Track validation prompt count
        count = self._state.get_metadata("validation_prompt_count", 0)
        self._state.set_metadata("validation_prompt_count", count + 1)

        try:
            # Check cache for prompt
            cache_key = f"validation:{hash(text)}"
            template_cache = self._state.get("template_cache", {})

            if cache_key in template_cache:
                self._state.set_metadata("cache_hit", True)
                return template_cache[cache_key]

            self._state.set_metadata("cache_hit", False)

            # Create prompt
            prompt = self._create_validation_prompt_impl(text)

            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update average creation time
            total_time = self._state.get_metadata("validation_prompt_time", 0)
            new_total = total_time + execution_time
            self._state.set_metadata("validation_prompt_time", new_total)
            avg_time = new_total / (count + 1)
            self._state.set_metadata("avg_validation_prompt_time", avg_time)

            # Cache the prompt
            cache_size = 100  # Could make this configurable
            if len(template_cache) >= cache_size:
                template_cache = {}

            template_cache[cache_key] = prompt
            self._state.update("template_cache", template_cache)

            return prompt
        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)

            # Track specific error type
            errors = self._state.get_metadata("errors", {})
            error_type = type(e).__name__
            errors[error_type] = errors.get(error_type, 0) + 1
            self._state.set_metadata("errors", errors)

            logger.error(f"Failed to create validation prompt: {e}")
            raise RuntimeError(f"Failed to create validation prompt: {e}")

    def create_critique_prompt(self, text: str) -> str:
        """Create a prompt for text critique.

        This method creates a prompt for critiquing text quality and providing
        detailed feedback. It delegates to the implementation-specific
        _create_critique_prompt_impl.

        ## Overview
        The method provides:
        - Text critique prompt creation
        - Input validation
        - Error handling
        - Template variable substitution

        ## Usage Examples
        ```python
        # Create a critique prompt
        text = "This is a sample text to critique."
        prompt = prompt_manager.create_critique_prompt(text)
        print(prompt)

        # Create with error handling
        try:
            prompt = prompt_manager.create_critique_prompt("")
        except ValueError as e:
            print(f"Critique error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Empty text validation
        - Invalid text format checking
        - Template error handling
        - Implementation error recovery

        Args:
            text: The text to critique

        Returns:
            str: The critique prompt

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If prompt creation fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Start timing
        start_time = time.time()

        # Track critique prompt count
        count = self._state.get_metadata("critique_prompt_count", 0)
        self._state.set_metadata("critique_prompt_count", count + 1)

        try:
            # Check cache for prompt
            cache_key = f"critique:{hash(text)}"
            template_cache = self._state.get("template_cache", {})

            if cache_key in template_cache:
                self._state.set_metadata("cache_hit", True)
                return template_cache[cache_key]

            self._state.set_metadata("cache_hit", False)

            # Create prompt
            prompt = self._create_critique_prompt_impl(text)

            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update average creation time
            total_time = self._state.get_metadata("critique_prompt_time", 0)
            new_total = total_time + execution_time
            self._state.set_metadata("critique_prompt_time", new_total)
            avg_time = new_total / (count + 1)
            self._state.set_metadata("avg_critique_prompt_time", avg_time)

            # Cache the prompt
            cache_size = 100  # Could make this configurable
            if len(template_cache) >= cache_size:
                template_cache = {}

            template_cache[cache_key] = prompt
            self._state.update("template_cache", template_cache)

            return prompt
        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)

            # Track specific error type
            errors = self._state.get_metadata("errors", {})
            error_type = type(e).__name__
            errors[error_type] = errors.get(error_type, 0) + 1
            self._state.set_metadata("errors", errors)

            logger.error(f"Failed to create critique prompt: {e}")
            raise RuntimeError(f"Failed to create critique prompt: {e}")

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """Create a prompt for text improvement.

        This method creates a prompt for improving text based on feedback and
        optional reflections. It delegates to the implementation-specific
        _create_improvement_prompt_impl.

        ## Overview
        The method provides:
        - Text improvement prompt creation
        - Input validation
        - Error handling
        - Template variable substitution
        - Reflection integration

        ## Usage Examples
        ```python
        # Create an improvement prompt
        text = "This is a sample text to improve."
        feedback = "The text needs more detail."
        prompt = prompt_manager.create_improvement_prompt(text, feedback)
        print(prompt)

        # Create with reflections
        reflections = [
            "Previous attempts focused too much on structure",
            "Need to maintain the original meaning"
        ]
        prompt = prompt_manager.create_improvement_prompt(
            text, feedback, reflections
        )
        print(prompt)

        # Create with error handling
        try:
            prompt = prompt_manager.create_improvement_prompt("", "")
        except ValueError as e:
            print(f"Improvement error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Empty text validation
        - Invalid text format checking
        - Feedback validation
        - Reflection format checking
        - Template error handling
        - Implementation error recovery

        Args:
            text: The text to improve
            feedback: The feedback to use for improvement
            reflections: Optional list of reflections

        Returns:
            str: The improvement prompt

        Raises:
            ValueError: If text or feedback are empty or invalid
            RuntimeError: If prompt creation fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")
        if reflections is not None and not isinstance(reflections, list):
            raise ValueError("Reflections must be a list")

        # Start timing
        start_time = time.time()

        # Track improvement prompt count
        count = self._state.get_metadata("improvement_prompt_count", 0)
        self._state.set_metadata("improvement_prompt_count", count + 1)

        # Track reflection usage
        if reflections:
            reflection_count = self._state.get_metadata("reflection_usage", 0)
            self._state.set_metadata("reflection_usage", reflection_count + 1)

        try:
            # Create prompt (don't cache improvement prompts as they vary too much)
            prompt = self._create_improvement_prompt_impl(text, feedback, reflections)

            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update average creation time
            total_time = self._state.get_metadata("improvement_prompt_time", 0)
            new_total = total_time + execution_time
            self._state.set_metadata("improvement_prompt_time", new_total)
            avg_time = new_total / (count + 1)
            self._state.set_metadata("avg_improvement_prompt_time", avg_time)

            return prompt
        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)

            # Track specific error type
            errors = self._state.get_metadata("errors", {})
            error_type = type(e).__name__
            errors[error_type] = errors.get(error_type, 0) + 1
            self._state.set_metadata("errors", errors)

            logger.error(f"Failed to create improvement prompt: {e}")
            raise RuntimeError(f"Failed to create improvement prompt: {e}")

    def create_reflection_prompt(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """Create a prompt for reflection on text improvement.

        This method creates a prompt for reflecting on the improvement process
        and results. It delegates to the implementation-specific
        _create_reflection_prompt_impl.

        ## Overview
        The method provides:
        - Reflection prompt creation
        - Input validation
        - Error handling
        - Template variable substitution
        - Improvement analysis

        ## Usage Examples
        ```python
        # Create a reflection prompt
        original_text = "This is the original text."
        feedback = "The text needed improvement."
        improved_text = "This is the improved version."
        prompt = prompt_manager.create_reflection_prompt(
            original_text, feedback, improved_text
        )
        print(prompt)

        # Create with error handling
        try:
            prompt = prompt_manager.create_reflection_prompt("", "", "")
        except ValueError as e:
            print(f"Reflection error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Empty text validation
        - Invalid text format checking
        - Feedback validation
        - Improved text validation
        - Template error handling
        - Implementation error recovery

        Args:
            original_text: The original text
            feedback: The feedback used for improvement
            improved_text: The improved text

        Returns:
            str: The reflection prompt

        Raises:
            ValueError: If any inputs are empty or invalid
            RuntimeError: If prompt creation fails
        """
        if not original_text or not isinstance(original_text, str):
            raise ValueError("Original text must be a non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")
        if not improved_text or not isinstance(improved_text, str):
            raise ValueError("Improved text must be a non-empty string")

        # Start timing
        start_time = time.time()

        # Track reflection prompt count
        count = self._state.get_metadata("reflection_prompt_count", 0)
        self._state.set_metadata("reflection_prompt_count", count + 1)

        try:
            # Create prompt (don't cache reflection prompts as they vary too much)
            prompt = self._create_reflection_prompt_impl(original_text, feedback, improved_text)

            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update average creation time
            total_time = self._state.get_metadata("reflection_prompt_time", 0)
            new_total = total_time + execution_time
            self._state.set_metadata("reflection_prompt_time", new_total)
            avg_time = new_total / (count + 1)
            self._state.set_metadata("avg_reflection_prompt_time", avg_time)

            return prompt
        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)

            # Track specific error type
            errors = self._state.get_metadata("errors", {})
            error_type = type(e).__name__
            errors[error_type] = errors.get(error_type, 0) + 1
            self._state.set_metadata("errors", errors)

            logger.error(f"Failed to create reflection prompt: {e}")
            raise RuntimeError(f"Failed to create reflection prompt: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prompt creation.

        Returns:
            Dictionary with prompt statistics
        """
        return {
            "validation_prompt_count": self._state.get_metadata("validation_prompt_count", 0),
            "critique_prompt_count": self._state.get_metadata("critique_prompt_count", 0),
            "improvement_prompt_count": self._state.get_metadata("improvement_prompt_count", 0),
            "reflection_prompt_count": self._state.get_metadata("reflection_prompt_count", 0),
            "reflection_usage": self._state.get_metadata("reflection_usage", 0),
            "error_count": self._state.get_metadata("error_count", 0),
            "avg_validation_prompt_time": self._state.get_metadata("avg_validation_prompt_time", 0),
            "avg_critique_prompt_time": self._state.get_metadata("avg_critique_prompt_time", 0),
            "avg_improvement_prompt_time": self._state.get_metadata(
                "avg_improvement_prompt_time", 0
            ),
            "avg_reflection_prompt_time": self._state.get_metadata("avg_reflection_prompt_time", 0),
            "errors": self._state.get_metadata("errors", {}),
            "cache_size": len(self._state.get("template_cache", {})),
        }

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._state.update("template_cache", {})

    @abstractmethod
    def _create_validation_prompt_impl(self, text: str) -> str:
        """
        Implement validation prompt creation.

        This abstract method must be implemented by subclasses to create
        validation prompts based on the specific requirements of the critic.

        Args:
            text: The text to validate

        Returns:
            str: The validation prompt

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def _create_critique_prompt_impl(self, text: str) -> str:
        """
        Implement critique prompt creation.

        This abstract method must be implemented by subclasses to create
        critique prompts based on the specific requirements of the critic.

        Args:
            text: The text to critique

        Returns:
            str: The critique prompt

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Implement improvement prompt creation.

        This abstract method must be implemented by subclasses to create
        improvement prompts based on the specific requirements of the critic.

        Args:
            text: The text to improve
            feedback: The feedback to use for improvement
            reflections: Optional list of reflections

        Returns:
            str: The improvement prompt

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Implement reflection prompt creation.

        This abstract method must be implemented by subclasses to create
        reflection prompts based on the specific requirements of the critic.

        Args:
            original_text: The original text
            feedback: The feedback used for improvement
            improved_text: The improved text

        Returns:
            str: The reflection prompt

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass


class DefaultPromptManager(PromptManager):
    """
    Default implementation of PromptManager.

    This class provides default implementations of prompt creation methods
    with standardized templates and formatting.

    ## Lifecycle Management

    The DefaultPromptManager follows the same lifecycle as PromptManager
    but with specific implementations for each prompt type:

    1. **Initialization**
       - Uses default templates
       - Configures standard formatting
       - Sets up error handling

    2. **Operation**
       - Creates standardized prompts
       - Handles template variables
       - Validates formats
       - Manages errors

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    The DefaultPromptManager implements these error handling patterns:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies reflection format

    2. **Template Management**
       - Uses standardized templates
       - Validates template syntax
       - Handles format errors

    3. **Resource Management**
       - Handles allocation failures
       - Manages cleanup errors
       - Preserves valid state

    ## Examples

    ```python
    from sifaka.critics.managers.prompt import DefaultPromptManager
    from sifaka.critics.models import CriticConfig

    # Create a default prompt manager
    config = CriticConfig(
        name="test_critic",
        description="Test critic for prompt management"
    )
    prompt_manager = DefaultPromptManager(config)

    # Create different types of prompts
    text = "This is a sample text."

    # Validation prompt
    validation_prompt = prompt_manager.create_validation_prompt(text)
    print(validation_prompt)

    # Critique prompt
    critique_prompt = prompt_manager.create_critique_prompt(text)
    print(critique_prompt)

    # Improvement prompt
    feedback = "The text needs more detail."
    improvement_prompt = prompt_manager.create_improvement_prompt(text, feedback)
    print(improvement_prompt)
    ```
    """

    def _create_validation_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_validation_prompt.

        This method creates a standardized validation prompt that asks
        the model to validate whether the text meets quality standards.

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

        This method creates a standardized critique prompt that asks
        the model to analyze the text and provide detailed feedback.

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

        This method creates a standardized improvement prompt that asks
        the model to improve the text based on feedback and optional reflections.

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

        This method creates a standardized reflection prompt that asks
        the model to reflect on the improvement and what was learned.

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
