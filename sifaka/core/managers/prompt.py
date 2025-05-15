"""
Prompt Manager Module

## Overview
This module provides the PromptManager class which handles prompt creation,
modification, and validation. It supports various prompt types including:
- Basic prompts with template-based generation
- Specialized prompt types (validation, critique, improvement, reflection)
- Prompt enhancement with feedback, history, context, and examples

## Components
1. **PromptManager**: Base class for prompt management
2. **BasePrompt**: Base class for all prompt templates
3. **DefaultPromptManager**: Default implementation for critics
4. **PromptCriticPromptManager**: Specialized implementation for prompt critics
5. **ReflexionCriticPromptManager**: Specialized implementation for reflexion critics

## Usage Examples
```python
from sifaka.core.managers.prompt import PromptManager, BasePrompt

# Create prompt manager
manager = PromptManager()

# Create basic prompt
prompt = manager.create_prompt("Write a story about a robot") if manager else ""

# Add feedback
prompt_with_feedback = manager.create_prompt_with_feedback(
    prompt,
    "Make the story more emotional"
) if manager else ""

# Add history
prompt_with_history = manager.create_prompt_with_history(
    prompt,
    ["Previous story about a sad robot", "Story about a happy robot"]
) if manager else ""

# Create complex prompt
complex_prompt = manager.create_prompt(
    "Write a story about a robot",
    feedback="Make it emotional",
    history=["Previous story"],
    context="Set in future",
    examples=["Example story"]
) if manager else ""
```

For critics usage:
```python
from sifaka.core.managers.prompt import DefaultPromptManager
from sifaka.critics.models import CriticConfig

# Create a prompt manager
config = CriticConfig(
    name="test_critic",
    description="Test critic for prompt management"
)
prompt_manager = DefaultPromptManager(config)

# Create a validation prompt
text = "This is a sample text to validate."
validation_prompt = prompt_manager.create_validation_prompt(text) if prompt_manager else ""

# Create a critique prompt
critique_prompt = prompt_manager.create_critique_prompt(text) if prompt_manager else ""

# Create an improvement prompt
feedback = "The text needs more detail."
improvement_prompt = prompt_manager.create_improvement_prompt(text, feedback) if prompt_manager else ""
```
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Union
import time

from pydantic import Field, ConfigDict, PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult
from sifaka.utils.state import StateManager, create_prompt_manager_state
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
OutputType = TypeVar("OutputType")


class PromptConfig(BaseConfig):
    """
    Configuration for prompt manager.

    This class defines the configuration parameters for prompt managers,
    including template format, history settings, and caching options.

    ## Architecture
    Extends BaseConfig to provide consistent configuration handling
    across all Sifaka components.

    Attributes:
        template_format (str): Format of prompt templates
        add_timestamps (bool): Whether to add timestamps to prompts
        max_history_items (int): Maximum number of history items to include
        max_examples (int): Maximum number of examples to include
        cache_size (int): Size of the prompt cache
    """

    template_format: str = Field(default="text", description="Format of prompt templates")
    add_timestamps: bool = Field(default=False, description="Whether to add timestamps to prompts")
    max_history_items: int = Field(
        default=5, description="Maximum number of history items to include"
    )
    max_examples: int = Field(default=3, description="Maximum number of examples to include")
    cache_size: int = Field(default=100, description="Size of the prompt cache")

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class PromptResult(BaseResult):
    """
    Result of prompt generation.

    This class represents the result of a prompt generation operation,
    including the generated prompt and the context used to generate it.

    ## Architecture
    Extends BaseResult to provide consistent result handling
    across all Sifaka components.

    Attributes:
        prompt (str): The generated prompt
        context (Dict[str, Any]): The context used to generate the prompt
    """

    prompt: str = Field(default="")
    context: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class BasePrompt(BaseComponent[Dict[str, Any], str]):
    """
    Base class for all prompts.

    This class provides the foundation for all prompt implementations,
    with template-based prompt generation capabilities.

    ## Architecture
    Extends BaseComponent to provide consistent behavior across
    all Sifaka components, with specialized prompt generation functionality.

    ## Lifecycle
    1. Initialization: Set up with template and configuration
    2. Generation: Generate prompts from context
    3. Processing: Process input to generate prompts

    Attributes:
        name (str): Name of the prompt
        description (str): Description of the prompt
    """

    def __init__(
        self,
        name: str,
        description: str,
        template: str,
        config: Optional[Optional[PromptConfig]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the prompt."""
        super().__init__(name, description, config or PromptConfig(**kwargs))
        self._state_manager.update("template", template)
        self._state_manager.update("initialized", True)

    def generate(self, context: Dict[str, Any]) -> str:
        """
        Generate a prompt from context.

        This method takes a context dictionary and applies it to the
        prompt template to generate a formatted prompt.

        Args:
            context (Dict[str, Any]): Dictionary of context values to apply to the template

        Returns:
            str: The generated prompt

        Raises:
            KeyError: If a required context key is missing
            Exception: If prompt generation fails for any other reason
        """
        template = self._state_manager.get("template") if self._state_manager else None
        try:
            return template.format(**context) if template else ""
        except KeyError as e:
            if logger:
                logger.error(f"Missing context key: {e}")
            return template if template else ""
        except Exception as e:
            if logger:
                logger.error(f"Error generating prompt: {e}")
            return template if template else ""

    def process(self, input: Dict[str, Any]) -> str:
        """
        Process the input context and return a prompt.

        This is the implementation of the abstract method from BaseComponent.

        Args:
            input (Dict[str, Any]): Dictionary of context values to apply to the template

        Returns:
            str: The generated prompt
        """
        return self.generate(input) if self else ""


class PromptManager(BaseComponent[Dict[str, Any], PromptResult]):
    """
    Prompt manager for Sifaka.

    This class provides template-based prompt generation and management,
    coordinating between multiple prompts and tracking generation results.

    ## Architecture
    The PromptManager follows a component-based architecture:
    - Inherits from BaseComponent for consistent behavior
    - Uses StateManager for state management
    - Implements caching for performance
    - Tracks statistics for monitoring

    ## Lifecycle
    1. Initialization: Set up with prompts and configuration
    2. Prompt Generation: Generate prompts from context
    3. Prompt Management: Add/remove prompts as needed
    4. Statistics: Track prompt generation performance
    """

    def __init__(
        self,
        prompts: Optional[List[BasePrompt]] = None,
        name: str = "prompt_manager",
        description: str = "Prompt manager for Sifaka",
        template_format: str = "text",
        add_timestamps: bool = False,
        max_history_items: int = 5,
        max_examples: int = 3,
        config: Optional[Optional[PromptConfig]] = None,
        **kwargs: Any,
    ):
        """Initialize the prompt manager."""
        # Create config if not provided
        if config is None:
            config = PromptConfig(
                name=name,
                description=description,
                template_format=template_format,
                add_timestamps=add_timestamps,
                max_history_items=max_history_items,
                max_examples=max_examples,
                **kwargs,
            )

        # Initialize base component
        super().__init__(name, description, config)

        # Store prompts in state
        if self._state_manager:
            self._state_manager.update("prompts", prompts or [])
            self._state_manager.update("result_cache", {})
            self._state_manager.update("initialized", True)

            # Set metadata
            self._state_manager.set_metadata("component_type", "prompt_manager")
            self._state_manager.set_metadata("creation_time", time.time())
            self._state_manager.set_metadata("prompt_count", len(prompts or []))

    def process(self, input: Dict[str, Any]) -> PromptResult:
        """Process the input context and return a prompt result."""
        # For process method, we'll use the first prompt or create a simple one
        prompts = self._state_manager.get("prompts", []) if self._state_manager else []
        if not prompts:
            return PromptResult(
                passed=False,
                message="No prompts available",
                metadata={"error_type": "no_prompts"},
                score=0.0,
                issues=["No prompts available"],
                suggestions=["Add prompts to the manager"],
                prompt="",
                context=input,
            )

        # Use the first prompt
        prompt = prompts[0]
        generated_prompt = prompt.generate(input) if prompt else ""

        return PromptResult(
            passed=True,
            message="Prompt generated successfully",
            metadata={"prompt_name": prompt.name},
            score=1.0,
            prompt=generated_prompt,
            context=input,
        )

    def add_prompt(self, prompt: BasePrompt) -> None:
        """Add a prompt to the manager."""
        # Validate prompt type
        if not isinstance(prompt, BasePrompt):
            raise ValueError(f"Expected BasePrompt instance, got {type(prompt)}")

        # Check for duplicate prompt names
        prompts = self._state_manager.get("prompts", []) if self._state_manager else []
        if any(p.name == prompt.name for p in prompts):
            if logger:
                logger.warning(
                    f"Prompt with name '{prompt.name}' already exists, it will be replaced"
                )
            # Remove existing prompt with same name
            if self:
                self.remove_prompt(prompt.name)
            # Get updated prompts list
            prompts = self._state_manager.get("prompts", []) if self._state_manager else []

        # Add prompt to the list
        if prompts is not None:
            prompts.append(prompt)
        if self._state_manager:
            self._state_manager.update("prompts", prompts)

        # Update metadata
        if self._state_manager:
            self._state_manager.set_metadata("prompt_count", len(prompts))

        if logger:
            logger.debug(f"Added prompt '{prompt.name}' to prompt manager '{self.name}'")

    def remove_prompt(self, prompt_name: str) -> None:
        """Remove a prompt by name."""
        # Validate input
        if not prompt_name or not isinstance(prompt_name, str):
            raise ValueError(f"Invalid prompt name: {prompt_name}")

        # Find prompt by name
        prompt_to_remove = None
        prompts = self._state_manager.get("prompts", []) if self._state_manager else []
        for prompt in prompts:
            if prompt.name == prompt_name:
                prompt_to_remove = prompt
                break

        if prompt_to_remove is None:
            raise ValueError(f"Prompt not found: {prompt_name}")

        # Remove prompt from list
        if prompts is not None:
            prompts.remove(prompt_to_remove)
        if self._state_manager:
            self._state_manager.update("prompts", prompts)

        # Update metadata
        if self._state_manager:
            self._state_manager.set_metadata("prompt_count", len(prompts))

        if logger:
            logger.debug(f"Removed prompt '{prompt_name}' from prompt manager '{self.name}'")

    def get_prompts(self) -> List[BasePrompt]:
        """Get all registered prompts."""
        return self._state_manager.get("prompts", []) if self._state_manager else []

    def create_prompt(self, input_value: str, **kwargs: Any) -> str:
        """Create a prompt from input value and optional parameters."""
        if not isinstance(input_value, str):
            raise ValueError(f"Expected string input, got {type(input_value)}")

        # Start with the input value
        prompt = input_value

        # Add feedback if provided
        if "feedback" in kwargs and kwargs["feedback"]:
            if self:
                prompt = self.create_prompt_with_feedback(prompt, kwargs["feedback"])

        # Add history if provided
        if "history" in kwargs and kwargs["history"]:
            if self:
                prompt = self.create_prompt_with_history(prompt, kwargs["history"])

        # Add context if provided
        if "context" in kwargs and kwargs["context"]:
            if self:
                prompt = self.create_prompt_with_context(prompt, kwargs["context"])

        # Add examples if provided
        if "examples" in kwargs and kwargs["examples"]:
            if self:
                prompt = self.create_prompt_with_examples(prompt, kwargs["examples"])

        return prompt

    def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
        """Create a prompt with feedback."""
        return f"{original_prompt}\n\nFeedback: {feedback}"

    def create_prompt_with_history(self, original_prompt: str, history: List[str]) -> str:
        """Create a prompt with history."""
        history_text = "\n".join(history)
        return f"{original_prompt}\n\nPrevious attempts:\n{history_text}"

    def create_prompt_with_context(self, original_prompt: str, context: str) -> str:
        """Create a prompt with context."""
        return f"{original_prompt}\n\nContext: {context}"

    def create_prompt_with_examples(self, original_prompt: str, examples: List[str]) -> str:
        """Create a prompt with examples."""
        examples_text = "\n".join([f"- {example}" for example in examples])
        return f"{original_prompt}\n\nExamples:\n{examples_text}"

    def format_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Format a prompt with optional formatting parameters."""
        # Default formatting
        formatted = prompt

        # Apply custom formatting based on kwargs
        if kwargs and kwargs.get("line_breaks", True):
            formatted = formatted.replace(". ", ".\n") if formatted else ""
        if kwargs and kwargs.get("indent"):
            indent = " " * kwargs["indent"]
            formatted = (
                "\n".join(indent + line for line in formatted.split("\n")) if formatted else ""
            )

        return formatted

    def validate_prompt(self, prompt: str) -> bool:
        """Validate a prompt."""
        # Basic validation
        if not prompt or not isinstance(prompt, str):
            return False

        # Add more validation as needed
        return True


class CriticPromptManager(ABC):
    """
    Abstract base class for critic prompt managers.

    This class defines the interface for prompt managers used by critics,
    providing methods for creating validation, critique, improvement,
    and reflection prompts.

    ## Architecture
    Uses Python's ABC (Abstract Base Class) to define an interface
    that all critic prompt managers must implement.

    ## Lifecycle
    1. Initialization: Set up with configuration
    2. Prompt Creation: Create specialized prompts for different critic operations
    3. Usage: Used by critics to generate prompts for validation, critique, etc.

    ## Error Handling
    Implementations should handle:
    - Invalid input validation
    - Template formatting errors
    - Configuration errors
    """

    @abstractmethod
    def create_validation_prompt(self, text: str) -> str:
        """
        Create a prompt for text validation.

        Args:
            text (str): The text to validate

        Returns:
            str: A formatted prompt for text validation

        Raises:
            ValueError: If text is invalid
        """
        pass

    @abstractmethod
    def create_critique_prompt(self, text: str) -> str:
        """
        Create a prompt for text critique.

        Args:
            text (str): The text to critique

        Returns:
            str: A formatted prompt for text critique

        Raises:
            ValueError: If text is invalid
        """
        pass

    @abstractmethod
    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: Optional[Optional[List[str]]] = None
    ) -> str:
        """
        Create a prompt for text improvement.

        Args:
            text (str): The text to improve
            feedback (str): Feedback to guide the improvement
            reflections (Optional[List[str]], optional): Previous reflections. Defaults to None.

        Returns:
            str: A formatted prompt for text improvement

        Raises:
            ValueError: If text or feedback is invalid
        """
        pass

    @abstractmethod
    def create_reflection_prompt(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Create a prompt for reflection on text improvement.

        Args:
            original_text (str): The original text
            feedback (str): The feedback provided
            improved_text (str): The improved text

        Returns:
            str: A formatted prompt for reflection

        Raises:
            ValueError: If any input is invalid
        """
        pass


class DefaultPromptManager(CriticPromptManager):
    """
    Default implementation of prompt manager for critics.

    This class provides concrete implementations of the methods defined
    in the CriticPromptManager interface, using standard templates for
    validation, critique, improvement, and reflection prompts.

    ## Architecture
    Implements the CriticPromptManager interface with standard templates
    for different critic operations.

    ## Lifecycle
    1. Initialization: Set up with configuration
    2. Prompt Creation: Create specialized prompts using standard templates
    3. Usage: Used by critics to generate prompts for validation, critique, etc.

    ## Error Handling
    Handles:
    - Invalid input validation
    - Template formatting errors
    - Configuration errors

    ## Examples
    ```python
    # Create a default prompt manager
    manager = DefaultPromptManager()

    # Create a validation prompt
    validation_prompt = manager.create_validation_prompt("Text to validate") if manager else ""

    # Create a critique prompt
    critique_prompt = manager.create_critique_prompt("Text to critique") if manager else ""
    ```
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the prompt manager."""
        self.config = config
        self._state_manager = create_prompt_manager_state()
        if self._state_manager:
            self._state_manager.update("initialized", True)

    def create_validation_prompt(self, text: str) -> str:
        """Create a prompt for text validation."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        return f"""Please validate the following text:

TEXT TO VALIDATE:
{text}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
VALID: [true/false]
REASON: [reason for validation result]

VALIDATION:"""

    def create_critique_prompt(self, text: str) -> str:
        """Create a prompt for text critique."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        return f"""Please critique the following text:

TEXT TO CRITIQUE:
{text}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
SCORE: [0-100]
STRENGTHS: [list of strengths]
WEAKNESSES: [list of weaknesses]
SUGGESTIONS: [list of suggestions for improvement]

CRITIQUE:"""

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: Optional[Optional[List[str]]] = None
    ) -> str:
        """Create a prompt for text improvement."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")

        reflection_text = ""
        if reflections and len(reflections) > 0:
            reflection_text = "\n\nPREVIOUS REFLECTIONS:\n"
            for i, reflection in enumerate(reflections):
                reflection_text += f"{i+1}. {reflection}\n"

        return f"""Please improve the following text:

TEXT TO IMPROVE:
{text}

FEEDBACK:
{feedback}{reflection_text}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
IMPROVED_TEXT: [improved text]

IMPROVEMENT:"""

    def create_reflection_prompt(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """Create a prompt for reflection on text improvement."""
        if not original_text or not isinstance(original_text, str):
            raise ValueError("Original text must be a non-empty string")

        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")

        if not improved_text or not isinstance(improved_text, str):
            raise ValueError("Improved text must be a non-empty string")

        return f"""Please reflect on the following text improvement:

ORIGINAL TEXT:
{original_text}

FEEDBACK:
{feedback}

IMPROVED TEXT:
{improved_text}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
REFLECTION: [reflection on the improvement process]
LESSONS: [lessons learned for future improvements]

REFLECTION:"""


class PromptCriticPromptManager(DefaultPromptManager):
    """
    Prompt manager for PromptCritic.

    This class extends DefaultPromptManager with specialized prompt templates
    for the PromptCritic implementation.
    """

    def create_validation_prompt(self, text: str) -> str:
        """Create a prompt for text validation."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        return f"""You are a text validator. Please validate the following text:

TEXT TO VALIDATE:
{text}

Consider the following aspects:
1. Grammar and spelling
2. Clarity and coherence
3. Style and tone
4. Overall quality

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
VALID: [true/false]
REASON: [detailed reason for validation result]
SCORE: [0-100]

VALIDATION:"""


class ReflexionCriticPromptManager(DefaultPromptManager):
    """
    Prompt manager for ReflexionCritic.

    This class extends DefaultPromptManager with specialized prompt templates
    that incorporate memory and reflections for the ReflexionCritic implementation.
    """

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: Optional[Optional[List[str]]] = None
    ) -> str:
        """Create a prompt for text improvement with reflections."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")

        reflection_text = ""
        if reflections and len(reflections) > 0:
            reflection_text = "\n\nREFLECTIONS ON PREVIOUS ATTEMPTS:\n"
            for i, reflection in enumerate(reflections):
                reflection_text += f"{i+1}. {reflection}\n"

        return f"""You are an expert text improver with the ability to learn from past attempts.

TEXT TO IMPROVE:
{text}

FEEDBACK:
{feedback}{reflection_text}

Use the reflections to guide your improvement process. Learn from past mistakes and build on successful strategies.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
IMPROVED_TEXT: [improved text]
REASONING: [explanation of your improvement approach]

IMPROVEMENT:"""


def create_prompt_manager(
    prompts: Optional[List[BasePrompt]] = None,
    name: str = "prompt_manager",
    description: str = "Prompt manager for Sifaka",
    template_format: str = "text",
    add_timestamps: bool = False,
    max_history_items: int = 5,
    max_examples: int = 3,
    cache_size: int = 100,
    **kwargs: Any,
) -> PromptManager:
    """
    Create a prompt manager.

    This factory function creates and configures a PromptManager
    with the specified parameters, providing a convenient way to create
    prompt managers with default or custom configurations.

    Args:
        prompts (List[BasePrompt], optional): List of prompts to manage. Defaults to None.
        name (str, optional): Name of the manager. Defaults to "prompt_manager".
        description (str, optional): Description of the manager. Defaults to "Prompt manager for Sifaka".
        template_format (str, optional): Format of prompt templates. Defaults to "text".
        add_timestamps (bool, optional): Whether to add timestamps to prompts. Defaults to False.
        max_history_items (int, optional): Maximum number of history items to include. Defaults to 5.
        max_examples (int, optional): Maximum number of examples to include. Defaults to 3.
        cache_size (int, optional): Size of the prompt cache. Defaults to 100.
        **kwargs (Any): Additional configuration parameters

    Returns:
        PromptManager: Configured PromptManager instance

    Example:
        ```python
        # Create a prompt manager with default settings
        prompt_manager = create_prompt_manager()

        # Create a prompt manager with custom settings
        custom_manager = create_prompt_manager(
            name="custom_prompts",
            template_format="markdown",
            max_history_items=10
        )
        ```
    """
    config = PromptConfig(
        name=name,
        description=description,
        template_format=template_format,
        add_timestamps=add_timestamps,
        max_history_items=max_history_items,
        max_examples=max_examples,
        cache_size=cache_size,
        **kwargs,
    )

    return PromptManager(
        prompts=prompts or [],
        name=name,
        description=description,
        config=config,
    )
