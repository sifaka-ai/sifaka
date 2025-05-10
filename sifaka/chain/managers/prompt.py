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

from typing import Any, Dict, List, Optional, TypeVar
import time

from pydantic import Field, ConfigDict

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
OutputType = TypeVar("OutputType")


class PromptConfig(BaseConfig):
    """Configuration for prompt manager."""

    template_format: str = Field(
        default="text", description="Format of prompt templates (text, json, etc.)"
    )
    add_timestamps: bool = Field(default=False, description="Whether to add timestamps to prompts")
    max_history_items: int = Field(
        default=5, description="Maximum number of history items to include"
    )
    max_examples: int = Field(default=3, description="Maximum number of examples to include")

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class BasePrompt(BaseComponent[Dict[str, Any], str]):
    """Base class for all prompts."""

    def __init__(
        self,
        name: str,
        description: str,
        template: str,
        config: Optional[PromptConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the prompt."""
        super().__init__(name, description, config or PromptConfig(**kwargs))
        self._state.update("template", template)
        self._state.update("initialized", True)

    def generate(self, context: Dict[str, Any]) -> str:
        """Generate a prompt from context."""
        template = self._state.get("template")
        try:
            return template.format(**context)
        except KeyError as e:
            logger.error(f"Missing context key: {e}")
            return template
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return template


class PromptResult(BaseResult):
    """Result of prompt generation operation."""

    prompt: str = Field(default="")
    context: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class PromptManager(BaseComponent[Dict[str, Any], PromptResult]):
    """
    Prompt manager for Sifaka chains.

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
        prompts: List[BasePrompt],
        name: str = "prompt_manager",
        description: str = "Prompt manager for Sifaka chains",
        template_format: str = "text",
        add_timestamps: bool = False,
        max_history_items: int = 5,
        max_examples: int = 3,
        config: Optional[PromptConfig] = None,
        **kwargs: Any,
    ):
        """Initialize the prompt manager.

        Args:
            prompts: List of prompts to use for generation
            name: Name of the manager
            description: Description of the manager
            template_format: Format of prompt templates
            add_timestamps: Whether to add timestamps to prompts
            max_history_items: Maximum number of history items to include
            max_examples: Maximum number of examples to include
            config: Additional configuration
            **kwargs: Additional keyword arguments for configuration
        """
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
        self._state.update("prompts", prompts)
        self._state.update("result_cache", {})
        self._state.update("initialized", True)

        # Set metadata
        self._state.set_metadata("component_type", "prompt_manager")
        self._state.set_metadata("creation_time", time.time())
        self._state.set_metadata("prompt_count", len(prompts))

    def process(self, input: Dict[str, Any]) -> PromptResult:
        """
        Process the input context and return a prompt result.

        This is the implementation of the abstract method from BaseComponent.

        Args:
            input: The context to generate a prompt from

        Returns:
            PromptResult with generated prompt and context

        Raises:
            ValueError: If input is invalid
        """
        # For process method, we'll use the first prompt or create a simple one
        prompts = self._state.get("prompts", [])
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
        prompt_text = prompt.generate(input)

        return PromptResult(
            passed=True,
            message="Prompt generated successfully",
            metadata={"prompt_name": prompt.name},
            score=1.0,
            prompt=prompt_text,
            context=input,
        )

    def generate(self, context: Dict[str, Any]) -> List[PromptResult]:
        """
        Generate prompts from context.

        Args:
            context: The context to generate prompts from

        Returns:
            List of PromptResult objects with generated prompts

        Raises:
            ValueError: If context is invalid
        """
        # Handle empty input
        if not context:
            return [
                PromptResult(
                    passed=False,
                    message="Empty context",
                    metadata={"error_type": "empty_context"},
                    score=0.0,
                    issues=["Context is empty"],
                    suggestions=["Provide non-empty context"],
                    prompt="",
                    context={},
                )
            ]

        # Record start time
        start_time = time.time()

        try:
            # Check cache
            cache_key = str(context)[:100]  # Use first 100 chars as key
            cache = self._state.get("result_cache", {})

            if cache_key in cache and self.config.cache_size > 0:
                self._state.set_metadata("cache_hit", True)
                return cache[cache_key]

            # Mark as cache miss
            self._state.set_metadata("cache_hit", False)

            # Get prompts from state
            prompts = self._state.get("prompts", [])
            if not prompts:
                return [
                    PromptResult(
                        passed=False,
                        message="No prompts available",
                        metadata={"error_type": "no_prompts"},
                        score=0.0,
                        issues=["No prompts available"],
                        suggestions=["Add prompts to the manager"],
                        prompt="",
                        context=context,
                    )
                ]

            # Generate prompts
            results = []
            for prompt in prompts:
                try:
                    prompt_text = prompt.generate(context)

                    # Add timestamp if configured
                    if self.config.add_timestamps:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        prompt_text = f"[{timestamp}] {prompt_text}"

                    # Create result
                    result = PromptResult(
                        passed=True,
                        message="Prompt generated successfully",
                        metadata={
                            "prompt_name": prompt.name,
                            "template_format": self.config.template_format,
                        },
                        score=1.0,
                        prompt=prompt_text,
                        context=context,
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )

                    results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = PromptResult(
                        passed=False,
                        message=f"Prompt generation error: {str(e)}",
                        metadata={
                            "prompt_name": prompt.name,
                            "error_type": str(type(e).__name__),
                        },
                        score=0.0,
                        issues=[str(e)],
                        suggestions=["Check context format and try again"],
                        prompt="",
                        context=context,
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
                    results.append(error_result)

                    # Record error
                    self.record_error(e)

            # Update statistics
            for result in results:
                self.update_statistics(result)

            # Cache result if caching is enabled
            if self.config.cache_size > 0:
                # Manage cache size
                if len(cache) >= self.config.cache_size:
                    # Remove oldest entry (simple approach)
                    if cache:
                        oldest_key = next(iter(cache))
                        del cache[oldest_key]

                cache[cache_key] = results
                self._state.update("result_cache", cache)

            return results

        except Exception as e:
            # Record error
            self.record_error(e)
            logger.error(f"Prompt generation error: {str(e)}")

            # Create error result
            return [
                PromptResult(
                    passed=False,
                    message=f"Prompt generation error: {str(e)}",
                    metadata={"error_type": str(type(e).__name__)},
                    score=0.0,
                    issues=[str(e)],
                    suggestions=["Check context format and try again"],
                    prompt="",
                    context=context,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prompt generation usage.

        Returns:
            Dictionary with usage statistics
        """
        # Get base statistics from parent class
        stats = super().get_statistics()

        # Add prompt-specific statistics
        stats.update(
            {
                "cache_size": len(self._state.get("result_cache", {})),
                "prompt_count": len(self._state.get("prompts", [])),
                "template_format": self.config.template_format,
                "add_timestamps": self.config.add_timestamps,
                "max_history_items": self.config.max_history_items,
                "max_examples": self.config.max_examples,
                "cache_enabled": self.config.cache_size > 0,
                "cache_limit": self.config.cache_size,
            }
        )

        return stats

    def clear_cache(self) -> None:
        """Clear the prompt generation result cache."""
        self._state.update("result_cache", {})
        logger.debug(f"Prompt cache cleared for {self.name}")

    def add_prompt(self, prompt: Any) -> None:
        """
        Add a prompt to the manager.

        ## Overview
        This method adds a new prompt to the manager's prompt list.
        The prompt will be used in subsequent generations.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate prompt type
           - Check prompt validity

        2. **Prompt Addition**: Add prompt
           - Add to prompt list
           - Update state
           - Clear cache

        Args:
            prompt: The prompt to add

        Raises:
            ValueError: If the prompt is invalid
            TypeError: If the input type is incorrect

        Examples:
            ```python
            manager = PromptManager(prompts=[])
            new_prompt = BasePrompt(
                name="story_prompt",
                description="Generates story prompts",
                template="Write a story about {topic}"
            )
            manager.add_prompt(new_prompt)
            ```
        """
        # Validate prompt type
        if not isinstance(prompt, BasePrompt):
            raise ValueError(f"Expected BasePrompt instance, got {type(prompt)}")

        # Check for duplicate prompt names
        prompts = self._state.get("prompts", [])
        if any(p.name == prompt.name for p in prompts):
            logger.warning(f"Prompt with name '{prompt.name}' already exists, it will be replaced")
            # Remove existing prompt with same name
            self.remove_prompt(prompt.name)
            # Get updated prompts list
            prompts = self._state.get("prompts", [])

        # Add prompt to the list
        prompts.append(prompt)
        self._state.update("prompts", prompts)

        # Update metadata
        self._state.set_metadata("prompt_count", len(prompts))

        # Clear cache since generation results may change
        self.clear_cache()

        logger.debug(f"Added prompt '{prompt.name}' to prompt manager '{self.name}'")

    def remove_prompt(self, prompt_name: str) -> None:
        """
        Remove a prompt by name.

        ## Overview
        This method removes a prompt from the manager's prompt list
        based on its name. The prompt will no longer be used in subsequent
        generations.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate prompt name
           - Find prompt to remove

        2. **Prompt Removal**: Remove prompt
           - Remove from prompt list
           - Update state
           - Clear cache

        Args:
            prompt_name: The name of the prompt to remove

        Raises:
            ValueError: If the prompt name is invalid or prompt not found

        Examples:
            ```python
            manager = PromptManager(prompts=[...])
            manager.remove_prompt("story_prompt")
            ```
        """
        # Validate input
        if not prompt_name or not isinstance(prompt_name, str):
            raise ValueError(f"Invalid prompt name: {prompt_name}")

        # Find prompt by name
        prompt_to_remove = None
        prompts = self._state.get("prompts", [])
        for prompt in prompts:
            if prompt.name == prompt_name:
                prompt_to_remove = prompt
                break

        if prompt_to_remove is None:
            raise ValueError(f"Prompt not found: {prompt_name}")

        # Remove prompt from list
        prompts.remove(prompt_to_remove)
        self._state.update("prompts", prompts)

        # Update metadata
        self._state.set_metadata("prompt_count", len(prompts))

        # Clear cache since generation results may change
        self.clear_cache()

        logger.debug(f"Removed prompt '{prompt_name}' from prompt manager '{self.name}'")

    def get_prompts(self) -> List[BasePrompt]:
        """
        Get all registered prompts.

        ## Overview
        This method returns a list of all prompts currently
        registered with the manager.

        ## Lifecycle
        1. **Prompt Retrieval**: Get prompts
           - Access prompt list
           - Return prompts

        Returns:
            The list of registered prompts

        Examples:
            ```python
            manager = PromptManager(prompts=[...])
            prompts = manager.get_prompts()
            print(f"Number of prompts: {len(prompts)}")
            ```
        """
        return self._state.get("prompts", [])

    def warm_up(self) -> None:
        """Prepare the prompt manager for use."""
        super().warm_up()

        # Pre-validate prompts
        prompts = self._state.get("prompts", [])
        for prompt in prompts:
            if hasattr(prompt, "warm_up"):
                prompt.warm_up()

        logger.debug(f"Prompt manager '{self.name}' warmed up with {len(prompts)} prompts")

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
            manager = PromptManager(prompts=[])
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
            # Limit history items if configured
            if isinstance(history, list) and len(history) > self.config.max_history_items:
                history = history[-self.config.max_history_items :]
            prompt = self.create_prompt_with_history(prompt, history)
        if context:
            prompt = self.create_prompt_with_context(prompt, context)
        if examples:
            # Limit examples if configured
            if isinstance(examples, list) and len(examples) > self.config.max_examples:
                examples = examples[: self.config.max_examples]
            prompt = self.create_prompt_with_examples(prompt, examples)

        # Add timestamp if configured
        if self.config.add_timestamps:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            prompt = f"[{timestamp}] {prompt}"

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
            manager = PromptManager(prompts=[])
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
            manager = PromptManager(prompts=[])
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


def create_prompt_manager(
    prompts: List[BasePrompt] = None,
    name: str = "prompt_manager",
    description: str = "Prompt manager for Sifaka chains",
    template_format: str = "text",
    add_timestamps: bool = False,
    max_history_items: int = 5,
    max_examples: int = 3,
    cache_size: int = 100,
    **kwargs: Any,
) -> PromptManager:
    """
    Create a prompt manager.

    Args:
        prompts: List of prompts to use for generation
        name: Name of the manager
        description: Description of the manager
        template_format: Format of prompt templates
        add_timestamps: Whether to add timestamps to prompts
        max_history_items: Maximum number of history items to include
        max_examples: Maximum number of examples to include
        cache_size: Size of the prompt cache
        **kwargs: Additional configuration parameters

    Returns:
        Configured PromptManager instance
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
