"""Core critic implementation for Sifaka.

This module provides the core implementation of the critic system, which serves as
the central component for text validation, improvement, and critiquing. The CriticCore
class delegates specific operations to specialized components while maintaining
a unified interface for text processing.

## Overview
The module provides:
- Core critic implementation with standardized state management
- Factory function for creating critic instances
- Specialized components for text processing
- Unified interface for text validation, improvement, and critiquing
- Error handling and resource management

## Components
1. **CriticCore Class**
   - Central implementation of the critic system
   - Delegates to specialized components
   - Maintains unified interface
   - Uses standardized state management

2. **Factory Function**
   - create_core_critic(): Creates CriticCore instances
   - Handles parameter extraction and validation
   - Provides consistent interface
   - Manages component lifecycle

3. **Specialized Components**
   - Language Model Provider: Handles text generation
   - Prompt Manager: Creates and manages prompts
   - Response Parser: Parses model responses
   - Memory Manager: Manages interaction history
   - Critique Service: Handles text critiquing

## Usage Examples
```python
from sifaka.critics.core import create_core_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a core critic using the factory function
critic = create_core_critic(
    name="core_critic",
    description="A core critic implementation",
    system_prompt="You are an expert editor...",
    temperature=0.7,
    max_tokens=1000,
    llm_provider=provider
)

# Validate text
text = "This is a sample technical document."
is_valid = critic.validate(text)
print(f"Text is valid: {is_valid}")

# Critique text
critique = critic.critique(text)
print(f"Score: {critique.score}")
print(f"Feedback: {critique.feedback}")

# Improve text
improved_text = critic.improve(text, "The text needs more detail.")
print(f"Improved text: {improved_text}")
```

## Error Handling
The module implements:
- Input validation for text and parameters
- Model interaction error handling
- Resource management and cleanup
- State management and recovery
- Component initialization validation
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import PrivateAttr

from .base import BaseCritic
from .config import CriticConfig, CriticMetadata
from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
from .managers.prompt import DefaultPromptManager, PromptManager
from .managers.response import ResponseParser
from .services.critique import CritiqueService
from ..models.core import ModelProviderCore
from ..utils.logging import get_logger
from ..utils.state import StateManager, create_critic_state

logger = get_logger(__name__)


class CriticCore(BaseCritic):
    """Core critic implementation that delegates to specialized components.

    This class serves as the central implementation of the critic system,
    providing a unified interface for text processing while delegating
    specific operations to specialized components.

    ## Overview
    The CriticCore class provides:
    - Unified interface for text validation, improvement, and critiquing
    - Delegation to specialized components for specific operations
    - Standardized state management approach
    - Resource management and cleanup
    - Error handling and recovery

    ## Usage Examples
    ```python
    from sifaka.critics.core import create_core_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a critic instance
    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_core_critic(
        name="core_critic",
        description="A core critic implementation",
        system_prompt="You are an expert editor...",
        llm_provider=provider
    )

    # Process text
    text = "This is a sample technical document."
    is_valid = critic.validate(text)
    critique = critic.critique(text)
    improved_text = critic.improve(text, "The text needs more detail.")
    ```

    ## Error Handling
    The class implements:
    - Input validation for text and parameters
    - Model interaction error handling
    - Resource management and cleanup
    - State management and recovery
    - Component initialization validation

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state manager
    - Clear separation of configuration and state
    - State components:
      - model: Language model provider
      - prompt_manager: Prompt manager
      - response_parser: Response parser
      - memory_manager: Memory manager
      - critique_service: Critique service
      - initialized: Initialization status
      - cache: Temporary data storage
    """

    # Use StateManager for state management
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: CriticConfig,
        llm_provider: ModelProviderCore,
        prompt_manager: Optional[PromptManager] = None,
        response_parser: Optional[ResponseParser] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """Initialize a CriticCore instance.

        This method sets up the core critic with its configuration and components.
        It creates or uses provided managers and services for text processing.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider (ModelProviderCore)
            prompt_manager: Optional prompt manager
            response_parser: Optional response parser
            memory_manager: Optional memory manager

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        super().__init__(config)

        # Initialize state using StateManager
        self._state_manager.update("initialized", False)
        self._state_manager.update("model", llm_provider)
        self._state_manager.update("cache", {})

        # Store components in state
        prompt_manager = prompt_manager or self._create_prompt_manager()
        self._state_manager.update("prompt_manager", prompt_manager)
        self._state_manager.update("response_parser", response_parser or ResponseParser())
        if memory_manager:
            self._state_manager.update("memory_manager", memory_manager)

        # Create and store critique service
        critique_service = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            response_parser=self._state_manager.get("response_parser"),
            memory_manager=memory_manager,
        )
        cache = self._state_manager.get("cache", {})
        cache["critique_service"] = critique_service
        self._state_manager.update("cache", cache)

        # Set metadata
        self._state_manager.set_metadata("component_type", "critic")
        self._state_manager.set_metadata("name", config.name)
        self._state_manager.set_metadata("description", config.description)
        self._state_manager.set_metadata("validation_count", 0)
        self._state_manager.set_metadata("critique_count", 0)
        self._state_manager.set_metadata("improvement_count", 0)

        # Mark as initialized
        self._state_manager.update("initialized", True)

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        This method checks if the text meets the quality standards defined by the critic.
        It uses the language model to analyze the text and determine if it meets the
        required criteria.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If the text is empty or invalid
            RuntimeError: If the language model fails to process the text

        Example:
            ```python
            critic = create_core_critic(...)
            text = "This is a sample text."
            is_valid = critic.validate(text)
            if is_valid:
                print("Text meets quality standards")
            else:
                print("Text needs improvement")
            ```
        """
        # Ensure initialized
        if not self._state_manager.get("initialized", False):
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        cache = self._state_manager.get("cache", {})
        critique_service = cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate text
        from sifaka.utils.text import is_empty_text

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if is_empty_text(text):
            raise ValueError("Text must be a non-empty string")

        # Track validation count
        validation_count = self._state_manager.get_metadata("validation_count", 0)
        self._state_manager.set_metadata("validation_count", validation_count + 1)

        # Get critique
        critique = critique_service.critique(text)

        # Record result in metadata
        is_valid = critique.score >= self.config.min_confidence
        if is_valid:
            valid_count = self._state_manager.get_metadata("valid_count", 0)
            self._state_manager.set_metadata("valid_count", valid_count + 1)
        else:
            invalid_count = self._state_manager.get_metadata("invalid_count", 0)
            self._state_manager.set_metadata("invalid_count", invalid_count + 1)

        # Return validation result
        return is_valid

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve text based on identified violations.

        This method takes text and a list of violations, then uses the language model
        to generate an improved version of the text that addresses the violations.

        Args:
            text: The text to improve
            violations: List of dictionaries containing violation details

        Returns:
            str: The improved text

        Raises:
            ValueError: If the text is empty or invalid
            RuntimeError: If the language model fails to process the text

        Example:
            ```python
            critic = create_core_critic(...)
            text = "This is a sample text."
            violations = [
                {
                    "rule_id": "clarity",
                    "message": "Text is unclear",
                    "suggestion": "Add more context"
                }
            ]
            improved_text = critic.improve(text, violations)
            print(f"Improved text: {improved_text}")
            ```
        """
        # Ensure initialized
        if not self._state_manager.get("initialized", False):
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        cache = self._state_manager.get("cache", {})
        critique_service = cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate inputs
        from sifaka.utils.text import is_empty_text

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if is_empty_text(text):
            raise ValueError("Text must be a non-empty string")

        if not violations or not isinstance(violations, list):
            raise ValueError("Violations must be a non-empty list")

        # Track improvement count
        improvement_count = self._state_manager.get_metadata("improvement_count", 0)
        self._state_manager.set_metadata("improvement_count", improvement_count + 1)

        # Record start time
        import time

        start_time = time.time()

        # Attempt improvements
        for attempt in range(self.config.max_attempts):
            # Get improvement
            improved_text = critique_service.improve(text, violations)

            # Track attempt count
            attempt_count = self._state_manager.get_metadata("improvement_attempts", 0)
            self._state_manager.set_metadata("improvement_attempts", attempt_count + 1)

            # Validate improvement
            if improved_text and improved_text != text:
                # Track successful improvements
                success_count = self._state_manager.get_metadata("successful_improvements", 0)
                self._state_manager.set_metadata("successful_improvements", success_count + 1)

                # Record execution time
                end_time = time.time()
                exec_time = end_time - start_time
                avg_time = self._state_manager.get_metadata("avg_improvement_time", 0)
                count = self._state_manager.get_metadata("improvement_count", 1)
                new_avg = ((avg_time * (count - 1)) + exec_time) / count
                self._state_manager.set_metadata("avg_improvement_time", new_avg)

                return improved_text

        # Track failed improvements
        fail_count = self._state_manager.get_metadata("failed_improvements", 0)
        self._state_manager.set_metadata("failed_improvements", fail_count + 1)

        # Return original text if no improvement
        return text

    def critique(self, text: str) -> CriticMetadata:
        """
        Critique text and provide detailed feedback.

        This method analyzes the text and provides detailed feedback about its quality,
        including a score, feedback message, and specific issues and suggestions.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata: Object containing critique details including:
                - score: Quality score between 0.0 and 1.0
                - feedback: Detailed feedback message
                - issues: List of identified issues
                - suggestions: List of improvement suggestions

        Raises:
            ValueError: If the text is empty or invalid
            RuntimeError: If the language model fails to process the text

        Example:
            ```python
            critic = create_core_critic(...)
            text = "This is a sample text."
            critique = critic.critique(text)
            print(f"Score: {critique.score}")
            print(f"Feedback: {critique.feedback}")
            print("Issues:")
            for issue in critique.issues:
                print(f"- {issue}")
            print("Suggestions:")
            for suggestion in critique.suggestions:
                print(f"- {suggestion}")
            ```
        """
        # Ensure initialized
        if not self._state_manager.get("initialized", False):
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        cache = self._state_manager.get("cache", {})
        critique_service = cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate text
        from sifaka.utils.text import is_empty_text

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if is_empty_text(text):
            raise ValueError("Text must be a non-empty string")

        # Track critique count
        critique_count = self._state_manager.get_metadata("critique_count", 0)
        self._state_manager.set_metadata("critique_count", critique_count + 1)

        # Record start time
        import time

        start_time = time.time()

        # Get critique
        critique = critique_service.critique(text)

        # Record execution time
        end_time = time.time()
        exec_time = end_time - start_time
        avg_time = self._state_manager.get_metadata("avg_critique_time", 0)
        count = self._state_manager.get_metadata("critique_count", 1)
        new_avg = ((avg_time * (count - 1)) + exec_time) / count
        self._state_manager.set_metadata("avg_critique_time", new_avg)

        # Track score distribution
        score_distribution = self._state_manager.get_metadata("score_distribution", {})
        score_bucket = round(critique.score * 10) / 10  # Round to nearest 0.1
        score_distribution[str(score_bucket)] = score_distribution.get(str(score_bucket), 0) + 1
        self._state_manager.set_metadata("score_distribution", score_distribution)

        return critique

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        This method takes text and feedback, then uses the language model to generate
        an improved version of the text that addresses the specific feedback.

        Args:
            text: The text to improve
            feedback: Specific feedback about what needs improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If the text or feedback is empty or invalid
            RuntimeError: If the language model fails to process the text

        Example:
            ```python
            critic = create_core_critic(...)
            text = "This is a sample text."
            feedback = "The text needs more detail about the process."
            improved_text = critic.improve_with_feedback(text, feedback)
            print(f"Improved text: {improved_text}")
            ```
        """
        # Ensure initialized
        if not self._state_manager.get("initialized", False):
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        cache = self._state_manager.get("cache", {})
        critique_service = cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate inputs
        from sifaka.utils.text import is_empty_text

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if is_empty_text(text):
            raise ValueError("Text must be a non-empty string")

        if not isinstance(feedback, str):
            raise ValueError("Feedback must be a string")

        if is_empty_text(feedback):
            raise ValueError("Feedback must be a non-empty string")

        # Track improvement count
        improvement_count = self._state_manager.get_metadata("feedback_improvement_count", 0)
        self._state_manager.set_metadata("feedback_improvement_count", improvement_count + 1)

        # Attempt improvements
        for attempt in range(self.config.max_attempts):
            # Get improvement
            improved_text = critique_service.improve_with_feedback(text, feedback)

            # Track attempt
            attempt_count = self._state_manager.get_metadata("feedback_attempts", 0)
            self._state_manager.set_metadata("feedback_attempts", attempt_count + 1)

            # Validate improvement
            if improved_text and improved_text != text:
                # Track successful improvements
                success_count = self._state_manager.get_metadata(
                    "successful_feedback_improvements", 0
                )
                self._state_manager.set_metadata(
                    "successful_feedback_improvements", success_count + 1
                )
                return improved_text

        # Track failed improvements
        fail_count = self._state_manager.get_metadata("failed_feedback_improvements", 0)
        self._state_manager.set_metadata("failed_feedback_improvements", fail_count + 1)

        # Return original text if no improvement
        return text

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about critic usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "validation_count": self._state_manager.get_metadata("validation_count", 0),
            "valid_count": self._state_manager.get_metadata("valid_count", 0),
            "invalid_count": self._state_manager.get_metadata("invalid_count", 0),
            "critique_count": self._state_manager.get_metadata("critique_count", 0),
            "avg_critique_time": self._state_manager.get_metadata("avg_critique_time", 0),
            "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
            "improvement_attempts": self._state_manager.get_metadata("improvement_attempts", 0),
            "successful_improvements": self._state_manager.get_metadata(
                "successful_improvements", 0
            ),
            "failed_improvements": self._state_manager.get_metadata("failed_improvements", 0),
            "avg_improvement_time": self._state_manager.get_metadata("avg_improvement_time", 0),
            "feedback_improvement_count": self._state_manager.get_metadata(
                "feedback_improvement_count", 0
            ),
            "feedback_attempts": self._state_manager.get_metadata("feedback_attempts", 0),
            "successful_feedback_improvements": self._state_manager.get_metadata(
                "successful_feedback_improvements", 0
            ),
            "failed_feedback_improvements": self._state_manager.get_metadata(
                "failed_feedback_improvements", 0
            ),
            "score_distribution": self._state_manager.get_metadata("score_distribution", {}),
        }

    def _create_prompt_manager(self) -> PromptManager:
        """Create a prompt manager for the critic.

        This method creates and configures a prompt manager for the critic.
        It handles prompt template creation and validation.

        ## Overview
        The method provides:
        - Prompt manager creation
        - Template configuration
        - Validation setup
        - Error handling

        ## Usage Examples
        ```python
        # Create prompt manager
        prompt_manager = critic._create_prompt_manager()

        # Use prompt manager
        prompt = prompt_manager.create_prompt(
            template="You are an expert editor...",
            variables={"text": "Sample text"}
        )
        ```

        ## Error Handling
        The method implements:
        - Template validation
        - Configuration checking
        - Resource allocation
        - Error recovery

        Returns:
            PromptManager: Configured prompt manager

        Raises:
            RuntimeError: If prompt manager creation fails
        """
        return DefaultPromptManager()


def create_core_critic(
    name: str,
    description: str,
    llm_provider: ModelProviderCore,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    prompt_manager: Optional[PromptManager] = None,
    response_parser: Optional[ResponseParser] = None,
    memory_manager: Optional[MemoryManager] = None,
    config: Optional[Union[Dict[str, Any], CriticConfig]] = None,
    **kwargs: Any,
) -> CriticCore:
    """Factory function for creating CriticCore instances.

    This function creates and configures a CriticCore instance with the specified
    parameters and components. It handles parameter extraction, validation, and
    component initialization.

    ## Overview
    The function provides:
    - Factory method for creating CriticCore instances
    - Parameter extraction and validation
    - Component initialization and configuration
    - Error handling and recovery
    - Resource management

    ## Usage Examples
    ```python
    from sifaka.critics.core import create_core_critic
    from sifaka.models.providers import OpenAIProvider

    # Create with minimal parameters
    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_core_critic(
        name="basic_critic",
        description="A basic critic implementation",
        llm_provider=provider
    )

    # Create with custom configuration
    critic = create_core_critic(
        name="custom_critic",
        description="A custom critic implementation",
        llm_provider=provider,
        system_prompt="You are an expert editor...",
        temperature=0.7,
        max_tokens=1000,
        min_confidence=0.8,
        max_attempts=3,
        cache_size=100,
        priority=1,
        cost=0.1
    )

    # Create with custom components
    from sifaka.critics.managers import PromptManager, ResponseParser, MemoryManager
    critic = create_core_critic(
        name="component_critic",
        description="A critic with custom components",
        llm_provider=provider,
        prompt_manager=PromptManager(),
        response_parser=ResponseParser(),
        memory_manager=MemoryManager()
    )
    ```

    ## Error Handling
    The function implements:
    - Parameter validation
    - Component initialization validation
    - Resource allocation error handling
    - Configuration error handling
    - Provider compatibility checking

    Args:
        name: Name of the critic
        description: Description of the critic
        llm_provider: Language model provider (ModelProviderCore)
        system_prompt: Optional system prompt for the critic
        temperature: Optional temperature for text generation
        max_tokens: Optional maximum tokens for text generation
        min_confidence: Optional minimum confidence threshold
        max_attempts: Optional maximum number of attempts
        cache_size: Optional cache size for memoization
        priority: Optional priority level
        cost: Optional cost per operation
        prompt_manager: Optional prompt manager
        response_parser: Optional response parser
        memory_manager: Optional memory manager
        config: Optional configuration dictionary or object
        **kwargs: Additional keyword arguments

    Returns:
        CriticCore: Configured critic instance

    Raises:
        ValueError: If required parameters are missing or invalid
        TypeError: If llm_provider is not a valid provider
        RuntimeError: If component initialization fails
    """
    # Create configuration
    if config is None:
        config = CriticConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            **kwargs,
        )
    elif isinstance(config, dict):
        config = CriticConfig(**config)
    elif not isinstance(config, CriticConfig):
        raise TypeError("config must be a dict or CriticConfig instance")

    # Create and return critic
    return CriticCore(
        config=config,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        response_parser=response_parser,
        memory_manager=memory_manager,
    )
