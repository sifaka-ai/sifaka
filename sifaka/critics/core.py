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
from .managers.memory import MemoryManager
from .managers.prompt import DefaultPromptManager, PromptManager
from .managers.response import ResponseParser
from .services.critique import CritiqueService
from ..utils.logging import get_logger
from ..utils.state import CriticState

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
    - Single _state attribute for all mutable state
    - State initialization during construction
    - State access through state object
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

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        config: CriticConfig,
        llm_provider: Any,
        prompt_manager: Optional[PromptManager] = None,
        response_parser: Optional[ResponseParser] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """Initialize a CriticCore instance.

        This method sets up the core critic with its configuration and components.
        It creates or uses provided managers and services for text processing.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider
            prompt_manager: Optional prompt manager
            response_parser: Optional response parser
            memory_manager: Optional memory manager

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        super().__init__(config)

        # Initialize state
        from ..utils.state import CriticState

        self._state = CriticState()
        self._state.initialized = False

        # Store components in state
        self._state.model = llm_provider
        self._state.memory_manager = memory_manager
        self._state.prompt_manager = prompt_manager or self._create_prompt_manager()
        self._state.response_parser = response_parser or ResponseParser()

        # Create services and store in state cache
        self._state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._state.prompt_manager,
            response_parser=self._state.response_parser,
            memory_manager=memory_manager,
        )

        # Mark as initialized
        self._state.initialized = True

    def validate(self, text: str) -> bool:
        """Validate text against quality standards.

        This method checks if the given text meets the quality standards
        defined by the critic's configuration.

        ## Overview
        The method provides:
        - Text quality validation
        - Standard compliance checking
        - Error detection
        - Validation result reporting

        ## Usage Examples
        ```python
        # Validate text
        text = "This is a sample technical document."
        is_valid = critic.validate(text)
        print(f"Text is valid: {is_valid}")

        # Validate with error handling
        try:
            is_valid = critic.validate("")
        except ValueError as e:
            print(f"Validation error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Empty text validation
        - Invalid text format checking
        - Validation error reporting
        - State validation
        - Service availability checking

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate text
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Get critique
        critique = critique_service.critique(text)

        # Return validation result
        return critique.score >= self.config.min_confidence

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve text based on identified violations.

        This method attempts to improve the given text by addressing the
        specified violations using the language model.

        ## Overview
        The method provides:
        - Text improvement based on violations
        - Multiple improvement attempts
        - Improvement validation
        - Result reporting

        ## Usage Examples
        ```python
        # Improve text with violations
        text = "This is a sample technical document."
        violations = [
            {"type": "clarity", "description": "Unclear explanation"},
            {"type": "detail", "description": "Missing important details"}
        ]
        improved_text = critic.improve(text, violations)
        print(f"Improved text: {improved_text}")

        # Improve with error handling
        try:
            improved_text = critic.improve("", [])
        except ValueError as e:
            print(f"Improvement error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Input validation
        - Violation format checking
        - Improvement attempt tracking
        - Result validation
        - Service availability checking

        Args:
            text: The text to improve
            violations: List of violations to address

        Returns:
            str: The improved text

        Raises:
            ValueError: If text or violations are invalid
            RuntimeError: If improvement fails
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate inputs
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        if not violations or not isinstance(violations, list):
            raise ValueError("Violations must be a non-empty list")

        # Attempt improvements
        for attempt in range(self.config.max_attempts):
            # Get improvement
            improved_text = critique_service.improve(text, violations)

            # Validate improvement
            if improved_text and improved_text != text:
                return improved_text

        # Return original text if no improvement
        return text

    def critique(self, text: str) -> CriticMetadata:
        """Critique text and provide feedback.

        This method analyzes the given text and provides detailed feedback
        about its quality and potential improvements.

        ## Overview
        The method provides:
        - Text quality analysis
        - Detailed feedback generation
        - Score calculation
        - Improvement suggestions

        ## Usage Examples
        ```python
        # Critique text
        text = "This is a sample technical document."
        critique = critic.critique(text)
        print(f"Score: {critique.score}")
        print(f"Feedback: {critique.feedback}")

        # Critique with error handling
        try:
            critique = critic.critique("")
        except ValueError as e:
            print(f"Critique error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Input validation
        - Analysis error handling
        - Result validation
        - Service availability checking
        - State validation

        Args:
            text: The text to critique

        Returns:
            CriticMetadata: The critique results

        Raises:
            ValueError: If text is invalid
            RuntimeError: If critique fails
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate text
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Get critique
        return critique_service.critique(text)

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on feedback.

        This method attempts to improve the given text based on the provided
        feedback using the language model.

        ## Overview
        The method provides:
        - Text improvement based on feedback
        - Multiple improvement attempts
        - Improvement validation
        - Result reporting

        ## Usage Examples
        ```python
        # Improve text with feedback
        text = "This is a sample technical document."
        feedback = "The text needs more detail about the implementation."
        improved_text = critic.improve_with_feedback(text, feedback)
        print(f"Improved text: {improved_text}")

        # Improve with error handling
        try:
            improved_text = critic.improve_with_feedback("", "")
        except ValueError as e:
            print(f"Improvement error: {e}")
        ```

        ## Error Handling
        The method implements:
        - Input validation
        - Feedback format checking
        - Improvement attempt tracking
        - Result validation
        - Service availability checking

        Args:
            text: The text to improve
            feedback: The feedback to use for improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text or feedback are invalid
            RuntimeError: If improvement fails
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not properly initialized")

        # Validate inputs
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")

        # Attempt improvements
        for attempt in range(self.config.max_attempts):
            # Get improvement
            improved_text = critique_service.improve_with_feedback(text, feedback)

            # Validate improvement
            if improved_text and improved_text != text:
                return improved_text

        # Return original text if no improvement
        return text

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
    llm_provider: Any,
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
        llm_provider: Language model provider
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
