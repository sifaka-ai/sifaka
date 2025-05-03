"""Core critic implementation for Sifaka.

This module provides the core implementation of the critic system, which serves as
the central component for text validation, improvement, and critiquing. The CriticCore
class delegates specific operations to specialized components while maintaining
a unified interface for text processing.

This implementation uses the standardized state management approach, which provides
consistent state handling across all components in the Sifaka framework.

## Factory Function Pattern

This module follows the standard Sifaka factory function pattern:

1. The `create_core_critic()` factory function creates a CriticCore instance:
   - Takes configuration parameters directly
   - Creates a CriticConfig object
   - Instantiates and returns a CriticCore

2. Factory function handles parameter extraction and validation:
   - Extract parameters from function arguments
   - Create configuration objects as needed
   - Instantiate and return the component

3. Factory function provides a consistent interface:
   - Clear parameter names matching component attributes
   - Consistent error handling
   - Standardized return types

## Component Lifecycle

### CriticCore Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Manager initialization
   - Service setup
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Text improvement
   - Text critiquing
   - Feedback processing

3. **Cleanup Phase**
   - Resource release
   - State cleanup
   - Error recovery

### Component Interactions

1. **Language Model Provider**
   - Handles text generation
   - Manages model responses
   - Controls model parameters

2. **Prompt Manager**
   - Creates specialized prompts
   - Manages prompt templates
   - Validates prompt formats

3. **Response Parser**
   - Parses model responses
   - Validates response formats
   - Extracts structured data

4. **Memory Manager**
   - Stores past interactions
   - Retrieves relevant context
   - Manages memory buffer

## Error Handling

1. **Input Validation Errors**
   - Empty or invalid text
   - Invalid feedback format
   - Invalid violation format

2. **Model Interaction Errors**
   - Provider connection failures
   - Response parsing errors
   - Format validation failures

3. **Resource Management Errors**
   - Memory allocation failures
   - Service initialization errors
   - Manager setup failures

## Examples

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
"""

from typing import Any, Dict, List, Optional, ClassVar, Union

from pydantic import PrivateAttr

from .base import BaseCritic
from .models import CriticConfig, CriticMetadata
from .managers.memory import MemoryManager
from .managers.prompt import DefaultPromptManager, PromptManager
from .managers.response import ResponseParser
from .services.critique import CritiqueService
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CriticCore(BaseCritic):
    """Core critic implementation that delegates to specialized components.

    This class serves as the central implementation of the critic system,
    providing a unified interface for text processing while delegating
    specific operations to specialized components.

    This implementation uses the standardized state management approach with
    a single _state attribute that manages all mutable state.

    ## State Management

    The CriticCore uses a standardized state management approach:

    1. **State Initialization**
       - Creates a state object using create_critic_state()
       - Stores all mutable state in the state object
       - Accesses state through the state manager

    2. **State Access**
       - Uses state.get_state() to access the state object
       - Updates state through the state object
       - Maintains clear separation between configuration and state

    3. **State Components**
       - model: The language model provider
       - prompt_manager: The prompt manager
       - response_parser: The response parser
       - memory_manager: The memory manager
       - critique_service: The critique service
       - initialized: Whether the critic is initialized
       - cache: A cache for storing temporary data
    """

    # State management
    _initialized = PrivateAttr(default=False)

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
        self._initialized = False

        # Store components as attributes
        self._model = llm_provider
        self._memory_manager = memory_manager
        self._prompt_manager = prompt_manager or self._create_prompt_manager()
        self._response_parser = response_parser or ResponseParser()

        # Create services
        self._critique_service = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._prompt_manager,
            response_parser=self._response_parser,
            memory_manager=memory_manager,
        )

        # Mark as initialized
        self._initialized = True

    def validate(self, text: str) -> bool:
        """Validate text against quality standards.

        This method checks if the given text meets the quality standards
        defined by the critic's configuration.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        return self._critique_service.validate(text)

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve text based on violations.

        This method enhances the given text by addressing the specified
        violations and applying improvements.

        Args:
            text: The text to improve
            violations: List of violations to address

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or violations are invalid
            RuntimeError: If improvement fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        return self._critique_service.improve(text, violations)

    def critique(self, text: str) -> CriticMetadata:
        """Critique text and provide feedback.

        This method analyzes the given text and provides detailed feedback
        about its quality and potential improvements.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata: Structured feedback about the text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        result = self._critique_service.critique(text)

        # Convert dictionary to CriticMetadata if needed
        if isinstance(result, dict):
            # Ensure required fields are present
            if not result:
                result = {
                    "score": 0.0,
                    "feedback": "",
                    "issues": [],
                    "suggestions": [],
                }
            elif "score" not in result:
                result["score"] = 0.0
            elif "feedback" not in result:
                result["feedback"] = ""

            return CriticMetadata(**result)
        return result

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on feedback.

        This method enhances the given text using the provided feedback
        to guide the improvement process.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or feedback is invalid
            RuntimeError: If improvement fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        return self._critique_service.improve(text, feedback)

    async def avalidate(self, text: str) -> bool:
        """Asynchronously validate text against quality standards.

        This method performs asynchronous validation of text against
        the quality standards defined by the critic's configuration.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        return await self._critique_service.avalidate(text)

    async def acritique(self, text: str) -> CriticMetadata:
        """Asynchronously critique text and provide feedback.

        This method performs asynchronous analysis of the given text
        and provides detailed feedback about its quality.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata: Structured feedback about the text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        result = await self._critique_service.acritique(text)

        # Convert dictionary to CriticMetadata if needed
        if isinstance(result, dict):
            # Ensure required fields are present
            if not result:
                result = {
                    "score": 0.0,
                    "feedback": "",
                    "issues": [],
                    "suggestions": [],
                }
            elif "score" not in result:
                result["score"] = 0.0
            elif "feedback" not in result:
                result["feedback"] = ""

            return CriticMetadata(**result)
        return result

    async def aimprove(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Asynchronously improve text based on violations.

        This method performs asynchronous improvement of the given text
        by addressing the specified violations.

        Args:
            text: The text to improve
            violations: List of violations to address

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or violations are invalid
            RuntimeError: If improvement fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        return await self._critique_service.aimprove(text, violations)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on feedback.

        This method performs asynchronous improvement of the given text
        using the provided feedback to guide the process.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or feedback is invalid
            RuntimeError: If improvement fails
        """
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError("CriticCore not properly initialized")

        # Delegate to critique service
        return await self._critique_service.aimprove(text, feedback)

    def _create_prompt_manager(self) -> PromptManager:
        """Create a default prompt manager.

        This method creates a default prompt manager instance using
        the critic's configuration.

        Returns:
            PromptManager: A configured prompt manager instance

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If manager creation fails
        """
        # Create new prompt manager
        return DefaultPromptManager(self.config)


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
    """
    Create a core critic.

    This factory function creates a configured CriticCore instance.
    It provides a standardized way to create critics with various configurations.

    Args:
        name: Name of the critic
        description: Description of what this critic does
        llm_provider: Language model provider to use
        system_prompt: System prompt for the language model
        temperature: Temperature for the language model
        max_tokens: Maximum tokens for the language model
        min_confidence: Minimum confidence threshold for valid critiques
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the critique result cache (0 to disable)
        priority: Priority of the critic (higher values = higher priority)
        cost: Computational cost of using this critic
        prompt_manager: Optional prompt manager to use
        response_parser: Optional response parser to use
        memory_manager: Optional memory manager to use
        config: Optional pre-configured CriticConfig or dict
        **kwargs: Additional keyword arguments for the config

    Returns:
        CriticCore: The created critic

    Examples:
        ```python
        from sifaka.critics.core import create_core_critic
        from sifaka.models.openai import create_openai_provider

        # Create a critic with default settings
        critic = create_core_critic(
            name="content_critic",
            description="Evaluates content quality",
            llm_provider=create_openai_provider(api_key="your-api-key")
        )

        # Create a critic with custom settings
        critic = create_core_critic(
            name="advanced_critic",
            description="Advanced content evaluation",
            system_prompt="You are an expert editor...",
            temperature=0.7,
            max_tokens=1000,
            min_confidence=0.8,
            llm_provider=create_openai_provider(api_key="your-api-key")
        )

        # Use the critic
        is_valid = critic.validate("This is a sample text.")
        feedback = critic.critique("This is a sample text.")
        improved = critic.improve_with_feedback(
            "This is a sample text.",
            "Add more details about the topic."
        )
        ```
    """
    # Try to use standardize_critic_config if available
    try:
        from sifaka.utils.config import standardize_critic_config

        # If standardize_critic_config is available, use it
        critic_config = standardize_critic_config(
            config=config,
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
    except (ImportError, AttributeError):
        # Extract config parameters
        config_params = {}
        if system_prompt is not None:
            config_params["system_prompt"] = system_prompt
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_tokens is not None:
            config_params["max_tokens"] = max_tokens
        if min_confidence is not None:
            config_params["min_confidence"] = min_confidence
        if max_attempts is not None:
            config_params["max_attempts"] = max_attempts
        if cache_size is not None:
            config_params["cache_size"] = cache_size
        if priority is not None:
            config_params["priority"] = priority
        if cost is not None:
            config_params["cost"] = cost

        # Add any remaining config parameters
        config_params.update(kwargs)

        # Create config
        if isinstance(config, CriticConfig):
            critic_config = config
        elif isinstance(config, dict):
            critic_config = CriticConfig(**config)
        else:
            critic_config = CriticConfig(
                name=name,
                description=description,
                **config_params,
            )

    # Create critic
    return CriticCore(
        config=critic_config,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        response_parser=response_parser,
        memory_manager=memory_manager,
    )
